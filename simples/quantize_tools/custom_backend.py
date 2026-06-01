from __future__ import annotations

import operator

import torch
import torch.ao.nn.intrinsic as nni
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import MinMaxObserver, QConfig, QConfigMapping
from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
)
from torch.ao.quantization.fuser_method_mappings import (
    fuse_conv_bn,
    fuse_conv_bn_relu,
    fuse_linear_bn,
)
from torch.ao.quantization.utils import MatchAllNode


def _conv_bn_add_relu_root_left(pattern):
    _relu, add_pattern = pattern
    _add, bn_conv, _shortcut = add_pattern
    _bn, conv = bn_conv
    return conv


def _conv_bn_add_relu_extra_left(pattern):
    _relu, add_pattern = pattern
    _add, _bn_conv, shortcut = add_pattern
    return [shortcut]


def _conv_add_relu_root_left(pattern):
    _relu, add_pattern = pattern
    _add, conv, _shortcut = add_pattern
    return conv


def _conv_add_relu_extra_left(pattern):
    _relu, add_pattern = pattern
    _add, _conv, shortcut = add_pattern
    return [shortcut]


def _conv_bn_add_relu_root_right(pattern):
    _relu, add_pattern = pattern
    _add, _shortcut, bn_conv = add_pattern
    _bn, conv = bn_conv
    return conv


def _conv_bn_add_relu_extra_right(pattern):
    _relu, add_pattern = pattern
    _add, shortcut, _bn_conv = add_pattern
    return [shortcut]


def _conv_add_relu_root_right(pattern):
    _relu, add_pattern = pattern
    _add, _shortcut, conv = add_pattern
    return conv


def _conv_add_relu_extra_right(pattern):
    _relu, add_pattern = pattern
    _add, shortcut, _conv = add_pattern
    return [shortcut]


def _as_relu_module(relu):
    if isinstance(relu, nn.Module):
        return relu
    return nn.ReLU()


def _fuse_conv_bn_add_relu_left(is_qat, relu, add_pattern):
    add, bn_conv, shortcut = add_pattern
    bn, conv = bn_conv
    fused_conv = fuse_conv_bn(is_qat, conv, bn)
    return nni.ConvAddReLU2d(fused_conv, add, _as_relu_module(relu))


def _fuse_conv_add_relu_left(is_qat, relu, add_pattern):
    add, conv, shortcut = add_pattern
    return nni.ConvAddReLU2d(conv, add, _as_relu_module(relu))


def _fuse_conv_bn_add_relu_right(is_qat, relu, add_pattern):
    add, shortcut, bn_conv = add_pattern
    bn, conv = bn_conv
    fused_conv = fuse_conv_bn(is_qat, conv, bn)
    return nni.ConvAddReLU2d(fused_conv, add, _as_relu_module(relu))


def _fuse_conv_add_relu_right(is_qat, relu, add_pattern):
    add, shortcut, conv = add_pattern
    return nni.ConvAddReLU2d(conv, add, _as_relu_module(relu))


def _fuse_linear_bn_relu(is_qat, linear, bn, relu):
    fused_linear = fuse_linear_bn(is_qat, linear, bn)
    return nni.LinearReLU(fused_linear, _as_relu_module(relu))


def _sequential_linear_relu(is_qat, linear, relu):
    return nni.LinearReLU(linear, _as_relu_module(relu))


def _sequential_conv_relu(is_qat, conv, relu):
    return nni.ConvReLU2d(conv, _as_relu_module(relu))


def build_bked_qconfig_mapping(symmetric: bool = True) -> QConfigMapping:
    qscheme_act = torch.per_tensor_symmetric if symmetric else torch.per_tensor_affine
    my_qconfig = QConfig(
        activation=MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=qscheme_act,
        ),
        weight=MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
        ),
    )

    return QConfigMapping().set_global(my_qconfig)


def build_bked_backend_config() -> BackendConfig:
    dtype_config = DTypeConfig(
        input_dtype=torch.qint8,
        output_dtype=torch.qint8,
        weight_dtype=torch.qint8,
        bias_dtype=torch.float32,
    )
    observation = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    configs: list[BackendPatternConfig] = []

    for add_op in (operator.add, torch.add):
        for relu_op in (nn.ReLU, F.relu, torch.relu):
            configs.extend(
                [
                    # relu(add(bn(conv(y)), shortcut))
                    BackendPatternConfig()
                    ._set_pattern_complex_format(
                        (relu_op, (add_op, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode))
                    )
                    .set_observation_type(observation)
                    .set_dtype_configs([dtype_config])
                    .set_fuser_method(_fuse_conv_bn_add_relu_left)
                    ._set_root_node_getter(_conv_bn_add_relu_root_left)
                    ._set_extra_inputs_getter(_conv_bn_add_relu_extra_left)
                    .set_fused_module(nni.ConvAddReLU2d),

                    # relu(add(shortcut, bn(conv(y))))
                    BackendPatternConfig()
                    ._set_pattern_complex_format(
                        (relu_op, (add_op, MatchAllNode, (nn.BatchNorm2d, nn.Conv2d)))
                    )
                    .set_observation_type(observation)
                    .set_dtype_configs([dtype_config])
                    .set_fuser_method(_fuse_conv_bn_add_relu_right)
                    ._set_root_node_getter(_conv_bn_add_relu_root_right)
                    ._set_extra_inputs_getter(_conv_bn_add_relu_extra_right)
                    .set_fused_module(nni.ConvAddReLU2d),

                    # relu(add(conv(y), shortcut))
                    BackendPatternConfig()
                    ._set_pattern_complex_format((relu_op, (add_op, nn.Conv2d, MatchAllNode)))
                    .set_observation_type(observation)
                    .set_dtype_configs([dtype_config])
                    .set_fuser_method(_fuse_conv_add_relu_left)
                    ._set_root_node_getter(_conv_add_relu_root_left)
                    ._set_extra_inputs_getter(_conv_add_relu_extra_left)
                    .set_fused_module(nni.ConvAddReLU2d),

                    # relu(add(shortcut, conv(y)))
                    BackendPatternConfig()
                    ._set_pattern_complex_format((relu_op, (add_op, MatchAllNode, nn.Conv2d)))
                    .set_observation_type(observation)
                    .set_dtype_configs([dtype_config])
                    .set_fuser_method(_fuse_conv_add_relu_right)
                    ._set_root_node_getter(_conv_add_relu_root_right)
                    ._set_extra_inputs_getter(_conv_add_relu_extra_right)
                    .set_fused_module(nni.ConvAddReLU2d),
                ]
            )
    configs.extend(
        [
            *[
                # relu(bn(conv(x)))
                BackendPatternConfig((nn.Conv2d, nn.BatchNorm2d, relu_op))
                .set_observation_type(observation)
                .set_dtype_configs([dtype_config])
                .set_fuser_method(
                    fuse_conv_bn_relu
                    if relu_op is nn.ReLU
                    else lambda is_qat, conv, bn, relu: fuse_conv_bn_relu(
                        is_qat, conv, bn, nn.ReLU()
                    )
                )
                .set_fused_module(nni.ConvReLU2d)
                for relu_op in (nn.ReLU, F.relu, torch.relu)
            ],
            # bn(conv(x))
            BackendPatternConfig((nn.Conv2d, nn.BatchNorm2d))
            .set_observation_type(observation)
            .set_dtype_configs([dtype_config])
            .set_fuser_method(fuse_conv_bn),

            # relu(conv(x))
            *[
                BackendPatternConfig((nn.Conv2d, relu_op))
                .set_observation_type(observation)
                .set_dtype_configs([dtype_config])
                .set_fuser_method(_sequential_conv_relu)
                .set_fused_module(nni.ConvReLU2d)
                for relu_op in (nn.ReLU, F.relu, torch.relu)
            ],


            # relu(bn(Linear(x)))
            *[
                BackendPatternConfig((nn.Linear, nn.BatchNorm1d, relu_op))
                .set_observation_type(observation)
                .set_dtype_configs([dtype_config])
                .set_fuser_method(_fuse_linear_bn_relu)
                .set_fused_module(nni.LinearReLU)
                for relu_op in (nn.ReLU, F.relu, torch.relu)
            ],

            # bn(Linear(x))
            BackendPatternConfig((nn.Linear, nn.BatchNorm1d))
            .set_observation_type(observation)
            .set_dtype_configs([dtype_config])
            .set_fuser_method(fuse_linear_bn),

            # relu(Linear(x))
            *[
                BackendPatternConfig((nn.Linear, relu_op))
                .set_observation_type(observation)
                .set_dtype_configs([dtype_config])
                .set_fuser_method(_sequential_linear_relu)
                .set_fused_module(nni.LinearReLU)
                for relu_op in (nn.ReLU, F.relu, torch.relu)
            ],


            BackendPatternConfig(nn.Conv2d)
            .set_observation_type(observation)
            .set_dtype_configs([dtype_config]),
            BackendPatternConfig(nn.Linear)
            .set_observation_type(observation)
            .set_dtype_configs([dtype_config]),
            BackendPatternConfig(nni.ConvReLU2d)
            .set_observation_type(observation)
            .set_dtype_configs([dtype_config]),
            BackendPatternConfig(nni.ConvAddReLU2d)
            .set_observation_type(observation)
            .set_dtype_configs([dtype_config]),
            BackendPatternConfig(nni.LinearReLU)
            .set_observation_type(observation)
            .set_dtype_configs([dtype_config]),

            BackendPatternConfig(nn.MaxPool2d)
            .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
            .set_dtype_configs([dtype_config]),

            BackendPatternConfig(torch.flatten)
            .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
            .set_dtype_configs([dtype_config]),

            BackendPatternConfig(F.relu)
            .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
            .set_dtype_configs([dtype_config]),

            BackendPatternConfig(nn.ReLU)
            .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
            .set_dtype_configs([dtype_config]),
        ]
    )

    return BackendConfig("bked_test_backend").set_backend_pattern_configs(configs)
