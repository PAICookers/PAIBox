from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from paibox.paiir.ir.lut_activation import LutLinear, LutReLU, LutReLUSymmetric

from .utils import get_obs_params, quantize_to_int

HAS_LUT = True


@dataclass(frozen=True)
class QParams:
    scale: float
    zero_point: int

    @classmethod
    def from_observer(cls, observer) -> "QParams":
        scale, zero_point = get_obs_params(observer)
        return cls(float(scale), int(zero_point))


def _activation_is_symmetric(observer) -> bool:
    return int(QParams.from_observer(observer).zero_point) == 0


def _weight_qparams(module: nn.Conv2d | nn.Linear) -> QParams:
    if not hasattr(module, "qconfig") or module.qconfig is None:
        return QParams(1.0, 0)

    weight_obs = module.qconfig.weight()
    weight_obs(module.weight)
    return QParams.from_observer(weight_obs)


def _as_conv(module: nn.Module) -> nn.Conv2d:
    conv = getattr(module, "0", module)
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f"expected Conv2d, got {type(conv).__name__}")
    return conv


def _as_linear(module: nn.Module) -> nn.Linear:
    linear = getattr(module, "0", module)
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"expected Linear, got {type(linear).__name__}")
    return linear


def quantized_conv2d_asymmetric(
    self,
    x_q,
    weight_q,
    x_scale,
    w_scale,
    x_zero_point,
    w_zero_point,
    bias,
):
    x_int = x_q.float()
    weight_int = weight_q.float()

    term1 = F.conv2d(
        x_int,
        weight_int,
        bias=None,
        stride=self.stride,
        padding=self.padding,
        dilation=getattr(self, "dilation", 1),
        groups=self.groups,
    )

    weight_zero_t = torch.ones_like(weight_int) * w_zero_point
    term2 = F.conv2d(
        x_int,
        weight_zero_t,
        stride=self.stride,
        padding=self.padding,
        dilation=getattr(self, "dilation", 1),
        groups=self.groups,
    )

    weight_sum = weight_int.sum(dim=(1, 2, 3)).view(1, self.out_channels, 1, 1)
    term3 = x_zero_point * weight_sum

    kh, kw = (
        self.kernel_size
        if isinstance(self.kernel_size, tuple)
        else (self.kernel_size, self.kernel_size)
    )
    ic_per_group = self.in_channels // self.groups
    term4 = x_zero_point * w_zero_point * (ic_per_group * kh * kw)

    output_acc = term1 - term2 - term3 + term4
    output_scale = x_scale * w_scale

    if bias is not None:
        bias_q = torch.round(bias / output_scale).view(1, -1, 1, 1)
        output_acc = output_acc + bias_q

    return output_acc, output_scale


def quantized_linear_asymmetric(
    x_q,
    weight_q,
    x_scale,
    w_scale,
    x_zero_point,
    w_zero_point,
    bias,
):
    x_int = x_q.float()
    weight_int = weight_q.float()

    term1 = F.linear(x_int, weight_int)
    weight_zero_t = torch.ones_like(weight_int) * w_zero_point
    term2 = F.linear(x_int, weight_zero_t)

    weight_sum = weight_int.sum(dim=1).view(1, -1)
    term3 = x_zero_point * weight_sum
    term4 = x_zero_point * w_zero_point * weight_int.shape[1]

    output_acc = term1 - term2 - term3 + term4
    output_scale = x_scale * w_scale

    if bias is not None:
        bias_q = torch.round(bias / output_scale).view(1, -1)
        output_acc = output_acc + bias_q

    return output_acc, output_scale


def _input_to_int(x: torch.Tensor, qparams: QParams, symmetric: bool = True) -> torch.Tensor:
    if isinstance(x, torch.Tensor) and getattr(x, "is_quantized", False):
        return x.int_repr().to(torch.float32)
    if x.dtype == torch.float32:
        if symmetric:
            x = quantize_to_int(
                x, qparams.scale, qparams.zero_point, -128, 127, torch.int8
            )
        else:
            x = quantize_to_int(
                x, qparams.scale, qparams.zero_point, 0, 255, torch.uint8)
    return x


class _ManualQuantParameters(nn.Module):
    activation_symmetric: bool

    def _set_qparams(
        self,
        input_qparams: QParams,
        weight_qparams: QParams,
        output_qparams: QParams,
    ) -> None:
        self.s_in, self.z_in = input_qparams.scale, input_qparams.zero_point
        self.s_w, self.z_w = weight_qparams.scale, weight_qparams.zero_point
        self.s_out, self.z_out = output_qparams.scale, output_qparams.zero_point

    def input_to_int(self, x: torch.Tensor) -> torch.Tensor:
        return _input_to_int(x, QParams(self.s_in, self.z_in), self.activation_symmetric)


class _ManualConvParameters:
    def _copy_conv_attrs(self, conv: nn.Conv2d) -> None:
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.padding_mode = conv.padding_mode
        self.bias_val = conv.bias


class ManualConvReLU2d(_ManualConvParameters, _ManualQuantParameters):
    def __init__(
        self,
        original_module: nn.Conv2d,
        input_qparams: QParams | tuple[float, int],
        weight_qparams: QParams | tuple[float, int],
        output_qparams: QParams | tuple[float, int],
        activation_symmetric: bool = False,
    ):
        super().__init__()
        input_qparams = _normalize_qparams(input_qparams)
        weight_qparams = _normalize_qparams(weight_qparams)
        output_qparams = _normalize_qparams(output_qparams)
        self._copy_conv_attrs(original_module)
        self.activation_symmetric = activation_symmetric
        self._set_qparams(input_qparams, weight_qparams, output_qparams)
        self.weight_q = quantize_to_int(
            original_module.weight, self.s_w, self.z_w, -128, 127, torch.int8
        )

        s_accum = self.s_in * self.s_w
        lut_scale = self.s_out / s_accum if s_accum != 0 else 0.0
        if self.activation_symmetric:
            self.lut = LutReLUSymmetric(
                min_val=-lut_scale * 128.0,
                max_val=lut_scale * 127.0,
                output_sign=1,
            )
        else:
            self.lut = LutReLU(
                min_val=-5.0, max_val=lut_scale * 255.0, output_sign=0)

    @classmethod
    def from_observed(cls, observed):
        conv = _as_conv(observed.float_module)
        return cls(
            conv,
            QParams.from_observer(observed.input_activation_post_process),
            _weight_qparams(conv),
            QParams.from_observer(observed.activation_post_process),
            activation_symmetric=_activation_is_symmetric(
                observed.input_activation_post_process
            ),
        )

    def forward(self, x):
        x = self.input_to_int(x)
        out_acc, _ = quantized_conv2d_asymmetric(
            self,
            x,
            self.weight_q,
            self.s_in,
            self.s_w,
            self.z_in,
            self.z_w,
            self.bias_val,
        )
        out_q = self.lut(out_acc)
        target_dtype = torch.int8 if self.activation_symmetric else torch.uint8
        return out_q if out_q.dtype == target_dtype else out_q.to(target_dtype)


class ManualLinear(_ManualQuantParameters):
    def __init__(
        self,
        original_module: nn.Linear,
        input_qparams: QParams | tuple[float, int],
        weight_qparams: QParams | tuple[float, int],
        output_qparams: QParams | tuple[float, int],
        activation_symmetric: bool = False,
    ):
        super().__init__()
        input_qparams = _normalize_qparams(input_qparams)
        weight_qparams = _normalize_qparams(weight_qparams)
        output_qparams = _normalize_qparams(output_qparams)
        self.bias_val = original_module.bias
        self.activation_symmetric = activation_symmetric
        self._set_qparams(input_qparams, weight_qparams, output_qparams)
        self.weight_q = quantize_to_int(
            original_module.weight, self.s_w, self.z_w, -128, 127, torch.int8
        )

        s_accum = self.s_in * self.s_w
        lut_scale = self.s_out / s_accum if s_accum != 0 else 0.0
        self.lut = LutLinear(
            min_val=-lut_scale * 128.0,
            max_val=lut_scale * 127.0,
            output_sign=1,
        )

    @classmethod
    def from_observed(cls, observed):
        linear = _as_linear(observed.float_module)
        return cls(
            linear,
            QParams.from_observer(observed.input_activation_post_process),
            _weight_qparams(linear),
            QParams.from_observer(observed.activation_post_process),
            activation_symmetric=_activation_is_symmetric(
                observed.input_activation_post_process
            ),
        )

    def forward(self, x):
        x = self.input_to_int(x)
        out_acc, out_scale = quantized_linear_asymmetric(
            x,
            self.weight_q,
            self.s_in,
            self.s_w,
            self.z_in,
            self.z_w,
            self.bias_val,
        )
        return out_acc * out_scale


class ManualLinearReLU(ManualLinear):
    @classmethod
    def from_observed(cls, observed):
        linear = _as_linear(observed.float_module)
        return cls(
            linear,
            QParams.from_observer(observed.input_activation_post_process),
            _weight_qparams(linear),
            QParams.from_observer(observed.activation_post_process),
            activation_symmetric=_activation_is_symmetric(
                observed.input_activation_post_process
            ),
        )

    def __init__(
        self,
        original_module: nn.Linear,
        input_qparams: QParams | tuple[float, int],
        weight_qparams: QParams | tuple[float, int],
        output_qparams: QParams | tuple[float, int],
        activation_symmetric: bool = False,
    ):
        nn.Module.__init__(self)
        input_qparams = _normalize_qparams(input_qparams)
        weight_qparams = _normalize_qparams(weight_qparams)
        output_qparams = _normalize_qparams(output_qparams)
        self.bias_val = original_module.bias
        self.activation_symmetric = activation_symmetric
        self._set_qparams(input_qparams, weight_qparams, output_qparams)
        self.weight_q = quantize_to_int(
            original_module.weight, self.s_w, self.z_w, -128, 127, torch.int8
        )

        s_accum = self.s_in * self.s_w
        lut_scale = self.s_out / s_accum if s_accum != 0 else 0.0
        if self.activation_symmetric:
            self.lut = LutReLUSymmetric(
                min_val=-lut_scale * 128.0,
                max_val=lut_scale * 127.0,
                output_sign=1,
            )
        else:
            self.lut = LutReLU(
                min_val=-5.0, max_val=lut_scale * 255.0, output_sign=0)

    def forward(self, x):
        x = self.input_to_int(x)
        out_acc, _ = quantized_linear_asymmetric(
            x,
            self.weight_q,
            self.s_in,
            self.s_w,
            self.z_in,
            self.z_w,
            self.bias_val,
        )
        out_q = self.lut(out_acc)
        target_dtype = torch.int8 if self.activation_symmetric else torch.uint8
        return out_q if out_q.dtype == target_dtype else out_q.to(target_dtype)


class ManualConv2d(_ManualConvParameters, _ManualQuantParameters):
    def __init__(
        self,
        original_module: nn.Conv2d,
        input_qparams: QParams | tuple[float, int],
        weight_qparams: QParams | tuple[float, int],
        output_qparams: QParams | tuple[float, int],
        activation_symmetric: bool = False,
    ):
        super().__init__()
        input_qparams = _normalize_qparams(input_qparams)
        weight_qparams = _normalize_qparams(weight_qparams)
        output_qparams = _normalize_qparams(output_qparams)
        self._copy_conv_attrs(original_module)
        self.activation_symmetric = activation_symmetric
        self._set_qparams(input_qparams, weight_qparams, output_qparams)
        self.weight_q = quantize_to_int(
            original_module.weight, self.s_w, self.z_w, -128, 127, torch.int8
        )

        s_accum = self.s_in * self.s_w
        lut_scale = self.s_out / s_accum if s_accum != 0 else 0.0
        self.lut = LutLinear(
            min_val=-lut_scale * 128.0,
            max_val=lut_scale * 127.0,
            output_sign=1,
        )

    @classmethod
    def from_observed(cls, observed):
        conv = _as_conv(observed.float_module)
        return cls(
            conv,
            QParams.from_observer(observed.input_activation_post_process),
            _weight_qparams(conv),
            QParams.from_observer(observed.activation_post_process),
            activation_symmetric=_activation_is_symmetric(
                observed.input_activation_post_process
            ),
        )

    def forward(self, x):
        x = self.input_to_int(x)
        out_acc, _ = quantized_conv2d_asymmetric(
            self,
            x,
            self.weight_q,
            self.s_in,
            self.s_w,
            self.z_in,
            self.z_w,
            self.bias_val,
        )
        out_q = self.lut(out_acc)
        target_dtype = torch.int8
        return out_q if out_q.dtype == target_dtype else out_q.to(target_dtype)


class ManualConvAddReLU2d(_ManualConvParameters, _ManualQuantParameters):
    def __init__(
        self,
        conv: nn.Conv2d,
        y_qparams: QParams | tuple[float, int],
        x_qparams: QParams | tuple[float, int],
        weight_qparams: QParams | tuple[float, int],
        out_qparams: QParams | tuple[float, int],
        activation_symmetric: bool = False,
    ):
        super().__init__()
        y_qparams = _normalize_qparams(y_qparams)
        x_qparams = _normalize_qparams(x_qparams)
        weight_qparams = _normalize_qparams(weight_qparams)
        out_qparams = _normalize_qparams(out_qparams)

        self._copy_conv_attrs(conv)
        self.activation_symmetric = activation_symmetric
        self.s_in, self.z_in = y_qparams.scale, y_qparams.zero_point
        self.s_w, self.z_w = weight_qparams.scale, weight_qparams.zero_point
        self.s_out, self.z_out = out_qparams.scale, out_qparams.zero_point
        self.x_scale, self.x_zp = x_qparams.scale, x_qparams.zero_point
        self.out_scale, self.out_zp = self.s_out, self.z_out
        self.weight_q = quantize_to_int(
            conv.weight, self.s_w, self.z_w, -128, 127, torch.int8)

        s_accum = self.s_in * self.s_w
        lut_scale = self.out_scale / s_accum if s_accum != 0 else 0.0
        if activation_symmetric:
            self.lut = LutReLUSymmetric(
                min_val=-lut_scale * 128.0,
                max_val=lut_scale * 127.0,
                output_sign=1,
            )
        else:
            self.lut = LutReLU(
                min_val=-5.0, max_val=lut_scale * 255.0, output_sign=0)

    @classmethod
    def from_observed(cls, observed):
        conv = _as_conv(observed.float_module)
        return cls(
            conv,
            QParams.from_observer(observed.y_activation_post_process),
            QParams.from_observer(observed.x_activation_post_process),
            _weight_qparams(conv),
            QParams.from_observer(observed.activation_post_process),
            activation_symmetric=_activation_is_symmetric(
                observed.y_activation_post_process
            ),
        )

    def forward(self, y, x):
        from .utils import approximate_scale_ratio

        x_int = self.input_to_int(x).to(torch.int32)

        target_scale = self.s_in * self.s_w
        exact_ratio = self.x_scale / target_scale if target_scale != 0 else 0.0
        M, n = approximate_scale_ratio(exact_ratio)

        q2_int = torch.round((x_int - self.x_zp) * M *
                             (2.0**n)).to(torch.int32)

        y_int = self.input_to_int(y)
        out_conv_acc, _ = quantized_conv2d_asymmetric(
            self,
            y_int,
            self.weight_q,
            self.s_in,
            self.s_w,
            self.z_in,
            self.z_w,
            self.bias_val,
        )
        out_conv_acc = torch.round(out_conv_acc).to(torch.int32)

        q_relu_int = self.lut(out_conv_acc + q2_int)
        target_dtype = torch.int8 if self.activation_symmetric else torch.uint8
        return q_relu_int if q_relu_int.dtype == target_dtype else q_relu_int.to(target_dtype)


def _normalize_qparams(qparams: QParams | tuple[float, int]) -> QParams:
    if isinstance(qparams, QParams):
        return qparams
    scale, zero_point = qparams
    return QParams(float(scale), int(zero_point))
