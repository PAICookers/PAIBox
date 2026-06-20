from typing import TypeVar

import pytest
import torch
from spikingjelly.activation_based import neuron as sj
from torch import Tensor, nn

from paibox.paiir.ir.graph import PAIIRGraph
from paibox.paiir.ir.ir_base import PAIIRNode
from paibox.paiir.ir.lut_activation import LutCustom
from paibox.paiir.ir.op_node import LayoutStage, OfflineCoreOp, ShapeStage, TransformOp
from paibox.paiir.lowering.converter import (
    _DEFAULT_MODULE_MAP,
    _USER_MODULE_MAP,
    torch_to_paiir,
)
from paibox.paiir.pipeline.data_format import DataFormat
from paibox.paiir.pipeline.layout_chain_canonicalization import (
    canonicalize_layout_chains,
)
from paibox.paiir.pipeline.layout_cross_node_elision import (
    commute_pre_activation_transforms,
)
from paibox.paiir.pipeline.passes import (
    flatten_general_add_chains,
    fuse_to_offline_cores,
    propagate_data_format,
    propagate_signal_semantics,
    specialize_general_adds,
)

_T = TypeVar("_T", bound=PAIIRNode)


@pytest.fixture(autouse=True)
def restore_default_module_map():
    """Restore the global neuron/module registry after each test.

    Built-in lowering rules live in ``_DEFAULT_MODULE_MAP`` while
    ``register_neuron(...)`` and ``register_module(...)`` write
    test/user overrides into ``_USER_MODULE_MAP``. Tests that register custom
    lowering hooks should not leak those overrides into later tests.
    """
    original_default = dict(_DEFAULT_MODULE_MAP)
    original_user = dict(_USER_MODULE_MAP)
    try:
        yield
    finally:
        _DEFAULT_MODULE_MAP.clear()
        _DEFAULT_MODULE_MAP.update(original_default)
        _USER_MODULE_MAP.clear()
        _USER_MODULE_MAP.update(original_user)


class SNNTwoLayer(nn.Module):
    """Conv-LIF -> Conv-IF: basic sequential SNN."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.lif1 = sj.LIFNode(tau=2.0, v_threshold=1.0)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.if1 = sj.IFNode(v_threshold=1.0)

    def forward(self, x):
        x = self.lif1(self.conv1(x))
        return self.if1(self.conv2(x))


class SNNWithMaxPool(nn.Module):
    """Conv-LIF -> MaxPool-LIF: pooling in SNN."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.lif1 = sj.LIFNode(tau=2.0)
        self.pool = nn.MaxPool2d(2)
        self.lif2 = sj.IFNode(v_threshold=1.0)

    def forward(self, x):
        x = self.lif1(self.conv(x))
        return self.lif2(self.pool(x))


class SNNDepthwiseSeparable(nn.Module):
    """DWConv-LIF -> PWConv-LIF: mobile-style SNN block."""

    def __init__(self):
        super().__init__()
        self.dwconv = nn.Conv2d(16, 16, 3, padding=1, groups=16)
        self.lif1 = sj.LIFNode(tau=2.0)
        self.pwconv = nn.Conv2d(16, 32, 1)
        self.lif2 = sj.LIFNode(tau=2.0)

    def forward(self, x):
        x = self.lif1(self.dwconv(x))
        return self.lif2(self.pwconv(x))


class SNNFlattenTransition(nn.Module):
    """Conv-IF -> flatten -> Linear-IF: spatial to dense transition."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.if1 = sj.IFNode(v_threshold=1.0)
        self.linear = nn.Linear(8 * 4 * 4, 10)
        self.if2 = sj.IFNode(v_threshold=1.0)

    def forward(self, x):
        x = self.if1(self.conv(x))
        x = x.flatten(1)
        return self.if2(self.linear(x))


class SNNResidualAdd(nn.Module):
    """Two conv branches + add + LIF: residual connection."""

    def __init__(self):
        super().__init__()
        self.conv_a = nn.Conv2d(3, 16, 3, padding=1)
        self.conv_b = nn.Conv2d(3, 16, 3, padding=1)
        self.lif = sj.LIFNode(tau=2.0)

    def forward(self, x):
        return self.lif(self.conv_a(x) + self.conv_b(x))


class MultiInputMerge(nn.Module):
    """Two separate inputs, separate conv branches, add -> LIF."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 16, 3, padding=1)
        self.lif = sj.LIFNode(tau=2.0)

    def forward(self, x, y):
        return self.lif(self.conv1(x) + self.conv2(y))


class SNNWithAvgPoolIF(nn.Module):
    """Conv-IF -> AvgPool-IF: split-core deployment pattern."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.if1 = sj.IFNode(v_threshold=1.0)
        self.pool = nn.AvgPool2d(kernel_size)
        self.if2 = sj.IFNode(v_threshold=1.0)

    def forward(self, x):
        x = self.if1(self.conv(x))
        return self.if2(self.pool(x))


class SNNWithAvgPool1dIF(nn.Module):
    """Conv1d-IF -> AvgPool1d-IF: 1D pooling for parametric window size tests.

    AvgPool1d(kernel_size) has window_size = kernel_size, making it easy to
    test various window sizes with a single dimension. Uses Conv1d to maintain
    3D tensor shape (batch, channels, length) expected by AvgPool1d.
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 3, padding=1)
        self.if1 = sj.IFNode(v_threshold=1.0)
        self.pool = nn.AvgPool1d(kernel_size)
        self.if2 = sj.IFNode(v_threshold=1.0)

    def forward(self, x):
        x = self.if1(self.conv(x))
        return self.if2(self.pool(x))


class SNNWithAvgPoolLIF(nn.Module):
    """Conv-IF -> AvgPool-LIF: shared-core by default, split-core candidate."""

    def __init__(self, kernel_size: int, tau: float = 4.0):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.if1 = sj.IFNode(v_threshold=1.0)
        self.pool = nn.AvgPool2d(kernel_size)
        self.lif2 = sj.LIFNode(tau=tau, v_threshold=1.0)

    def forward(self, x):
        x = self.if1(self.conv(x))
        return self.lif2(self.pool(x))


class SNNWithAvgPool1dLIF(nn.Module):
    """Conv1d-IF -> AvgPool1d-LIF: 1D fixture for split-core LIF tests."""

    def __init__(self, kernel_size: int, tau: float = 4.0):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 3, padding=1)
        self.if1 = sj.IFNode(v_threshold=1.0)
        self.pool = nn.AvgPool1d(kernel_size)
        self.lif2 = sj.LIFNode(tau=tau, v_threshold=1.0)

    def forward(self, x):
        x = self.if1(self.conv(x))
        return self.lif2(self.pool(x))


class SimpleCNN(nn.Module):
    """Conv-ReLU-Conv: minimal ANN without classifier head."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.conv2(x)


class ANNClassifier(nn.Module):
    """Conv-ReLU -> AvgPool -> flatten -> Linear-Sigmoid."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(8)
        self.linear = nn.Linear(16, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.sigmoid(self.linear(x))


class ANNConvBNReLU(nn.Module):
    """Conv-BN-ReLU -> Conv-BN-ReLU: typical ANN backbone with BN bypass."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        return self.relu2(self.bn2(self.conv2(x)))


class ANNResidualSubtract(nn.Module):
    """Two linear branches, subtraction merge, tanh: difference network."""

    def __init__(self):
        super().__init__()
        self.linear_a = nn.Linear(8, 16)
        self.linear_b = nn.Linear(8, 16)
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(self.linear_a(x) - self.linear_b(x))


class SJMNISTValidationNet(nn.Module):
    """MNIST-sized SpikingJelly reference network for future online-core validation.

    The topology stays intentionally small:

    - grayscale `1x28x28` input
    - flatten
    - dense hidden layer
    - SpikingJelly IF neuron
    - dense classifier head

    This keeps the model easy to reuse across compile, export, and later
    online-training validation, while still being a real SpikingJelly network.
    """

    def __init__(self, hidden_features: int = 128):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(28 * 28, hidden_features, bias=False)
        self.if1 = sj.IFNode(v_threshold=1.0)
        self.fc2 = nn.Linear(hidden_features, 10, bias=False)

    def forward(self, x):
        x = self.flatten(x)
        x = self.if1(self.fc1(x))
        return self.fc2(x)


class SPPFBlock(nn.Module):
    """Conv-LIF -> cascaded MaxPool x3 -> cat -> Conv-LIF: SPPF pattern."""

    def __init__(self):
        super().__init__()
        self.cv1 = nn.Sequential(nn.Conv2d(16, 8, 1), sj.IFNode(v_threshold=1.0))
        self.m = nn.MaxPool2d(5, stride=1, padding=2)
        self.cv2 = nn.Sequential(nn.Conv2d(32, 16, 1), sj.IFNode(v_threshold=1.0))

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class MultiSpike4(nn.Module):
    """Custom neuron: quantized step activation with 5 output levels."""

    class quant4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=4))

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)


def make_multispike4_lut() -> LutCustom:
    """Create LutCustom for MultiSpike4 neuron.

    round(clamp(x, 0, 4)): only 5 effective levels
    thresholds [0,1,2,3,4,4,...,4], values = thresholds clamped to [0,4]
    """
    thresholds = torch.cat([torch.arange(5), torch.full((251,), 4)])
    values = thresholds.clone()
    return LutCustom(thresholds, values)


class UnsupportedSoftmax(nn.Module):
    """Model using an op unsupported by PAIIR (torch.softmax)."""

    def forward(self, x):
        return torch.softmax(x, dim=1)


class UnsupportedSinModel(nn.Module):
    """Model using torch.sin (unsupported by PAIIR)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)

    def forward(self, x):
        return torch.sin(self.conv(x))


def make_img_1ch_4x4() -> Tensor:
    """Single-channel 4x4 image, for flatten/linear tests."""
    return torch.randn(1, 1, 4, 4)


def make_img_1ch_28x28() -> Tensor:
    """Single-channel 28x28 image, matching the default MNIST validation net."""
    return torch.randn(1, 1, 28, 28)


def make_img_3ch_8x8() -> Tensor:
    """3-channel 8x8 image, standard small input."""
    return torch.randn(1, 3, 8, 8)


def make_img_3ch_9x9() -> Tensor:
    """3-channel 9x9 image, for odd-kernel pool tests."""
    return torch.randn(1, 3, 9, 9)


def make_img_3ch_32x32() -> Tensor:
    """3-channel 32x32 image, standard larger input."""
    return torch.randn(1, 3, 32, 32)


def make_vec_8d() -> Tensor:
    """8-dimensional vector, for linear layer tests."""
    return torch.randn(1, 8)


def make_vec_64d() -> Tensor:
    """64-dimensional vector, for AvgPool1d tests with various kernel sizes."""
    return torch.randn(1, 1, 64)


def make_img_16ch_8x8() -> Tensor:
    """16-channel 8x8 feature map, for depthwise/separable tests."""
    return torch.randn(1, 16, 8, 8)


def find_nodes(graph: PAIIRGraph, node_type: type[_T]) -> list[_T]:
    """Return all nodes of a given type."""
    return [n for n in graph.nodes.values() if isinstance(n, node_type)]


def find_node_names(graph: PAIIRGraph, node_type: type[PAIIRNode]) -> list[str]:
    """Return names of all nodes of a given type."""
    return [name for name, n in graph.nodes.items() if isinstance(n, node_type)]


def find_first(graph: PAIIRGraph, node_type: type[_T]) -> _T:
    """Return the first node of a given type."""
    return next(n for n in graph.nodes.values() if isinstance(n, node_type))


def make_transform(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    input_dims: tuple[int, ...],
) -> TransformOp:
    """Build one simple transform-compatible test node."""
    return TransformOp(
        (
            LayoutStage(input_dims),
            ShapeStage(lambda _input_shape, bound=torch.Size(output_shape): bound),
        )
    )


def find_transform_nodes(graph: PAIIRGraph) -> list[TransformOp]:
    """Return all transform-like routing nodes in the graph."""
    return [n for n in graph.nodes.values() if isinstance(n, TransformOp)]


def offline_nodes(graph: PAIIRGraph) -> list[OfflineCoreOp]:
    """Return all OfflineCoreOp nodes."""
    return find_nodes(graph, OfflineCoreOp)


def convert_and_fuse(model: nn.Module, *sample_inputs: Tensor) -> PAIIRGraph:
    """Trace a PyTorch model to PAIIR and fuse."""
    unfused = torch_to_paiir(model, *sample_inputs)
    unfused = canonicalize_layout_chains(unfused)
    unfused = commute_pre_activation_transforms(unfused)
    unfused = flatten_general_add_chains(unfused)
    unfused = specialize_general_adds(unfused)
    return fuse_to_offline_cores(unfused)


def convert_fuse_propagate(
    model: nn.Module,
    *sample_inputs: Tensor,
    input_formats: dict[str, DataFormat] | None = None,
) -> PAIIRGraph:
    """Full pipeline: trace -> fuse -> propagate semantics and data format."""
    fused = convert_and_fuse(model, *sample_inputs)
    propagate_signal_semantics(fused, input_formats=input_formats)
    propagate_data_format(fused, input_formats=input_formats)
    return fused
