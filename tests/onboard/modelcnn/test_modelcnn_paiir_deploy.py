"""Deploy smokes for the quantized MNIST FC and LeNet5 exports."""

import os
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

import torch
import torch.nn as nn

from paibox.backendv2.mapper import Mapper
from paibox.paiir import ANNNodeV25, compile_to_paiir, register_neuron
from paibox.paiir.ir.lut_activation import LutReLUSymmetric
from tests.onboard.modelcnn.minist.lenet5_minist.model import LeNet5

MODEL_DIR = Path(__file__).resolve().parent
DEBUG_DIR = MODEL_DIR / "debug" / "paiir_modelcnn_deploy"
FC_EXPORT_DIR = MODEL_DIR / "minist" / "fc_minist" / "exported_params_symmetric"
LENET5_EXPORT_DIR = (
    MODEL_DIR / "minist" / "lenet5_minist" / "exported_params_symmetric"
)
SAMPLE_INPUT = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
CUSTOMER_LUT_MIN = -5.0


def _lut_relu_max(
    input_scale: float, weight_scale: float, output_scale: float
) -> float:
    return output_scale / (weight_scale * input_scale) * 127.0

def _lut_relu_min(
    input_scale: float, weight_scale: float, output_scale: float
) -> float:
    return -output_scale / (weight_scale * input_scale) * 128.0


FC_LUT_SPECS = {
    "layer1_0": (0.022238, 0.000945, 0.034190),
    "layer2_0": (0.034190, 0.001758, 0.029330),
}

LENET5_LUT_SPECS = {
    "features_0": (0.022304, 0.002591, 0.031766),
    "features_3": (0.031766, 0.002261, 0.037156),
    "classifier_0": (0.037156, 0.004294, 0.068156),
    "classifier_2": (0.068156, 0.005105, 0.129020),
}


class CustomerLutReLU(nn.Module):
    """Float-forward shim that lowers to the exact customer LutReLU."""

    _is_leaf_module = True

    def __init__(self, min_val: float, max_val: float, output_sign: int) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.output_sign = output_sign

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


def _make_customer_lut_relu(
    input_scale: float, weight_scale: float, output_scale: float
) -> CustomerLutReLU:
    # The export keeps customer LUT calibration scalars, so we rebuild the
    # deploy activation from those scalars before lowering.
    return CustomerLutReLU(
        min_val=_lut_relu_min(input_scale, weight_scale, output_scale),
        max_val=_lut_relu_max(input_scale, weight_scale, output_scale),
        output_sign=1,
    )


try:
    register_neuron(
        CustomerLutReLU,
        lambda mod: ANNNodeV25(
            LutReLUSymmetric(
                min_val=mod.min_val, max_val=mod.max_val, output_sign=mod.output_sign
            )
        ),
    )
except ValueError:
    pass


class QuantizedMnistFCDeployModel(nn.Module):
    """Deploy-side FC topology after the export has folded BatchNorm."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(28 * 28, 512, bias=True)
        self.relu1 = _make_customer_lut_relu(*FC_LUT_SPECS["layer1_0"])
        self.layer2 = nn.Linear(512, 256, bias=True)
        self.relu2 = _make_customer_lut_relu(*FC_LUT_SPECS["layer2_0"])
        self.classifier = nn.Linear(256, 10, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        return self.classifier(x)


class _TeeStream:
    def __init__(self, *streams) -> None:
        self._streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

    def fileno(self) -> int:
        for stream in self._streams:
            fileno = getattr(stream, "fileno", None)
            if fileno is not None:
                return fileno()
        raise OSError("tee stream has no file descriptor")


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _stream_log(path: Path, header_lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    terminal = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
    with path.open("w", encoding="utf-8") as log_file:
        tee = _TeeStream(terminal, log_file)
        with redirect_stdout(tee):
            for line in header_lines:
                print(line)
            yield


def _load_tensor(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    assert isinstance(tensor, torch.Tensor), f"{path} must contain a tensor"
    return tensor


def _load_export_pair(
    export_dir: Path, prefix: str
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = _load_tensor(export_dir / f"{prefix}_weight_int8.pth")
    bias = _load_tensor(export_dir / f"{prefix}_bias_int32.pth")
    assert weight.dtype == torch.int8
    assert bias.dtype == torch.int32
    return weight, bias


def _assert_eval_inference_model(model: nn.Module) -> None:
    assert model.training is False
    assert all(module.training is False for module in model.modules())
    assert not any(isinstance(module, nn.Dropout) for module in model.modules())
    assert not any(
        isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
        for module in model.modules()
    )


def _bind_exported_params(
    module: nn.Linear | nn.Conv2d,
    weight_int8: torch.Tensor,
    bias_int32: torch.Tensor,
) -> None:
    assert tuple(module.weight.shape) == tuple(weight_int8.shape)
    assert module.bias is not None
    assert tuple(module.bias.shape) == tuple(bias_int32.shape)

    # The compile path still reads nn.Module weights directly, so we keep the
    # exported integer tensors both as explicit buffers and copied parameters.
    module.register_buffer("weight_int8", weight_int8.clone())
    module.register_buffer("bias_int32", bias_int32.clone())

    with torch.no_grad():
        module.weight.copy_(weight_int8.to(module.weight.dtype))
        module.bias.copy_(bias_int32.to(module.bias.dtype))

    assert torch.equal(module.weight_int8, weight_int8)
    assert torch.equal(module.bias_int32, bias_int32)


def _build_fc_model_from_exports() -> QuantizedMnistFCDeployModel:
    # FC export already fused BatchNorm away, so deploy uses the folded topology.
    model = QuantizedMnistFCDeployModel().eval()
    _bind_exported_params(
        model.layer1, *_load_export_pair(FC_EXPORT_DIR, "manual_quant_layer1_0")
    )
    _bind_exported_params(
        model.layer2, *_load_export_pair(FC_EXPORT_DIR, "manual_quant_layer2_0")
    )
    _bind_exported_params(
        model.classifier, *_load_export_pair(FC_EXPORT_DIR, "manual_quant_classifier")
    )
    _assert_eval_inference_model(model)
    return model


def _build_lenet5_model_from_exports() -> LeNet5:
    model = LeNet5().eval()
    # Swap training-time ReLU with the calibrated deploy LUT form.
    model.features[1] = _make_customer_lut_relu(*LENET5_LUT_SPECS["features_0"])
    model.features[4] = _make_customer_lut_relu(*LENET5_LUT_SPECS["features_3"])
    model.classifier[1] = _make_customer_lut_relu(*LENET5_LUT_SPECS["classifier_0"])
    model.classifier[3] = _make_customer_lut_relu(*LENET5_LUT_SPECS["classifier_2"])
    _bind_exported_params(
        model.features[0],
        *_load_export_pair(LENET5_EXPORT_DIR, "manual_quant_features_0"),
    )
    _bind_exported_params(
        model.features[3],
        *_load_export_pair(LENET5_EXPORT_DIR, "manual_quant_features_3"),
    )
    _bind_exported_params(
        model.classifier[0],
        *_load_export_pair(LENET5_EXPORT_DIR, "manual_quant_classifier_0"),
    )
    _bind_exported_params(
        model.classifier[2],
        *_load_export_pair(LENET5_EXPORT_DIR, "manual_quant_classifier_2"),
    )
    _bind_exported_params(
        model.classifier[4],
        *_load_export_pair(LENET5_EXPORT_DIR, "manual_quant_classifier_4"),
    )
    # Re-apply eval() after module replacement so every child stays in deploy mode.
    model.eval()
    _assert_eval_inference_model(model)
    return model


def _compile_and_backend_smoke(model: nn.Module, case_name: str):
    with torch.no_grad():
        forward_output = model(SAMPLE_INPUT)
    assert tuple(forward_output.shape) == (1, 10)

    graph = compile_to_paiir(model, SAMPLE_INPUT)

    summary_path = DEBUG_DIR / f"{case_name}_paiir_summary.log"
    with _stream_log(
        summary_path,
        [
            f"case: {case_name}",
            f"model: {type(model).__name__}",
            f"sample_shape: {tuple(SAMPLE_INPUT.shape)}",
            f"forward_output_shape: {tuple(forward_output.shape)}",
            "",
            "PAIIRGraph.summary()",
            "",
        ],
    ):
        graph.summary()

    backend_error: Exception | None = None
    backend_status = "succeeded"
    mapper = None
    backend_log_path = DEBUG_DIR / f"{case_name}_backendv2.log"
    with _pushd(DEBUG_DIR / case_name):
        with _stream_log(
            backend_log_path,
            [
                f"case: {case_name}",
                f"model: {type(model).__name__}",
                f"sample_shape: {tuple(SAMPLE_INPUT.shape)}",
                "",
                "backend_stdout",
                "",
            ],
        ):
            try:
                mapper = Mapper()
                mapper.compile(graph)
            except Exception as exc:  # pragma: no cover - exercised in real runs
                backend_error = exc
                backend_status = f"failed with {type(exc).__name__}: {exc}"
            finally:
                print("")
                print(
                    f"routing_groups: {len(mapper.routing_groups) if mapper is not None else 'n/a'}"
                )
                print(f"backend_compile: {backend_status}")

    if backend_error is not None:
        raise AssertionError(
            f"{case_name} backend compile failed; see {backend_log_path}"
        ) from backend_error

    assert mapper is not None
    return graph, len(mapper.routing_groups)


def test_mnist_fc_quantized_exports_compile_to_paiir_and_backend() -> None:
    graph, routing_group_count = _compile_and_backend_smoke(
        _build_fc_model_from_exports(), "mnist_fc"
    )
    assert len(graph.nodes) > 0
    assert routing_group_count > 0


def test_lenet5_quantized_exports_compile_to_paiir_and_backend() -> None:
    graph, routing_group_count = _compile_and_backend_smoke(
        _build_lenet5_model_from_exports(), "mnist_lenet5"
    )
    assert len(graph.nodes) > 0
    assert routing_group_count > 0

if __name__ == "__main__":
    test_mnist_fc_quantized_exports_compile_to_paiir_and_backend()
    test_lenet5_quantized_exports_compile_to_paiir_and_backend()