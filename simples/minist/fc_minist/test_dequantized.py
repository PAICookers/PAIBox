"""
测试已导出的量化参数在 FC MNIST 上的推理精度 (反量化回 FP32 测试)
"""

from torch.ao.quantization.quantize_fx import fuse_fx
from train import build_dataloaders, evaluate, DATA_DIR
from model import MNIST_FC
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# 确保能找到 paibox 和简单的上级目录
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..")))


EXPORT_DIR = BASE_DIR / "exported_params_symmetric"

# 从 test_modelcnn_paiir_deploy.py 获取的 FC Scales: (s_in, s_w, s_out)
# classifier 是输出层，并没有独立 LUT 激活，只有量化参数。它的 in_scale 等于上一层的 out_scale。
DEQUANT_SCALES = {
    "manual_quant_layer1_0": (0.022238, 0.000945),
    "manual_quant_layer2_0": (0.034190, 0.001758),
    # 0.029330 就是 layer2 的 s_out，w_scale 填一个近似值仅作演示反量化，不影响 argmax 或者按需读取
    "manual_quant_classifier": (0.029330, 0.002000),
}


def _lut_relu_max(input_scale: float, weight_scale: float, output_scale: float) -> float:
    return output_scale / (weight_scale * input_scale) * 255.0


class CustomerLutReLU(nn.Module):
    """Float-forward shim for quant activation scaling."""

    def __init__(self, min_val: float, max_val: float, output_sign: int = 0) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.output_sign = output_sign

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在浮点运算下模拟 ReLU
        return torch.relu(x)


# 之前部署时用的 FC LUT_SPECS (s_in, s_w, s_out)
FC_LUT_SPECS = {
    "layer1_0": (0.022238, 0.000945, 0.034190),
    "layer2_0": (0.034190, 0.001758, 0.029330),
}


def _make_customer_lut_relu(input_scale: float, weight_scale: float, output_scale: float) -> CustomerLutReLU:
    return CustomerLutReLU(
        min_val=-5.0,
        max_val=_lut_relu_max(input_scale, weight_scale, output_scale),
        output_sign=0,
    )


def _load_export_pair(export_dir: Path, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    w_path = export_dir / f"{prefix}_weight_int8.pth"
    b_path = export_dir / f"{prefix}_bias_int32.pth"
    if not w_path.exists() or not b_path.exists():
        raise FileNotFoundError(f"Cannot find {prefix} in {export_dir}")
    weight = torch.load(w_path, map_location="cpu")
    bias = torch.load(b_path, map_location="cpu")
    return weight, bias


def _bind_exported_params(module: nn.Module, weight_int8: torch.Tensor, bias_int32: torch.Tensor, s_in: float, s_w: float) -> None:
    # 第一步反量化：Weight = Int8 * s_w
    # Bias = Int32 * (s_in * s_w)
    # 原 module 此前可能未开启 bias=True，或者 bias 尚未初始化（FC 的原始定义是 bias=False，但 fuse_fx 后会产生 bias）
    with torch.no_grad():
        module.weight.copy_(weight_int8.to(module.weight.dtype) * s_w)
        if module.bias is not None:
            module.bias.copy_(bias_int32.to(module.bias.dtype) * (s_in * s_w))
        else:
            module.bias = nn.Parameter(bias_int32.to(
                module.weight.dtype) * (s_in * s_w))

# 简单部署一个展开的等效模型来测，这避免了原版 BatchNorm 造成的前向阻碍


class QuantizedMnistFCDeployModel(nn.Module):
    """Deployment-side FC topology after BN fusion in the quantized export."""

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


def test_quantized_model():
    print("=== PAIBox MNIST FC 量化反解模型推理测试 (Dequantize back to FP32) ===")
    device = torch.device("cpu")

    # 使用部署用拓扑，消除 BN 造成的干扰
    model = QuantizedMnistFCDeployModel().eval()

    print("[1] 绑定导出的量化权重和偏置，并执行反量化 (Dequantize) ...")
    params_mapping = [
        (model.layer1, "manual_quant_layer1_0"),
        (model.layer2, "manual_quant_layer2_0"),
        (model.classifier, "manual_quant_classifier"),
    ]

    for layer, prefix in params_mapping:
        w_q, b_q = _load_export_pair(EXPORT_DIR, prefix)
        s_in, s_w = DEQUANT_SCALES[prefix]
        _bind_exported_params(layer, w_q, b_q, s_in, s_w)
        print(f"    - Layer [{prefix}] loaded and dequantized successfully.")

    model.to(device)

    # 2. 评测精度
    print("\n[2] 准备 Dataset 并执行评估...")
    _, _, test_loader = build_dataloaders(
        DATA_DIR, batch_size=256, val_split=0.1, num_workers=2)
    criterion = nn.CrossEntropyLoss()

    loss, acc = evaluate(model, test_loader, criterion, device)

    print("\n" + "="*50)
    print("===> 最终反量化精度复测报告:")
    print(f"     Dequantized Load Eval Acc: {acc:.2f}%")
    print(f"     Dequantized Load Eval Loss: {loss:.4f}")
    print("="*50)


if __name__ == "__main__":
    test_quantized_model()
