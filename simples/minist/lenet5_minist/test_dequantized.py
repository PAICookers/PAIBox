"""
Evaluate exported symmetric quantized parameters on LeNet5 (MNIST).
Dequantizes Int8/Int32 back to FP32 for a simulated FP32-level accuracy test.
"""

from train import DATA_DIR, build_dataloaders, evaluate
from model import LeNet5
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# 确保能找到 paibox 和简单的上级目录
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..")))


# ---------------------------------------------------------------------------
# Core Configuration
# ---------------------------------------------------------------------------

EXPORT_DIR = BASE_DIR / "exported_params_symmetric"

# 从部署日志获取的量化 Scale (in_scale, w_scale, out_scale)
LAYER_SCALES = {
    "features_0": (0.022304, 0.002591, 0.031766),
    "features_3": (0.031766, 0.002261, 0.037156),
    "classifier_0": (0.037156, 0.004294, 0.068156),
    "classifier_2": (0.068156, 0.005105, 0.129020),
    "classifier_4": (0.129020, 0.005000, 1.0),  # 最后一层无激活，仅记录输入及近似权重的 scale
}

# ---------------------------------------------------------------------------
# Component Shim & Loader
# ---------------------------------------------------------------------------


def _load_int_params(prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads Int8 weight and Int32 bias tensors from the export directory."""
    weight_path = EXPORT_DIR / f"manual_quant_{prefix}_weight_int8.pth"
    bias_path = EXPORT_DIR / f"manual_quant_{prefix}_bias_int32.pth"

    if not (weight_path.exists() and bias_path.exists()):
        raise FileNotFoundError(
            f"Missing symmetric exported files for {prefix}.")

    return (torch.load(weight_path, map_location="cpu"),
            torch.load(bias_path, map_location="cpu"))


def _inject_dequantized_weights(module: nn.Module, weight_q: torch.Tensor,
                                bias_q: torch.Tensor, s_in: float, s_w: float) -> None:
    """
    Dequantizes integer parameters back to FP32 using the standard affine equations:
      - FP32 Weight = Int8 * scale_w
      - FP32 Bias = Int32 * (scale_in * scale_w)
    """
    with torch.no_grad():
        # 反量化操作，放回网络权重
        module.weight.copy_(weight_q.float() * s_w)
        if module.bias is not None:
            module.bias.copy_(bias_q.float() * (s_in * s_w))

# ---------------------------------------------------------------------------
# Test Evaluation
# ---------------------------------------------------------------------------


def construct_eval_model() -> LeNet5:
    """Constructs the LeNet5 evaluating model with DEQUANTIZED weights."""
    model = LeNet5().eval()

    # Layer topology mappings mapping
    # (target layer, corresponding relu to replace, prefix key)
    layer_map = [
        (model.features[0], model.features[1], "features_0"),
        (model.features[3], model.features[4], "features_3"),
        (model.classifier[0], model.classifier[1], "classifier_0"),
        (model.classifier[2], model.classifier[3], "classifier_2"),
        (model.classifier[4], None, "classifier_4"),
    ]

    for conv_linear, relu, prefix in layer_map:
        s_in, s_w, s_out = LAYER_SCALES[prefix]

        # 读取对应前缀参数反量化并赋给模型层
        w_q, b_q = _load_int_params(prefix)
        _inject_dequantized_weights(conv_linear, w_q, b_q, s_in, s_w)
        print(f"  [+] Loaded & Dequantized: [{prefix}]")

    return model


def main() -> None:
    print("=== PAIBox MNIST LeNet-5 Dequantized Simulation Test ===")
    device = torch.device("cpu")

    print("\n[1] Building evaluation model mapped with FP32 de-quantized parameters...")
    model = construct_eval_model().to(device)

    print("\n[2] Loading MNIST Test Dataloader...")
    _, _, test_loader = build_dataloaders(
        DATA_DIR, batch_size=256, val_split=0.1, num_workers=2)
    criterion = nn.CrossEntropyLoss()

    print("\n[3] Evaluating Model...")
    loss, acc = evaluate(model, test_loader, criterion, device=device)

    print("\n" + "=" * 55)
    print("=== Int8 Dequantized Model Accuracy Report ===")
    print(f"      Eval Accuracy: {acc:.2f}%")
    print(f"      Eval Loss:     {loss:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
