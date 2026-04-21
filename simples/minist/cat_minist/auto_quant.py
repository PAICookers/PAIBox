"""
MNIST Cat 网络自动量化脚本

功能:
    1. 加载训练好的 FP32 模型
    2. 配置对称量化策略
    3. 使用训练集的少量样本做校准
    4. 将 FX 图转换为手动量化模型
    5. 评估量化后精度并导出参数
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.ao.quantization import (
    QConfig,
    MinMaxObserver,
    get_default_qconfig_mapping,
    quantize_fx,
)
from torch.ao.quantization.backend_config import get_qnnpack_backend_config
from torch.ao.quantization.quantize_fx import fuse_fx

from model import MNISTCatNet
from paibox.paiir.lowering.converter import propagate_shapes
from simples.quantize_tools import convert_fx_to_manual, export_manual_model_params
from train import DATA_DIR, build_dataloaders, evaluate


def get_quantization_config() -> tuple[object, bool]:
    """配置对称量化策略。"""
    qconfig = QConfig(
        activation=MinMaxObserver.with_args(
            dtype=torch.quint8,
            qscheme=torch.per_tensor_symmetric,
        ),
        weight=MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
        ),
    )

    qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    qconfig_mapping.set_object_type(nn.Conv2d, qconfig)
    qconfig_mapping.set_object_type(nn.Linear, qconfig)
    qconfig_mapping.set_object_type(nn.ReLU, qconfig)
    return qconfig_mapping, True


def main() -> None:
    print("=== MNISTCatNet 自动对称量化 ===\n")
    device = torch.device("cpu")

    print("[1] 正在加载数据与 FP32 模型...")
    train_loader, val_loader, test_loader = build_dataloaders(
        DATA_DIR,
        batch_size=256,
        val_split=0.1,
        num_workers=2,
    )

    model_path = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"找不到预训练文件，请先运行 train.py: {model_path}"
        )

    fp32_model = MNISTCatNet(num_classes=10)
    fp32_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    fp32_model.eval()
    fp32_model.to(device)

    criterion = nn.CrossEntropyLoss()
    _, fp32_acc = evaluate(fp32_model, test_loader, criterion, device)
    print(f"原始 FP32 精度: {fp32_acc:.2f}%")

    print("\n[原始 FP32 Model FX Graph]")
    traced_fp32 = torch.fx.symbolic_trace(fp32_model)
    traced_fp32.graph.print_tabular()

    print("\n[2] Preparing FX (插入 Observer)...")
    qconfig_mapping, activation_symmetric = get_quantization_config()
    example_inputs = torch.randn(1, 1, 28, 28).to(device)
    fused_model = fuse_fx(fp32_model)
    prepared_model = quantize_fx.prepare_fx(
        fused_model,
        qconfig_mapping,
        (example_inputs,),
        backend_config=get_qnnpack_backend_config(),
    )
    print("准备好的 FX 模型图结构 (带 Observer):")
    prepared_model.graph.print_tabular()

    print("\n[3] 正在进行校准 (Calibration)...")
    num_calib_batches = 10
    with torch.no_grad():
        for batch_index, (images, _) in enumerate(train_loader):
            if batch_index >= num_calib_batches:
                break
            prepared_model(images.to(device))

    print("\n[4] 正在将 FX 模型转化为手动量化模型...")
    manual_model = convert_fx_to_manual(
        prepared_model,
        use_lut=True,
        activation_symmetric=activation_symmetric,
    )
    manual_model.eval()

    propagate_shapes(manual_model, example_inputs)
    print("生成的目标底层模型图结构：")
    manual_model.graph.print_tabular()

    print("\n[5] 正在评估量化后精度...")
    _, q_acc = evaluate(manual_model, test_loader, criterion, device)

    print("\n" + "=" * 50)
    print("===> 最终精度对比报告:")
    print(f"     FP32 (高精度):    {fp32_acc:.2f}%")
    print(f"     Symmetric Int8:   {q_acc:.2f}%")
    print("=" * 50)

    print("\n[6] 导出量化参数与权重...")
    export_dir = os.path.join(BASE_DIR, "exported_params_symmetric")
    os.makedirs(export_dir, exist_ok=True)
    export_manual_model_params(manual_model, export_dir)
    print(f"[Done] 量化提取文件已保存到: {export_dir}")


if __name__ == "__main__":
    main()