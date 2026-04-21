"""
convconv MNIST 模型自动量化测试脚本

功能:
    1. 加载训练好的 FP32 ConvBNConvBN 模型
    2. 使用 FX + Observer 执行校准
    3. 使用 simples.quantize_tools 将模型转换为手动量化算子
    4. 对比 FP32 与量化后模型在测试集上的精度
    5. 导出可导出的量化参数文件
"""

from simples.quantize_tools import convert_fx_to_manual, export_manual_model_params
from paibox.paiir.lowering.converter import propagate_shapes
from train import build_dataloaders, evaluate, DATA_DIR
from model import ConvBNConvBN
from torch.ao.quantization.quantize_fx import fuse_fx
from torch.ao.quantization.backend_config import get_qnnpack_backend_config
from torch.ao.quantization import QConfig, MinMaxObserver, get_default_qconfig_mapping, quantize_fx
import torch.nn as nn
import torch
import os
import sys
import argparse

# 必须在导入 torch 之前设置！解决多个 OpenMP 副本导致冲突的报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..")))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="convconv MNIST 模型自动量化测试")
    parser.add_argument("--batch-size", type=int, default=256, help="批大小")
    parser.add_argument("--workers", type=int, default=2,
                        help="DataLoader 工作进程数")
    parser.add_argument("--val-split", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--calib-batches", type=int,
                        default=10, help="校准使用的训练集 batch 数")
    parser.add_argument("--use-lut", action="store_true",
                        help="是否在量化转换中启用 LUT ReLU")
    return parser.parse_args()


def get_quantization_config():
    my_qconfig = QConfig(
        activation=MinMaxObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_symmetric),
        weight=MinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
    )

    qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    qconfig_mapping.set_object_type(nn.Conv2d, my_qconfig)
    qconfig_mapping.set_object_type(nn.Linear, my_qconfig)
    qconfig_mapping.set_object_type(nn.ReLU, my_qconfig)
    return qconfig_mapping


def main() -> None:
    args = get_args()
    device = torch.device("cpu")

    print("=== convconv 自动量化测试 (CPU) ===\n")

    print("[1] 加载数据...")
    train_loader, _, test_loader = build_dataloaders(
        DATA_DIR,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.workers,
    )

    print("[2] 加载 FP32 模型...")
    model_path = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到预训练文件，请先运行 train.py: {model_path}")

    fp32_model = ConvBNConvBN(num_classes=10)
    fp32_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    fp32_model.eval().to(device)

    criterion = nn.CrossEntropyLoss()
    _, fp32_acc = evaluate(fp32_model, test_loader, criterion, device)
    print(f"[结果] FP32 精度: {fp32_acc:.2f}%")

    print("\n[3] 构建 FX 量化准备图...")
    qconfig_mapping = get_quantization_config()
    example_inputs = torch.randn(1, 1, 28, 28).to(device)

    fused_model = fuse_fx(fp32_model)
    prepared_model = quantize_fx.prepare_fx(
        fused_model,
        qconfig_mapping,
        (example_inputs,),
        backend_config=get_qnnpack_backend_config(),
    )
    print("FX 量化准备图打印。")
    prepared_model.graph.print_tabular()

    print("[4] 校准模型...")
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            if i >= args.calib_batches:
                break
            prepared_model(images.to(device))

    print("[5] 转换为手动量化模型...")
    manual_model = convert_fx_to_manual(
        prepared_model,
        use_lut=args.use_lut,
        activation_symmetric=True,
    )
    print("手动量化模型打印。")
    manual_model.graph.print_tabular()

    propagate_shapes(manual_model, example_inputs)

    print("[6] 评估量化模型...")
    _, q_acc = evaluate(manual_model, test_loader, criterion, device)

    print("\n" + "=" * 52)
    print("量化测试结果")
    print(f"FP32 Accuracy : {fp32_acc:.2f}%")
    print(f"INT8 Accuracy : {q_acc:.2f}%")
    print("=" * 52)

    print("\n[7] 导出量化参数...")
    export_dir = os.path.join(BASE_DIR, "exported_params_symmetric")
    os.makedirs(export_dir, exist_ok=True)
    export_manual_model_params(manual_model, export_dir)
    print(f"[完成] 参数导出目录: {export_dir}")


if __name__ == "__main__":
    main()
