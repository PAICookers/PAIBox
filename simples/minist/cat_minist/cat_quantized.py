"""
MNIST Cat 量化工作文档示例。

功能：
    1. 读取现有的 FP32 best_model.pth
    2. 使用 FX prepare_fx 插入 Observer 并做校准
    3. 使用 convert_fx_to_manual 得到量化模型
    4. 导出校准数据 JSON
    5. 导出量化后每层的输入 / 输出 / 权重 scale 和 zp
    6. 导出量化之后的计算图 TXT
    7. 将量化模型的 state_dict 导出为 pth
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.ao.quantization import QConfig, MinMaxObserver, get_default_qconfig_mapping
from torch.ao.quantization.backend_config import get_qnnpack_backend_config
from torch.ao.quantization.quantize_fx import fuse_fx, prepare_fx
from torch.utils.data import DataLoader

from paibox.backendv2.mapper import Mapper
from paibox.paiir import compile_to_paiir
from simples.minist.cat_minist.model import MNISTCatNet
from simples.quantize_tools import (
    convert_fx_to_manual,
    convert_ready_paiir,
    export_manual_model_params,
    export_quantized_model_summary,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
for path in (REPO_ROOT, BASE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


DATA_DIR = os.path.join(REPO_ROOT, "data")
SOURCE_CKPT = os.path.join(
    REPO_ROOT, "minist", "cat_minist", "checkpoints", "best_model.pth"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_SUMMARY_JSON = os.path.join(OUTPUT_DIR, "quantized_model_layers.json")
MODEL_GRAPH_TXT = os.path.join(OUTPUT_DIR, "quantized_model_graph.txt")
EXPORT_PARAMS_DIR = os.path.join(OUTPUT_DIR, "exported_params")


def build_calibration_loader(
    data_dir: str,
    batch_size: int = 256,
    num_workers: int = 2,
) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_test_loader(
    data_dir: str,
    batch_size: int = 256,
    num_workers: int = 2,
) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def get_quantization_config(symmetric: bool = True):
    qscheme_act = torch.per_tensor_symmetric if symmetric else torch.per_tensor_affine
    qscheme_weight = torch.per_tensor_symmetric

    my_qconfig = QConfig(
        activation=MinMaxObserver.with_args(
            dtype=torch.quint8,
            qscheme=qscheme_act,
        ),
        weight=MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=qscheme_weight,
        ),
    )

    qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    qconfig_mapping.set_object_type(nn.Conv2d, my_qconfig)
    qconfig_mapping.set_object_type(nn.Linear, my_qconfig)
    qconfig_mapping.set_object_type(nn.ReLU, my_qconfig)
    return qconfig_mapping


def load_fp32_model() -> MNISTCatNet:
    if not os.path.exists(SOURCE_CKPT):
        raise FileNotFoundError(f"找不到预训练文件: {SOURCE_CKPT}")

    model = MNISTCatNet(num_classes=10)
    checkpoint = torch.load(SOURCE_CKPT, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def main() -> None:
    device = torch.device("cpu")
    symmetric = True
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== MNIST Cat FX 量化并导出 pth ===")
    print(f"[1] 读取 FP32 权重: {SOURCE_CKPT}")
    fp32_model = load_fp32_model().to(device)

    print("[2] 构建量化配置和示例输入")
    qconfig_mapping = get_quantization_config(symmetric=symmetric)
    example_inputs = torch.randn(1, 1, 28, 28).to(device)

    print("[3] 融合模型并插入 Observer")
    fused_model = fuse_fx(fp32_model)
    prepared_model = prepare_fx(
        fused_model,
        qconfig_mapping,
        (example_inputs,),
        backend_config=get_qnnpack_backend_config(),
    )

    print("[4] 使用训练集前几个 batch 做校准")
    calib_loader = build_calibration_loader(DATA_DIR)
    calib_batches = 10
    with torch.no_grad():
        for i, (images, _) in enumerate(calib_loader):
            if i >= calib_batches:
                break
            prepared_model(images.to(device))

    print("[5] 转换为量化模型")
    quantized_model = convert_fx_to_manual(
        prepared_model,
        activation_symmetric=symmetric,
    )

    print("[5.5] 测试量化前后精度对比 (MNIST测试集验证可能有一定耗时，请等待...)")
    test_loader = build_test_loader(DATA_DIR)
    fp32_acc = evaluate_model(fp32_model, test_loader, device)
    quantized_acc = evaluate_model(quantized_model, test_loader, device)
    print("=" * 50)
    print(f"  => FP32 模型准确率: {fp32_acc:.2f}%")
    print(f"  => 量化后模型准确率: {quantized_acc:.2f}%")
    print("=" * 50)

    print("[6] 导出量化层参数 JSON 和计算图 TXT")
    export_quantized_model_summary(
        quantized_model,
        MODEL_SUMMARY_JSON,
        MODEL_GRAPH_TXT,
    )
    print(f"[完成] 量化层参数已保存到: {MODEL_SUMMARY_JSON}")
    print(f"[完成] 量化计算图已保存到: {MODEL_GRAPH_TXT}")

    print(f"[7] 导出权重和偏置等参量到文件夹 {EXPORT_PARAMS_DIR}")
    os.makedirs(EXPORT_PARAMS_DIR, exist_ok=True)
    export_manual_model_params(quantized_model, EXPORT_PARAMS_DIR)
    print(f"[完成] 权重及偏置已导出到: {EXPORT_PARAMS_DIR}")

    deploy_model = convert_ready_paiir(quantized_model)

    print("[8] 部署模型: ")
    graph = compile_to_paiir(
        deploy_model,
        example_inputs,
    )
    mapper = Mapper()
    mapper.compile(graph)


if __name__ == "__main__":
    main()
