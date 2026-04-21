"""
ResNet (CIFAR-10) 自动对称/非对称量化脚本

功能:
    1. 加载基于 FP32 训练好的最优 ResNetCIFAR10 模型
    2. 配置量化策略
    3. 利用 FX Graph 自动插桩 (Observer) 并使用训练集部分数据进行校准
    4. 将 FX 模型转换为手动的量化算子 (ManualQuant 等)
    5. 测试量化后的模型精度并导出参数
"""

from simples.quantize_tools import (
    convert_fx_to_manual,
    export_manual_model_params,
)
from simples.res_cifar10.model import ResNetCIFAR10
from paibox.fx_converter.trace import propagate_tensor_shape
import os
import sys
import torch
import torch.nn as nn
from torch.ao.quantization import (
    get_default_qconfig_mapping,
    quantize_fx,
    QConfig,
    MinMaxObserver,
)
from torch.ao.quantization.backend_config import get_qnnpack_backend_config
from torch.ao.quantization.quantize_fx import fuse_fx
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# 调整工作目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..")))


def build_dataloaders(data_dir, batch_size=256, val_split=0.1, num_workers=2):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_test)
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)

    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # 兼容可能的字典或元组输出
            if isinstance(output, tuple):
                output = output[0]
            elif isinstance(output, dict):
                output = output[list(output.keys())[0]]

            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            test_correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader)
    val_acc = 100. * test_correct / len(test_loader.dataset)
    return test_loss, val_acc


def get_quantization_config(symmetric=True):
    """配置量化策略"""
    qscheme_act = torch.per_tensor_symmetric if symmetric else torch.per_tensor_affine
    qscheme_weight = torch.per_tensor_symmetric

    my_qconfig = QConfig(
        activation=MinMaxObserver.with_args(
            dtype=torch.quint8, qscheme=qscheme_act),
        weight=MinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=qscheme_weight)
    )

    qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    qconfig_mapping.set_object_type(nn.Conv2d, my_qconfig)
    qconfig_mapping.set_object_type(nn.Linear, my_qconfig)
    qconfig_mapping.set_object_type(nn.ReLU, my_qconfig)

    return qconfig_mapping


def main():
    print(f"=== ResNetCIFAR10 自动量化 ===")
    device = torch.device("cpu")
    symmetric = True  # 设置为对称量化，如需非对称可改为 False

    # 1. 准备数据和模型
    print("[1] 正在加载数据与 FP32 模型...")
    train_loader, val_loader, test_loader = build_dataloaders(
        DATA_DIR, batch_size=256, val_split=0.1, num_workers=2
    )

    model_path = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
    if not os.path.exists(model_path):
        print(f"警告：找不到预训练文件 {model_path}，使用随机初始化模型仅作测试展示。")
        fp32_model = ResNetCIFAR10(num_classes=10)
    else:
        fp32_model = ResNetCIFAR10(num_classes=10)
        checkpoint = torch.load(model_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            fp32_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            fp32_model.load_state_dict(checkpoint)

    fp32_model.eval()
    fp32_model.to(device)

    criterion = nn.CrossEntropyLoss()
    _, fp32_acc = evaluate(fp32_model, test_loader, criterion, device)
    print(f"原始 FP32 精度: {fp32_acc:.2f}%")

    print("\n[原始 FP32 Model FX Graph]")
    traced_fp32 = torch.fx.symbolic_trace(fp32_model)
    traced_fp32.graph.print_tabular()

    # 2. 准备融合与配置 QConfig
    print("\n[2] Preparing FX (配置基于 Observer 的计算图)...")
    qconfig_mapping = get_quantization_config(symmetric)
    example_inputs = torch.randn(1, 3, 32, 32).to(device)

    # 算子融合 (Conv+BN+ReLU 等)
    fp32_model = fuse_fx(fp32_model)

    print("\n[融合后的 FP32 Model FX Graph]")
    traced_fp32 = torch.fx.symbolic_trace(fp32_model)
    traced_fp32.graph.print_tabular()

    prepared_model = quantize_fx.prepare_fx(
        fp32_model, qconfig_mapping, (example_inputs,
                                      ), backend_config=get_qnnpack_backend_config()
    )
    print("准备好的 FX 模型图结构 (带 Observer):")
    prepared_model.graph.print_tabular()

    # 3. 数据校准
    print("\n[3] 正在对模型进行校准运算 (Calibration)...半自动找截断点")
    num_calib_batches = 10
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            if i >= num_calib_batches:
                break
            prepared_model(images.to(device))

    prepared_model.graph.print_tabular()  # 校准后模型图结构展示

    # 4. 转换模型
    print("\n[4] 正在将带 Observer 的 FX 模型转化为完全离线的自动 ManualQuant 模型...")
    manual_model = convert_fx_to_manual(
        prepared_model, use_lut=True, activation_symmetric=symmetric)

    propagate_tensor_shape(manual_model, example_inputs)
    print("生成的目标底层模型图结构：")
    manual_model.graph.print_tabular()

    # 5. 模型推理评估
    print("\n[5] 正在评估合并后的定点网络精度...")
    _, q_acc = evaluate(manual_model, test_loader, criterion, device)

    print("\n" + "="*50)
    print("===> 最终精度对比报告:")
    print(f"     FP32 (高精度):    {fp32_acc:.2f}%")
    mode_name = "Symmetric Int8" if symmetric else "Asymmetric Int8"
    print(f"     {mode_name}: {q_acc:.2f}%")
    print("="*50)

    # 6. 导出手动量化后的参数及 Bias
    print("\n[6] 导出 Int8 权重及量化参数...")
    export_dir = os.path.join(
        BASE_DIR, "exported_params_symmetric" if symmetric else "exported_params_asymmetric")
    os.makedirs(export_dir, exist_ok=True)
    export_manual_model_params(manual_model, export_dir)
    print(f"[Done] 量化提取文件已成功保存到: {export_dir}")


if __name__ == "__main__":
    main()
