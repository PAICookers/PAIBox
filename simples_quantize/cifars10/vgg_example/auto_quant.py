"""
SimpleVGG (CIFAR-10) 自动对称量化脚本

功能:
    1. 加载基于 FP32 训练好的最优 SimpleVGG 模型
    2. 配置对称量化 (Symmetric Quantization) 策略
    3. 利用 FX Graph 自动插桩 (Observer) 并使用训练集/验证集数据进行校准
    4. 将 FX 模型转换为完全手动的量化算子 (ManualQuantConv2d / ManualQuantLinear)
    5. 测试量化后的模型精度并导出参数
"""

from models import SimpleVGG
from train import build_dataloaders, evaluate, DATA_DIR
from paibox.paiir.lowering.converter import propagate_shapes
from simples.quantize_tools import (
    convert_fx_to_manual,
    export_manual_model_params,
)
import os
import sys
import torch
import torch.nn as nn
from torch.ao.quantization import (
    get_default_qconfig_mapping,
    QConfigMapping,
    quantize_fx,
    QConfig,
    MinMaxObserver,
    MovingAverageMinMaxObserver
)
from torch.ao.quantization.backend_config import get_qnnpack_backend_config

# 确保能找到我们要引用的外部模块
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..")))

# 从当前目录引用训练时写的工具


def get_quantization_config():
    """配置量化策略"""
    # 自定义 QConfig 以匹配手动量化逻辑
    # 激活值: MinMaxObserver ((max-min)/255)
    # 权重: Per-Tensor Asymmetric (zero_point = 0)
    # qscheme = torch.per_tensor_symmetric  # 对称量化 ，zp= 0
    # qscheme = torch.per_tensor_affine  # 非对称权重量化（zero_point 可不为 0）
    my_qconfig = QConfig(
        activation=MinMaxObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_affine),
        weight=MinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_affine)
    )

    qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    qconfig_mapping.set_object_type(nn.Conv2d, my_qconfig)
    qconfig_mapping.set_object_type(nn.Linear, my_qconfig)
    # Fix: 确保 ReLU 使用相同的 qconfig 以便正确融合
    qconfig_mapping.set_object_type(nn.ReLU, my_qconfig)
    return qconfig_mapping


def main():
    print("=== SimpleVGG 全局对称量化 (Auto Symmetric Quantization) ===\n")
    device = torch.device("cpu")  # 量化验证统一在 CPU 进行

    # 1. 准备数据和模型
    print("[1] 正在加载数据与 FP32 模型...")
    # 这里用训练集的子集进行校准，我们在校准时不要用 test_loader。
    train_loader, val_loader, test_loader = build_dataloaders(
        DATA_DIR, batch_size=256, val_split=0.1, num_workers=2
    )

    # 提取预训练权重
    model_path = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到预训练文件，请先运行 train.py 进行训练: {model_path}")

    fp32_model = SimpleVGG(num_classes=10)
    fp32_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    fp32_model.eval()
    fp32_model.to(device)

    # 评估原始 FP32 模型的准确率
    # 注意: 这里损失函数需要随便配一个用于 evaluate API 调用
    criterion = nn.CrossEntropyLoss()
    _, fp32_acc = evaluate(fp32_model, test_loader, criterion, device)
    print(f"原始 FP32 精度: {fp32_acc:.2f}%")

    print("\n[原始 FP32 Model FX Graph]")
    traced_fp32 = torch.fx.symbolic_trace(fp32_model)
    traced_fp32.graph.print_tabular()

    # 2. 准备配置 QConfig 配置，并使用 FX 插入 Observer (插桩)
    print("\n[2] Preparing FX (配置基于 Observer 的计算图)...")
    qconfig_mapping = get_quantization_config()
    # CIFAR-10 数据样例
    example_inputs = torch.randn(1, 3, 32, 32).to(device)

    prepared_model = quantize_fx.prepare_fx(
        fp32_model, qconfig_mapping, (example_inputs,
                                      ), backend_config=get_qnnpack_backend_config()
    )
    print("准备好的 FX 模型图结构 (带 Observer):")
    prepared_model.graph.print_tabular()

    # 3. 数据校准 (Calibration) -> 获取并计算 tensor statistics，决定 scale 和 zp
    print("\n[3] 正在对模型进行校准运算 (Calibration)... 半自动找截断点")
    # 让一定量的数据跑过网络以记录 min/max，通常不需要很大的 batch 数量
    num_calib_batches = 10
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            if i >= num_calib_batches:
                break
            # 喂入量化校验网络
            prepared_model(images.to(device))

    # 4. 转换模型
    print("\n[4] 正在将带 Observer 的 FX 模型转化为完全离线的自动 ManualQuant 模型...")
    use_lut = True  # 是否开启基于查表的 ReLU，False 表示继续使用标准后端运算
    manual_model = convert_fx_to_manual(prepared_model, use_lut=use_lut)

    # 因为自定义替换掉了一些 op，需要把 Tensor 本身的形状信息在图中重新传播一次，以方便之后可能的拆分映射
    propagate_shapes(manual_model, example_inputs)
    print("生成的目标底层模型图结构：")
    manual_model.graph.print_tabular()

    # 5. 模型推理评估
    print("\n[5] 正在评估合并后的定点网络精度...")
    _, q_acc = evaluate(manual_model, test_loader, criterion, device)

    print("\n" + "="*50)
    print("===> 最终精度对比报告:")
    print(f"     FP32 (高精度):    {fp32_acc:.2f}%")
    print(f"     Symmetric Int8: {q_acc:.2f}%")
    print("="*50)

    # 6. 导出手动量化后的参数及 Bias
    print("\n[6] 导出 Int8 权重及量化参数 (Symmetric)...")
    export_dir = os.path.join(BASE_DIR, "exported_params_symmetric")
    os.makedirs(export_dir, exist_ok=True)
    export_manual_model_params(manual_model, export_dir)
    print(f"[Done] 量化提取文件已成功保存到: {export_dir}")


if __name__ == "__main__":
    main()
