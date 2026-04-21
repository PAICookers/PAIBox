import torch
import torch.nn as nn
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig_mapping
import sys
from model import ResNetMNIST
import os


sys.path.append(os.path.dirname(__file__))


def main():
    # 1. 实例化并加载预训练权重（如果存在的话，这里为了演示直接使用初始权重）
    model = ResNetMNIST()
    model.eval()

    # 加载权重（假设有保存最近或者最好的模型，否则用默认初始化）
    ckpt_path = os.path.join(os.path.dirname(
        __file__), "checkpoints", "best_model.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"Loaded weights from {ckpt_path}")
    else:
        print("No pretrained weights found, using random initialization.")

    # 2. 设置量化配置 (FX 图匹配量化)
    # 给所有模块配置全局通用的 QConfig (使用fbgemm在x86 CPU上效果更好)
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")

    # 定义 dummy input 用于 tracing
    dummy_input = (torch.randn(1, 1, 28, 28),)

    # 3. Prepare: 插入 Observer 来统计 min/max，并推断量化参数
    prepared_model = prepare_fx(model, qconfig_mapping, dummy_input)

    # 4. 校准 (Calibration)
    print("Calibrating...")
    for _ in range(10):  # 用少量数据即可，实际应该用训练/验证集数据
        dummy_data = torch.randn(1, 1, 28, 28)
        prepared_model(dummy_data)

    # 5. Convert: 将模型转换为量化版，把Observer换成具体的Quantize节点
    quantized_model = convert_fx(prepared_model)

    print("\n--- Quantization Info (Scale & Zero Point) ---")

    # 打印转换后的 FX Graph 节点来解析每一层的 scale 和 zp
    named_modules = dict(quantized_model.named_modules())
    for node in quantized_model.graph.nodes:
        if node.op == "call_function" and node.target.__name__ in ("quantize_per_tensor", "quantize_per_channel"):
            # 对于 quantize_per_tensor 节点，也就是输入的量化阶段
            scale_node = node.args[1]
            zp_node = node.args[2]
            scale = getattr(quantized_model, scale_node.target) if hasattr(
                scale_node, 'target') else scale_node
            zp = getattr(quantized_model, zp_node.target) if hasattr(
                zp_node, 'target') else zp_node
            print(f"Input Quantize [{node.name}]: Scale={scale}, ZP={zp}")

        elif node.op == "call_function" and "quantized" in str(node.target):
            # 对于量化的算子 (如 quantized::add, quantized::max_pool2d 等)
            scale = None
            zp = None
            # 分析量化算子的输出 scale 和 zp
            if hasattr(node.target, "__name__") and "add" in node.target.__name__:
                scale_node, zp_node = node.args[2], node.args[3]
                scale = getattr(quantized_model, scale_node.target) if hasattr(
                    scale_node, 'target') else scale_node
                zp = getattr(quantized_model, zp_node.target) if hasattr(
                    zp_node, 'target') else zp_node
                print(
                    f"Layer [{node.name}] (quantized.add) -> Output Scale={scale}, Output ZP={zp}")

        elif node.op == "call_module":
            mod = named_modules[node.target]
            # 如果是量化后的层 (如 QuantizedConv2d, QuantizedLinear)
            if hasattr(mod, "weight") and hasattr(mod.weight(), "q_scale"):
                w_scale = mod.weight().q_scale() if mod.weight().qscheme(
                ) == torch.per_tensor_affine else mod.weight().q_per_channel_scales()
                w_zp = mod.weight().q_zero_point() if mod.weight().qscheme(
                ) == torch.per_tensor_affine else mod.weight().q_per_channel_zero_points()

                out_scale = getattr(mod, "scale", None)
                out_zp = getattr(mod, "zero_point", None)

                print(f"Layer [{node.target}] ({type(mod).__name__}):")
                if isinstance(w_scale, torch.Tensor) and w_scale.numel() > 1:
                    print(
                        f"  -> Weight per-channel Scale shape: {w_scale.shape}")
                else:
                    print(f"  -> Weight Scale={w_scale}, ZP={w_zp}")

                if out_scale is not None:
                    print(f"  -> Output Scale={out_scale}, ZP={out_zp}")

    print("\n--- Model structure (quantized graph) ---")
    print(quantized_model)


if __name__ == "__main__":

    main()
