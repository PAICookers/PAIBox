
import os
import torch
from .ops import ManualQuantConvReLU2d, ManualQuantLinear, ManualQuantLinearReLU


def export_manual_model_params(model, export_dir):
    """
    导出手动量化模型中的 Int8 权重和 Int32 偏置
    """
    os.makedirs(export_dir, exist_ok=True)

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, (ManualQuantConvReLU2d, ManualQuantLinear, ManualQuantLinearReLU)):
            # 名字处理: features.0 -> features_0
            safe_name = name.replace(".", "_")
            print(f"Exporting layer: {name} -> {safe_name}")

            # 1. 导出 Weight (Int8)
            # module.weight_q 已经在 __init__ 中量化为 int8
            w_filename = f"{safe_name}_weight_int8.pth"
            torch.save(module.weight_q, os.path.join(export_dir, w_filename))

            # 2. 导出 Bias (Int32)
            # Bias 量化通常基于 Accumulator Scale = s_in * s_w
            if module.bias_val is not None:
                accum_scale = module.s_in * module.s_w
                if accum_scale != 0:
                    bias_q = torch.round(module.bias_val / accum_scale).int()
                    b_filename = f"{safe_name}_bias_int32.pth"
                    torch.save(bias_q, os.path.join(export_dir, b_filename))

            count += 1

    print(f"Export finished! {count} layers saved to: {export_dir}")
