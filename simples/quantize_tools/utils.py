
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from paibox.fx_converter.lut_activation import LutReLU
    HAS_LUT = True
except ImportError:
    HAS_LUT = False
    LutReLU = None


def quantize_to_int(t, scale, zero_point, qmin, qmax, dtype=torch.uint8):
    """量化到整数域 (返回指定整数类型的 Tensor)"""
    if scale == 0:
        return t
    t_q = torch.round(t / scale) + zero_point
    t_q = torch.clamp(t_q, qmin, qmax)
    return t_q.to(dtype)


def get_obs_params(observer):
    """从 Observer 中获取 scale 和 zero_point"""
    if hasattr(observer, "calculate_qparams"):
        s, zp = observer.calculate_qparams()
        if s.numel() == 1:
            return s.item(), zp.item()
        return s, zp
    return 1.0, 0


def save_checkpoint(model, path="model.pth"):
    """保存模型参数和权重"""
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)
    print(f"模型已保存到 {path}")


def save_weights_separately(model, path="model_weights"):
    """分别保存每层的权重"""
    for name, param in model.named_parameters():
        weight_path = f"{path}_{name}.pth"
        torch.save(param.data, weight_path)
        print(f"权重已保存: {weight_path}")


def approximate_scale_ratio(scale_ratio, min_n=-32, max_n=31, max_m=255):
    """寻找最佳的 M 和 n，使得 M * (2^n) 最逼近 scale_ratio"""
    if isinstance(scale_ratio, torch.Tensor):
        scale_ratio = scale_ratio.item()

    min_error = float('inf')
    best_m = 1
    best_n = 0
    for n in range(min_n, max_n + 1):
        m = int(round(scale_ratio * (2.0 ** -n)))
        if m == 0:
            continue
        if 0 < m <= max_m:
            error = abs(scale_ratio - m * (2.0 ** n))
            if error < min_error:
                min_error = error
                best_m = m
                best_n = n
    return best_m, best_n


if __name__ == "__main__":
    a, b = approximate_scale_ratio(1/9)
    print(f"最佳 M: {a}, 最佳 n: {b}, 近似值: {a * (2.0 ** b)}")
