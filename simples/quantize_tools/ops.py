
import torch
import torch.nn as nn
import torch.nn.functional as F
from paibox.paiir.ir.lut_activation import LutReLU, LutReLUSymmetric, LutLinear

from .utils import quantize_to_int

HAS_LUT = True


def quantized_conv2d_asymmetric(self, x_q, weight_q, x_scale, w_scale,
                                x_zero_point, w_zero_point, bias):
    """执行非对称量化卷积 (4-term decomposition)"""
    # x_q: uint8, weight_q: int8
    x_int = x_q.float()
    weight_int = weight_q.float()

    # 1. conv(x, w)
    term1 = F.conv2d(x_int, weight_int, bias=None,
                     stride=self.stride, padding=self.padding, groups=self.groups)

    # 2. conv(x, w_zp) -> w_zp is scalar per tensor
    # weight_zero_t = torch.full_like(weight_int, w_zero_point)
    # term2 = F.conv2d(x_int, weight_zero_t, ...)
    # 优化: constant kernel convolution
    # w_zp * sum(x_window)
    # 简单起见，这里直接用 full tensor，实际部署可以优化
    weight_zero_t = torch.ones_like(weight_int) * w_zero_point
    term2 = F.conv2d(x_int, weight_zero_t, stride=self.stride,
                     padding=self.padding, groups=self.groups)

    # 3. conv(x_zp, w) -> x_zp * sum(w)
    # sum per output channel
    weight_sum = weight_int.sum(dim=(1, 2, 3)).view(
        1, self.out_channels, 1, 1)
    term3 = x_zero_point * weight_sum

    # 4. conv(x_zp, w_zp)
    kh, kw = self.kernel_size if isinstance(
        self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
    ic_per_group = self.in_channels // self.groups
    term4 = x_zero_point * w_zero_point * (ic_per_group * kh * kw)

    output_acc = term1 - term2 - term3 + term4
    output_scale = x_scale * w_scale

    if bias is not None:
        # bias 通常量化为 scale = s_in * s_w
        bias_q = torch.round(bias / output_scale).view(1, -1, 1, 1)
        output_acc = output_acc + bias_q

    return output_acc, output_scale


def quantized_linear_asymmetric(x_q, weight_q, x_scale, w_scale,
                                x_zero_point, w_zero_point, bias):
    x_int = x_q.float()
    weight_int = weight_q.float()

    term1 = F.linear(x_int, weight_int)

    weight_zero_t = torch.ones_like(weight_int) * w_zero_point
    term2 = F.linear(x_int, weight_zero_t)

    weight_sum = weight_int.sum(dim=1).view(1, -1)
    term3 = x_zero_point * weight_sum

    in_features = weight_int.shape[1]
    term4 = x_zero_point * w_zero_point * in_features

    output_acc = term1 - term2 - term3 + term4
    output_scale = x_scale * w_scale

    if bias is not None:
        bias_q = torch.round(bias / output_scale).view(1, -1)
        output_acc = output_acc + bias_q

    return output_acc, output_scale


class ManualQuantConvReLU2d(nn.Module):
    def __init__(self, original_module, s_in, z_in, s_w, z_w, s_out, z_out, activation_symmetric=False):
        super().__init__()
        self.in_channels = original_module.in_channels
        self.out_channels = original_module.out_channels
        self.kernel_size = original_module.kernel_size
        self.stride = original_module.stride
        self.padding = original_module.padding
        self.groups = original_module.groups
        self.bias_val = original_module.bias

        # 量化参数
        self.activation_symmetric = activation_symmetric
        self.s_in, self.z_in = s_in, z_in
        self.s_w, self.z_w = s_w, z_w
        self.s_out, self.z_out = s_out, z_out

        # 预先量化权重
        self.weight_q = quantize_to_int(
            original_module.weight, s_w, z_w, -128, 127, torch.int8)

        s_accum = s_in * s_w
        lut_scale = s_out / s_accum if s_accum != 0 else 0

        if self.activation_symmetric:
            self.lut = LutReLUSymmetric(
                min_val=-lut_scale*128, max_val=lut_scale*127, output_sign=1)
        else:
            self.lut = LutReLU(
                min_val=-5, max_val=lut_scale*255, output_sign=0)

    def forward(self, x):
        # 1. 检查输入类型，如果是 Float 则第一层量化
        if x.dtype == torch.float32:
            if self.activation_symmetric:
                x = quantize_to_int(
                    x, self.s_in, self.z_in, -128, 127, torch.int8)
            else:
                x = quantize_to_int(x, self.s_in, self.z_in,
                                    0, 255, torch.uint8)

        # 2. 卷积计算 (返回 Accumulator)
        out_acc, out_scale = quantized_conv2d_asymmetric(
            self, x, self.weight_q, self.s_in, self.s_w, self.z_in, self.z_w, self.bias_val
        )

        # 3. 激活与重量化

        out_q = self.lut(out_acc)
        target_dtype = torch.int8 if self.activation_symmetric else torch.uint8
        if out_q.dtype != target_dtype:
            out_q = out_q.to(target_dtype)

        return out_q


class ManualQuantLinear(nn.Module):
    def __init__(self, original_module, s_in, z_in, s_w, z_w, s_out, z_out, activation_symmetric=False):
        super().__init__()
        self.bias_val = original_module.bias
        self.activation_symmetric = activation_symmetric
        self.s_in, self.z_in = s_in, z_in
        self.s_w, self.z_w = s_w, z_w
        self.s_out, self.z_out = s_out, z_out

        self.weight_q = quantize_to_int(
            original_module.weight, s_w, z_w, -128, 127, torch.int8)
        
        s_accum = s_in * s_w
        lut_scale = s_out / s_accum if s_accum != 0 else 0

        self.lut = LutLinear(min_val=-lut_scale*128,
                             max_val=lut_scale*127, output_sign=1)
        

    def forward(self, x):
        if x.dtype == torch.float32:
            if self.activation_symmetric:
                x = quantize_to_int(
                    x, self.s_in, self.z_in, -128, 127, torch.int8)
            else:
                x = quantize_to_int(x, self.s_in, self.z_in,
                                    0, 255, torch.uint8)

        out_acc, out_scale = quantized_linear_asymmetric(
            x, self.weight_q, self.s_in, self.s_w, self.z_in, self.z_w, self.bias_val
        )

        
        out_float = out_acc * out_scale
        return out_float


class ManualQuantLinearReLU(nn.Module):
    def __init__(self, original_module, s_in, z_in, s_w, z_w, s_out, z_out, activation_symmetric=False):
        super().__init__()
        self.bias_val = original_module.bias

        # 量化参数
        self.activation_symmetric = activation_symmetric
        self.s_in, self.z_in = s_in, z_in
        self.s_w, self.z_w = s_w, z_w
        self.s_out, self.z_out = s_out, z_out

        # 预先量化权重
        self.weight_q = quantize_to_int(
            original_module.weight, s_w, z_w, -128, 127, torch.int8)

        s_accum = s_in * s_w
        lut_scale = s_out / s_accum if s_accum != 0 else 0

        if self.activation_symmetric:
            self.lut = LutReLUSymmetric(
                min_val=-lut_scale*128, max_val=lut_scale*127, output_sign=1)
        else:
            self.lut = LutReLU(
                min_val=-5, max_val=lut_scale*255, output_sign=0)

    def forward(self, x):
        # 1. 检查输入类型，如果是 Float 则第一层量化
        if x.dtype == torch.float32:
            if self.activation_symmetric:
                x = quantize_to_int(
                    x, self.s_in, self.z_in, -128, 127, torch.int8)
            else:
                x = quantize_to_int(x, self.s_in, self.z_in,
                                    0, 255, torch.uint8)

        # 2. 线性计算 (返回 Accumulator)
        out_acc, out_scale = quantized_linear_asymmetric(
            x, self.weight_q, self.s_in, self.s_w, self.z_in, self.z_w, self.bias_val
        )

        # 3. 激活与重量化
        out_q = self.lut(out_acc)
        target_dtype = torch.int8 if self.activation_symmetric else torch.uint8
        if out_q.dtype != target_dtype:
            out_q = out_q.to(target_dtype)

        return out_q


class ManualQuantConv2d(nn.Module):
    def __init__(self, original_module, s_in, z_in, s_w, z_w, s_out, z_out, activation_symmetric=False,
                 ):
        super().__init__()
        self.in_channels = original_module.in_channels
        self.out_channels = original_module.out_channels
        self.kernel_size = original_module.kernel_size
        self.stride = original_module.stride
        self.padding = original_module.padding
        self.groups = original_module.groups
        self.bias_val = original_module.bias

        # 量化参数
        self.s_in, self.z_in = s_in, z_in
        self.s_w, self.z_w = s_w, z_w
        self.s_out, self.z_out = s_out, z_out

        # 预先量化权重
        self.weight_q = quantize_to_int(
            original_module.weight, s_w, z_w, -128, 127, torch.int8)

        s_accum = s_in * s_w
        lut_scale = s_out / s_accum if s_accum != 0 else 0

        self.lut = LutLinear(min_val=-lut_scale*128,
                             max_val=lut_scale*127, output_sign=1)

    def forward(self, x):
        # 1. 检查输入类型，如果是 Float 则第一层量化
        if x.dtype == torch.float32:
            x = quantize_to_int(x, self.s_in, self.z_in, 0, 255, torch.uint8)

        # 2. 卷积计算 (返回 Accumulator)
        out_acc, out_scale = quantized_conv2d_asymmetric(
            self, x, self.weight_q, self.s_in, self.s_w, self.z_in, self.z_w, self.bias_val
        )

        out_q = self.lut(out_acc)
        target_dtype = torch.int8
        if out_q.dtype != target_dtype:
            out_q = out_q.to(target_dtype)

        return out_q


class ManualIntAddResidual(nn.Module):
    """
    通用的纯 INT 加法层(含卷积)，自动接管 conv2 的权重和各种量化尺度。
    """

    def __init__(self, original_conv2, y_in_scale, y_in_zp, w_scale, w_zp, conv2_out_scale, out_scale, out_zp, x_scale, x_zp, activation_symmetric=False):
        super().__init__()
        self.conv = ManualQuantConv2d(
            original_module=original_conv2,
            s_in=y_in_scale, z_in=y_in_zp,
            s_w=w_scale, z_w=w_zp,
            s_out=conv2_out_scale, z_out=0,
            activation_symmetric=activation_symmetric
        )
        self.activation_symmetric = activation_symmetric
        self.conv2_out_scale = conv2_out_scale
        self.out_scale = out_scale
        self.out_zp = out_zp
        self.x_scale = x_scale
        self.x_zp = x_zp

        s_accum = self.conv.s_in * self.conv.s_w
        lut_scale = out_scale / s_accum if s_accum != 0 else 0

        if activation_symmetric:
            self.lut = LutReLUSymmetric(
                min_val=-lut_scale*128, max_val=lut_scale*127, output_sign=1)
        else:
            self.lut = LutReLU(
                min_val=-5, max_val=lut_scale*255, output_sign=0)

    def forward(self, y, x):
        from .utils import approximate_scale_ratio

        # shortcut 分支: 先转到本分支的量化整型域，再重标定到 accumulator 域
        if isinstance(x, torch.Tensor) and x.is_quantized:
            x_int = x.int_repr().to(torch.float32)
        else:
            if x.dtype == torch.float32:
                if self.activation_symmetric:
                    x_q = quantize_to_int(
                        x, self.x_scale, self.x_zp, -128, 127, torch.int8)
                else:
                    x_q = quantize_to_int(
                        x, self.x_scale, self.x_zp, 0, 255, torch.uint8)
                x_int = x_q.to(torch.float32)
            else:
                x_int = x.to(torch.float32)

        target_scale = self.conv.s_in * self.conv.s_w
        exact_ratio = self.x_scale / target_scale if target_scale != 0 else 0
        M, n = approximate_scale_ratio(exact_ratio)

        q2_int = torch.round((x_int - self.x_zp) * M *
                             (2.0 ** n)).to(torch.int32)

        # conv 分支: 直接拿 accumulator，避免与 shortcut 量纲不一致
        if isinstance(y, torch.Tensor) and y.is_quantized:
            y_int = y.int_repr().to(torch.float32)
        else:
            if y.dtype == torch.float32:
                if self.activation_symmetric:
                    y_q = quantize_to_int(
                        y, self.conv.s_in, self.conv.z_in, -128, 127, torch.int8)
                else:
                    y_q = quantize_to_int(
                        y, self.conv.s_in, self.conv.z_in, 0, 255, torch.uint8)
                y_int = y_q.to(torch.float32)
            else:
                y_int = y.to(torch.float32)

        out_conv_acc, _ = quantized_conv2d_asymmetric(
            self.conv,
            y_int,
            self.conv.weight_q,
            self.conv.s_in,
            self.conv.s_w,
            self.conv.z_in,
            self.conv.z_w,
            self.conv.bias_val,
        )
        out_conv_acc = torch.round(out_conv_acc).to(torch.int32)

        q_add_int = out_conv_acc + q2_int
        q_relu_int = self.lut(q_add_int)

        target_dtype = torch.int8 if self.activation_symmetric else torch.uint8
        if q_relu_int.dtype != target_dtype:
            q_relu_int = q_relu_int.to(target_dtype)

        # 直接返回纯张量
        return q_relu_int
