import torch
import torch.nn as nn
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig_mapping
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import ResNetMNIST, ResidualBlock
from simples.quantize_tools.ops import ManualQuantConv2d
from simples.quantize_tools.utils import quantize_to_int
import sys
import os

from paibox.fx_converter.lut_activation import LutReLU

# Bypass the OpenMP multiple initialization error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 引入根目录保证能够导入 simples 里的量化工具
ROOT_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)


sys.path.append(os.path.dirname(__file__))


def approximate_scale_ratio(scale_ratio, min_n=-32, max_n=31, max_m=255):
    """
    寻找最佳的 M 和 n，使得 M * (2^n) 最逼近 scale_ratio 
    其中 M 为 uint8 (1~255)，n 为 6 bit 有符号整数 (-32~31)。
    """
    if isinstance(scale_ratio, torch.Tensor):
        scale_ratio = scale_ratio.item()

    min_error = float('inf')
    best_m = 1
    best_n = 0
    for n in range(min_n, max_n + 1):
        # 尝试计算当前的 M。目标：M * 2^n ≈ scale_ratio -> M ≈ scale_ratio * 2^(-n)
        m = int(round(scale_ratio * (2.0 ** -n)))
        if m == 0:  # 避免全 0 的无意义逼近
            continue

        if 0 < m <= max_m:
            error = abs(scale_ratio - m * (2.0 ** n))
            if error < min_error:
                min_error = error
                best_m = m
                best_n = n
    return best_m, best_n


class ManualIntAddResidualBlock(nn.Module):
    """
    计算 x + conv(y) 的纯 INT 加法层。
    1. 使用 conv 输出的 Scale 和 ZP 作为基准，对 x 进行重新量化。
    2. 执行纯整数相加并应用截断式 ReLU。
    3. 将结果转换（重新量化）到原版 add_relu 指定的 target output scale 和 zp 上。
    """

    def __init__(self, original_conv2, y_in_scale, y_in_zp, w_scale, w_zp, conv2_out_scale, out_scale, out_zp):
        super().__init__()
        # self.conv 在内部被替换为 ManualQuantConv2d
        # ManualQuantConv2d 输出假设是围绕 0 对称的 int32 累加器放缩结果 (无 zp)
        self.conv = ManualQuantConv2d(
            original_module=original_conv2,
            s_in=y_in_scale, z_in=y_in_zp,
            s_w=w_scale, z_w=w_zp,
            s_out=conv2_out_scale, z_out=0  # 这里没有 zp
        )
        self.conv2_out_scale = conv2_out_scale
        self.out_scale = out_scale
        self.out_zp = out_zp

        # 结合原本的 s_accum 与目标的 out_scale，使用 LutReLU 处理激活和重量化
        s_accum = self.conv.s_in * self.conv.s_w
        lut_scale = out_scale / s_accum if s_accum != 0 else 0
        self.lut = LutReLU(min_val=-5, max_val=lut_scale*255, output_sign=0)

    def forward(self, x_q, y_q):
        # 1. 提取 x_q 的底层整数值以及原有的量化参数
        x_int = x_q.int_repr().to(torch.float32)  # 先转 float32 保证乘除法精度
        x_scale = x_q.q_scale()
        x_zp = x_q.q_zero_point()

        # Target scale 是卷积累加器的 scale = s_in * s_w
        target_scale = self.conv.s_in * self.conv.s_w

        # 使用定点数 M * (2^n) 来逼近真实的小数 exact_ratio
        # n 为6位有符号整数，可以直接表示左移(n>0)或右移(n<0)
        exact_ratio = x_scale / target_scale
        M, n = approximate_scale_ratio(exact_ratio)
        print(
            f"Approximating scale ratio: {exact_ratio:.6f} ≈ M={M} * 2^{n} = {M * (2.0 ** n):.6f} (Error={abs(exact_ratio - M * (2.0 ** n)):.6f})")

        # 重新根据定点运算公式计算:
        # (x_int - x_zp) * M * (2 ** n)
        q2_int = torch.round((x_int - x_zp) * M * (2.0 ** n)).to(torch.int32)

        # 2. 将上一层的输入 y_q (QuantizedTensor) 转化为底层 uint8 再传入 ManualQuantConv2d
        y_uint8 = y_q.int_repr().to(torch.uint8)
        out_conv_acc = self.conv(y_uint8)

        # 这时的 q_add_int 还是规模较大的 int32 累加器，并没有被挤压到 0-255
        q_add_int = out_conv_acc + q2_int

        # 4 & 5 & 6. 使用 LutReLU 统一进行截断、反量化与重新量化的激活映射
        q_relu_int = self.lut(q_add_int)

        if q_relu_int.dtype != torch.uint8:
            q_relu_int = q_relu_int.to(torch.uint8)

        # 把这封信装进含有 scale 和 zp 的信封里，返回给后续 FX 节点
        out_q = torch._make_per_tensor_quantized_tensor(
            q_relu_int, self.out_scale, self.out_zp)

        return out_q


def test_model(model, test_loader):
    """
    专门用来评估量化模型精度的循环。
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return 100. * correct / total


def main():
    # 1. 正常加载和执行 FX 量化，获取经过初步转换的网络
    model = ResNetMNIST()

    # 加载 best_model 权重

    ckpt_path = os.path.join(os.path.dirname(
        __file__), "checkpoints", "best_model.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"Loaded weights from {ckpt_path}")
    else:
        print(f"Warning: best_model.pth not found at {ckpt_path}")

    # 准备测试数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    test_dataset = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model.eval()
    dummy_input = (torch.randn(1, 1, 28, 28),)
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")
    prepared_model = prepare_fx(model, qconfig_mapping, dummy_input)

    print("Calibrating (using test_loader)...")
    for i, (data, _) in enumerate(test_loader):
        prepared_model(data)
        if i >= 2:  # 用几批数据进行真实的量化参数校准
            break

    quantized_model = convert_fx(prepared_model)

    baseline_acc = test_model(quantized_model, test_loader)
    print(f"\n[测试] --- 官方原生量化模型的测试集精度: {baseline_acc:.2f}% ---")

    # 2. 从图里提取原来 add_relu 的量化参数
    add_relu_scale = quantized_model.res_block_scale_0
    add_relu_zp = quantized_model.res_block_zero_point_0

    # 获取原本的模型 conv2 结构来初始化 ManualQuantConv2d
    orig_conv2 = model.res_block.conv2

    # 为了 ManualQuantConv2d 我们需要提取校准的 Scale 和 ZP (从 quantized_model 提取)
    # 输入(也就是y_q) 的参数
    y_in_scale = quantized_model.res_block.conv1.scale
    y_in_zp = quantized_model.res_block.conv1.zero_point

    # 权重和偏置的参数 (我们通过量化图里的 QConv2d 获取)
    w_scale = quantized_model.res_block.conv2.weight().q_scale() if quantized_model.res_block.conv2.weight().qscheme(
        # 简化
    ) == torch.per_tensor_affine else quantized_model.res_block.conv2.weight().q_per_channel_scales().mean()
    w_zp = quantized_model.res_block.conv2.weight().q_zero_point() if quantized_model.res_block.conv2.weight().qscheme(
        # 简化
    ) == torch.per_tensor_affine else quantized_model.res_block.conv2.weight().q_per_channel_zero_points().float().mean()

    # Conv2 出去的参数 (准备用它作为 x 对齐基准)
    conv2_out_scale = quantized_model.res_block.conv2.scale

    # 3. 构造我们自定义的纯 INT 加法残差块，传入需要的所有尺度参数
    Manualaddrelu = ManualIntAddResidualBlock(
        original_conv2=orig_conv2,
        y_in_scale=y_in_scale, y_in_zp=y_in_zp,
        w_scale=w_scale, w_zp=w_zp,
        conv2_out_scale=conv2_out_scale,
        out_scale=add_relu_scale, out_zp=add_relu_zp
    )

    # 4. 对计算图进行修改，替换掉原图中的 conv2 与 add_relu，加入 custom_res_block
    quantized_model.add_submodule("custom_res_block", Manualaddrelu)

    for node in quantized_model.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.quantized.add_relu:
            # 原节点结构: add_relu = torch.ops.quantized.add_relu(res_block_conv2, conv1, scale, zp)
            # node.args[0] 对应 res_block_conv2 的输出节点
            # node.args[1] 对应捷径层 conv1 的输出节点 (也是 custom 的参数 x_q)
            conv_out_node = node.args[0]
            x_node = node.args[1]

            # y_q 参数对应的是原来 conv2 的输入节点
            y_node = conv_out_node.args[0]

            with quantized_model.graph.inserting_before(node):
                # 注意我们定义的 forward 参数顺序是 (x_q, y_q)
                new_node = quantized_model.graph.call_module(
                    "custom_res_block", args=(x_node, y_node))

            node.replace_all_uses_with(new_node)
            quantized_model.graph.erase_node(node)
            quantized_model.graph.erase_node(conv_out_node)
            break

    quantized_model.graph.lint()
    quantized_model.recompile()

    print("\n--- 替换为纯INT加法后的计算树结构 ---")
    quantized_model.graph.print_tabular()

    # 4. 推理替换操作后的图，测试我们的方法精度有没有下跌
    custom_acc = test_model(quantized_model, test_loader)
    print(f"\n[测试] --- 自定义常数INT加法模型的测试集精度: {custom_acc:.2f}% ---")


if __name__ == "__main__":
    main()
