import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import SimpleVGG

try:
    from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight, int8_weight_only
except ImportError:
    print("请先安装 torchao 库: pip install torchao")
    exit(1)

# 全局路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
WEIGHT_PATH = os.path.join(CKPT_DIR, "best_model.pth")
SAVE_PATH = os.path.join(CKPT_DIR, "quantized_model_torchao.pth")


def evaluate_model(model, loader, device):
    """评估模型准确率和推理耗时"""
    model.eval()
    correct = 0
    total = 0

    # 预热一次
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            _ = model(images)
            break

    start_time = time.time()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    inf_time = time.time() - start_time
    return 100.0 * correct / total, inf_time


def export_layer_data(model, loader, device, save_dir):
    """导出每一层的权重并打印激活/权重的详细校准参数（min, max, scale, zp）"""
    print(f"\n开始提取校准参数并导出权重到目录: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    activation_data = {}

    def calc_asym_scale_zp(t_min, t_max):
        """非对称量化 (UINT8: 0 ~ 255)"""
        scale = (t_max - t_min) / 255.0 if t_max > t_min else 1.0
        zp = int(round(-t_min / scale)) if scale != 0 else 0
        zp = max(0, min(255, zp))
        return scale, zp

    def calc_sym_scale_zp(t_min, t_max):
        """对称量化 (INT8: -127 ~ 127)"""
        amax = max(abs(t_min), abs(t_max))
        scale = amax / 127.0 if amax > 0 else 1.0
        return scale, 0

    def get_activation_hook(name):
        def hook(module, inputs, output):
            if name not in activation_data:
                # 获取 Input
                inp = inputs[0].detach()
                inp_min, inp_max = inp.min().item(), inp.max().item()
                in_scale, in_zp = calc_asym_scale_zp(inp_min, inp_max)

                # 获取 Output
                out = output.detach()
                out_min, out_max = out.min().item(), out.max().item()
                out_scale, out_zp = calc_asym_scale_zp(out_min, out_max)

                activation_data[name] = {
                    "in_scale": in_scale, "in_zp": in_zp,
                    "out_scale": out_scale, "out_zp": out_zp,
                    "out_min": out_min, "out_max": out_max
                }
        return hook

    # 1. 注册 forward hooks (仅拦截核心计算层)
    hooks = []
    layer_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) or "Linear" in module.__class__.__name__ or "Conv" in module.__class__.__name__:
            if name == "":
                continue
            layer_modules[name] = module
            hooks.append(module.register_forward_hook(
                get_activation_hook(name)))

    # 2. 前向传播提取激活 max/min
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            _ = model(images)
            break

    for h in hooks:
        h.remove()

    # 3. 提取权重 max/min 和保存权重
    state_dict = model.state_dict()
    # 为了安全推算权重 min/max 并避免 torchao 张量子类对部分算子的限制，直接从原始浮点权重读取统计数据
    orig_state_dict = torch.load(WEIGHT_PATH, map_location="cpu")

    print("\n" + "="*120)
    print(f"{'Layer':<15} | {'Out Min':<10} | {'Out Max':<10} | {'In Scale':<10} | {'In ZP':<6} | {'Out Scale':<10} | {'Out ZP':<6} | {'W Scale':<10} | {'W ZP':<6}")
    print("-" * 120)

    for layer_name in activation_data.keys():
        calib = activation_data[layer_name]

        # 处理并计算权重的 scale 和 zp
        w_scale, w_zp = 0.0, 0
        weight_key = f"{layer_name}.weight"

        # 如果能找到原始浮点权重，直接从 FP32 原始权重计算校准系数
        if weight_key in orig_state_dict:
            wq_float = orig_state_dict[weight_key]
            w_min, w_max = wq_float.min().item(), wq_float.max().item()
            w_scale, w_zp = calc_sym_scale_zp(w_min, w_max)

        # 打印综合表格
        print(f"{layer_name:<15} | "
              f"{calib['out_min']:>10.4f} | {calib['out_max']:>10.4f} | "
              f"{calib['in_scale']:>10.6f} | {calib['in_zp']:<6d} | "
              f"{calib['out_scale']:>10.6f} | {calib['out_zp']:<6d} | "
              f"{w_scale:>10.6f} | {w_zp:<6d}")

        # 仅保存每层权重
        safe_name = layer_name.replace(".", "_")
        bias_key = f"{layer_name}.bias"
        if weight_key in state_dict:
            torch.save(state_dict[weight_key], os.path.join(
                save_dir, f"{safe_name}_weight.pt"))
        if bias_key in state_dict:
            torch.save(state_dict[bias_key], os.path.join(
                save_dir, f"{safe_name}_bias.pt"))

    print("="*120)
    print(f"成功导出 {len(activation_data)} 层的权重参数到 {save_dir}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 准备测试数据
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    print(f"加载测试数据集从 {DATA_DIR}...")
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=eval_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=2048, shuffle=False, num_workers=2)

    # 2. 加载原始模型
    print(f"加载原始模型权重: {WEIGHT_PATH}")
    if not os.path.exists(WEIGHT_PATH):
        raise FileNotFoundError(f"找不到权重文件: {WEIGHT_PATH}")

    model = SimpleVGG(num_classes=10)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))
    model.to(device)

    # 3. 评估原始模型
    print("\n评估原始 FP32 模型...")
    orig_acc, orig_time = evaluate_model(model, test_loader, device)
    print(f"原始模型准确率: {orig_acc:.2f}%, 推理总耗时: {orig_time:.4f} 秒")

    # 4. 使用 torchao 进行量化
    # 此处使用经典的 INT8 动态激活 + INT8 权重 (Dynamic Quantization)
    print("\n使用 torchao 进行模型量化 (INT8 Dynamic Activation + INT8 Weight)...")
    quantize_(model, int8_dynamic_activation_int8_weight())

    # 如果仅希望量化权重，可以注释上面那行并使用下面这行：
    # quantize_(model, int8_weight_only())

    # 5. 评估量化后模型
    print("\n评估量化后模型...")
    quant_acc, quant_time = evaluate_model(model, test_loader, device)
    print(f"量化模型准确率: {quant_acc:.2f}%, 推理总耗时: {quant_time:.4f} 秒")

    # 6. 保存量化模型
    # 注意：在较旧版本的 PyTorch 中保存量化模型会有特定的格式，
    # TorchAO 通常通过 standard state dict 保存即可
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n量化模型权重已保存至: {SAVE_PATH}")

    # ===== 新增：导出每一层的参数和校准数据 =====
    export_dir = os.path.join(BASE_DIR, "exported_layers_torchao")
    export_layer_data(model, test_loader, device, export_dir)

    # 7. 比较文件大小
    orig_size = os.path.getsize(WEIGHT_PATH) / 1024.0
    quant_size = os.path.getsize(SAVE_PATH) / 1024.0
    print(f"模型大小对比:")
    print(f"  FP32 原始模型: {orig_size:.2f} KB")
    print(f"  INT8 量化模型: {quant_size:.2f} KB")


if __name__ == "__main__":
    main()
