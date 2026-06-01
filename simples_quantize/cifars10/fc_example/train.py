"""
CIFAR-10 训练脚本

功能:
    1. 自动下载 CIFAR-10 数据集
    2. 训练三层全连接网络
    3. 每个 epoch 在验证集上评估准确率
    4. 保存最优模型权重到 checkpoints/ 目录

运行方式:
    python train.py
"""

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

from models import CIFAR10_FC

# ─────────────────────────────────────────────
# 全局路径配置
# ─────────────────────────────────────────────
# 当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CIFAR-10 数据集保存目录（与 fc_example 同级的 data/ 文件夹）
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data")
# 模型权重保存目录
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")


def get_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="CIFAR-10 FC 三层网络训练")
    parser.add_argument("--epochs",     type=int,   default=60,    help="训练轮数")
    parser.add_argument("--batch-size", type=int,   default=1024,   help="批大小")
    parser.add_argument("--lr",         type=float,
                        default=1e-3,  help="初始学习率")
    parser.add_argument("--weight-decay", type=float,
                        default=1e-4, help="L2 正则化系数")
    parser.add_argument("--dropout",    type=float,
                        default=0.3,   help="Dropout 概率")
    parser.add_argument("--val-split",  type=float,
                        default=0.1,   help="验证集比例")
    parser.add_argument("--workers",    type=int,
                        default=2,     help="DataLoader 工作进程数")
    parser.add_argument("--seed",       type=int,   default=42,    help="随机种子")
    return parser.parse_args()


def build_dataloaders(
    data_dir: str,
    batch_size: int,
    val_split: float,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    构建训练集、验证集、测试集的 DataLoader。

    CIFAR-10 数据集会在 data_dir 不存在时自动下载。

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # ---------- 数据预处理 ----------
    # 训练集：随机水平翻转 + 随机裁剪做数据增强，再归一化
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),           # 随机水平翻转
        transforms.RandomCrop(32, padding=4),        # 随机裁剪（填内4像素后裁回32）
        transforms.ToTensor(),                        # 转为 Tensor，像素值缩放到 [0,1]
        transforms.Normalize(                         # 归一化到标准正态分布
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    # 验证/测试集：不需要增强，只归一化
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    # ---------- 下载/加载数据集 ----------
    print(f"[数据] 数据目录: {os.path.abspath(data_dir)}")
    full_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=eval_transform
    )

    # ---------- 划分训练集 / 验证集 ----------
    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    train_dataset, val_dataset = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),  # 固定划分，保证可复现
    )
    # 验证集使用 eval_transform（覆盖 full_train 的 train_transform）
    val_dataset.dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=eval_transform
    )

    print(f"[数据] 训练: {n_train} | 验证: {n_val} | 测试: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    训练一个 epoch。

    Returns:
        (avg_loss, accuracy)
    """
    model.train()  # 开启训练模式（BatchNorm / Dropout 生效）
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # 梯度清零
        outputs = model(images)        # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()                # 反向传播
        optimizer.step()               # 更新参数

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    在给定数据集上评估模型。

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()  # 关闭训练模式（BatchNorm / Dropout 固定）
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main() -> None:
    args = get_args()

    # ── 随机种子，保证可复现 ──
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── 设备选择 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用: {device}")

    # ── 权重保存目录 ──
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"[保存] 权重目录: {os.path.abspath(CKPT_DIR)}")

    # ── 数据 ──
    train_loader, val_loader, test_loader = build_dataloaders(
        DATA_DIR, args.batch_size, args.val_split, args.workers
    )

    # ── 模型 ──
    model = CIFAR10_FC(num_classes=10, dropout_p=args.dropout).to(device)
    print(
        f"[模型] 参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── 损失函数 ──
    # CrossEntropyLoss 内部含 Softmax，输出层不需要额外激活
    criterion = nn.CrossEntropyLoss()

    # ── 优化器：Adam + 权重衰减 ──
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ── 学习率调度器：验证集 loss 不下降时自动乘以 0.5 ──
    # patience=5 表示连续 5 个 epoch 没改善就降 lr
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # ── 训练循环 ──
    best_val_acc = 0.0
    best_ckpt_path = os.path.join(CKPT_DIR, "best_model.pth")

    print("\n" + "=" * 60)
    print(
        f"  开始训练: {args.epochs} epochs | batch={args.batch_size} | lr={args.lr}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        # 训练一个 epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        # 在验证集上评估
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        # 根据验证集 loss 动态调整学习率
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}%  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%  "
            f"lr={current_lr:.6f}"
        )

        # 保存当前 epoch 权重
        epoch_ckpt = os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            },
            epoch_ckpt,
        )

        # 保存最优权重（验证集准确率最高）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  ★ 保存最优模型 (val_acc={val_acc:.2f}%) -> {best_ckpt_path}")

    print("\n" + "=" * 60)
    print(f"  训练完成！最优验证集准确率: {best_val_acc:.2f}%")

    # ── 最终测试集评估 ──
    # 加载最优权重
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"  测试集准确率: {test_acc:.2f}%  (loss={test_loss:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
