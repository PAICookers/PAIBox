"""
MNIST 训练脚本 (Conv+BN+Conv+BN)

功能:
    1. 自动下载 MNIST 数据集
    2. 训练 Conv+BN+Conv+BN 网络
    3. 每个 epoch 在验证集上评估准确率
    4. 保存最优模型权重到 checkpoints/ 目录
"""

from model import ConvBNConvBN
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch
import argparse
import os

# 必须在导入 torch 之前设置！解决多个 OpenMP 副本导致冲突的报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MNIST 数据集保存目录（与 minist 同级的 data/ 文件夹）
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
# 模型权重保存目录
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST Conv+BN+Conv+BN 网络训练")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=256, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--weight-decay", type=float,
                        default=1e-4, help="L2 正则化系数")
    parser.add_argument("--val-split", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--workers", type=int, default=2,
                        help="DataLoader 工作进程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def build_dataloaders(
    data_dir: str,
    batch_size: int,
    val_split: float,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    print(f"[数据] 数据目录: {os.path.abspath(data_dir)}")
    full_train = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val

    train_dataset, val_dataset = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    val_dataset.dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=False,
        transform=transform,
    )

    print(f"[数据] 训练: {n_train} | 验证: {n_val} | 测试: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

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
    model.eval()
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

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用: {device}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"[保存] 权重目录: {os.path.abspath(CKPT_DIR)}")

    train_loader, val_loader, test_loader = build_dataloaders(
        DATA_DIR,
        args.batch_size,
        args.val_split,
        args.workers,
    )

    model = ConvBNConvBN(num_classes=10).to(device)
    print(
        f"[模型] 参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(CKPT_DIR, "best_model.pth")

    print("\n" + "=" * 60)
    print(
        f"  开始训练: {args.epochs} epochs | batch={args.batch_size} | lr={args.lr}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}%  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%  "
            f"lr={current_lr:.6f}"
        )

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  ★ 保存最优模型 (val_acc={val_acc:.2f}%) -> {best_ckpt_path}")

    print("\n" + "=" * 60)
    print(f"  训练完成！最优验证集准确率: {best_val_acc:.2f}%")

    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"  测试集准确率: {test_acc:.2f}%  (loss={test_loss:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
