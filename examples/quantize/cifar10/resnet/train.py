if __name__ == "__main__":

    import os

    # 忽略 OpenMP 重复初始化错误，必须在导入 torch 之前设置才有效
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    from model import ResNetCIFAR10
    import argparse
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import torch.optim as optim
    import torch.nn as nn
    import torch

    # 全局路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", ".."))
    DATA_DIR = os.path.join(REPO_ROOT, "examples", "data")
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

    def get_args():
        parser = argparse.ArgumentParser(description="CIFAR-10 ResNet 训练")
        parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
        parser.add_argument("--batch-size", type=int, default=128, help="批大小")
        parser.add_argument("--lr", type=float, default=0.0005, help="初始学习率")
        return parser.parse_args()

    def train():
        args = get_args()

        # 创建权重保存目录
        os.makedirs(CKPT_DIR, exist_ok=True)
        print(f"[保存] 权重目录: {os.path.abspath(CKPT_DIR)}")

        # 检测硬件并设置device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[设备] 使用: {device}")

        # 建立模型
        model = ResNetCIFAR10(num_classes=10).to(device)

        # 数据增强与加载
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

        train_dataset = datasets.CIFAR10(
            root=DATA_DIR, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(
            root=DATA_DIR, train=False, download=True, transform=transform_test)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_val_acc = 0.0
        best_ckpt_path = os.path.join(CKPT_DIR, "best_model.pth")

        if os.path.exists(best_ckpt_path):
            try:
                ckpt = torch.load(best_ckpt_path, map_location=device)
                if "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                    best_val_acc = ckpt.get("val_acc", 0.0)
                    print(
                        f"[*] 已加载模型权重: {best_ckpt_path}, 之前最佳准确率为 {best_val_acc:.2f}%")
                else:  # 兼容只有 state_dict 的旧情况
                    model.load_state_dict(ckpt)
                    print(f"[*] 已加载模型权重 (state_dict): {best_ckpt_path}")
            except Exception as e:
                print(f"[*] 模型权重加载失败: {e}，将重新开始训练。")

        print("\n" + "=" * 65)
        print(
            f"  开始训练: {args.epochs} epochs | batch={args.batch_size} | lr={args.lr}")
        print("=" * 65)

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                if batch_idx % 100 == 0:
                    print(
                        f"Epoch [{epoch}/{args.epochs}] | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")

            train_acc = 100. * correct / total

            # Test phase
            model.eval()
            test_loss = 0
            test_correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    _, predicted = output.max(1)
                    test_correct += predicted.eq(target).sum().item()

            test_loss /= len(test_loader)
            val_acc = 100. * test_correct / len(test_loader.dataset)

            print(
                f"--> Epoch {epoch} 训练平均 Loss: {total_loss/len(train_loader):.4f} | 训练集 Acc: {train_acc:.2f}% | 测试集 Loss: {test_loss:.4f} | 测试集 Acc: {val_acc:.2f}%")

            # # 保存当前 epoch 状态
            # torch.save(
            #     {
            #         "epoch": epoch,
            #         "model_state_dict": model.state_dict(),
            #         "optimizer_state_dict": optimizer.state_dict(),
            #         "val_acc": val_acc,
            #     },
            #     os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pth"),
            # )

            # 保存表现最好的模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_ckpt_path)
                print(
                    f"  ★ 保存最优模型 (val_acc={val_acc:.2f}%) -> {best_ckpt_path}")

            print("-" * 65)

        print("\n" + "=" * 65)
        print(f"  训练完成！最优测试集准确率: {best_val_acc:.2f}%")
        print("=" * 65)

    train()
