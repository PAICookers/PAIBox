import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    残差块 (Residual Block)：
    这是深度残差网络（ResNet）的基本构建单元。它包含两条路径：
    1. 主路径（经过两层 3x3 卷积）
    2. 捷径/直连路径 (Shortcut)：直接将输入加到输出上，防止在训练极深的网络时发生梯度消失。
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 第一层卷积：可能包含下采样（如果 stride=2，会导致图片尺寸减半）
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批量归一化，加速训练并提升稳定性

        # 第二层卷积：保持特征图尺寸不变 (stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 捷径（Shortcut）路径的设计
        # 如果特征图大小减半 (stride != 1) 或者通道数量有变化，
        # 我们需要一个额外的 1x1 卷积去修改输入的形状，这样主路径和捷径才能对齐相加。
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True)
            )

    def forward(self, x):
        # 主路径：前向传播
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 将捷径路径的结果与主路径的结果相加 (这是残差网络的核心操作)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR10(nn.Module):
    """
    针对 CIFAR-10 数据集的残差网络优化版本
    包含最多两层 ResNet 结构，参数量维持在 ~2万 左右
    """

    def __init__(self, num_classes=10):
        super(ResNetCIFAR10, self).__init__()

        # CIFAR-10 图片尺寸为 3通道 32x32
        # 1. 初始卷积层，使用 stride=1 保持最佳的 32x32 初始分辨率
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # 2. 只有两层残差块阶段 (2 Res Blocks)
        # 阶段一：16通道，特征图保持 32x32
        self.layer1 = ResidualBlock(16, 16, stride=1)
        # 阶段二：通道翻倍到 32，特征图降维到 16x16
        self.layer2 = ResidualBlock(16, 32, stride=2)

        # 3. 普通最大池化与全连接层
        # 经过 layer2 后特征图为 16x16，使用 4x4 的池化可以将其降维到 4x4
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.linear = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))    # -> [batch_size, 16, 32, 32]

        # 依次通过两层残差结构
        out = self.layer1(out)                   # -> [batch_size, 16, 32, 32]
        out = self.layer2(out)                   # -> [batch_size, 32, 16, 16]

        # 普通最大池化及全连接
        out = self.maxpool(out)                  # -> [batch_size, 32, 4, 4]
        out = torch.flatten(out, 1)              # -> [batch_size, 32 * 16]
        out = self.linear(out)                   # -> [batch_size, 10]
        return out
