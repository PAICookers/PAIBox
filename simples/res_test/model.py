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
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主路径：前向传播
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 将捷径路径的结果与主路径的结果相加 (这是残差网络的核心操作)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetMNIST(nn.Module):
    """
    极简版结构：为了易于理解，当前模型仅包含一个残差块。
    """

    def __init__(self, num_classes=10):
        super(ResNetMNIST, self).__init__()

        # 1. 初始卷积层，使用 stride=2 将 28x28 的输入降尺寸到 14x14
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # 2. 仅有的一个残差块，输入和输出均为 16 个特征通道，输入输出都在 14x14 这个尺幅内
        self.res_block = ResidualBlock(16, 16, stride=1)

        # 3. 全连接分类层
        # 前面 2x 降维后宽长为 14x14，再加上最后的平均池化层再除以 2 的尺寸 = 7x7
        # 平面上的所有特征点被串成一维数列，总长度就是 16 * 7 * 7 = 784
        self.linear = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        # 1. 第一层卷积提取并使用步长 2 降尺寸
        out = F.relu(self.bn1(self.conv1(x)))    # -> [batch_size, 16, 14, 14]

        # 2. 从这个唯一的残差层路过
        out = self.res_block(out)                # -> [batch_size, 16, 14, 14]

        # 3. 所有像素缩小一半，用平均池化 (Average Pooling) 把 14x14 图片浓缩为 7x7
        # -> [batch_size, 16, 7, 7]
        out = torch.max_pool2d(out, 2)

        # 4. 数据拉直以喂给全连接层处理分类信息
        out = torch.flatten(out, 1)              # -> [batch_size, 784]

        # 5. 得到全类别（比如10个数字）的预测权重
        out = self.linear(out)                   # -> [batch_size, 10]
        return out
