import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    ResNet18 的基本块 (BasicBlock)
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 快捷连接
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18CIFAR10(nn.Module):
    """
    针对 CIFAR-10 数据集设计的 ResNet-18
    因为 CIFAR-10 图片尺寸只有 32x32，去掉了开头的 7x7 conv 和 MaxPool，
    改用 3x3 conv，且第一层步长保持 1，这能保留更多空间信息。
    """

    def __init__(self, num_classes=10):
        super(ResNet18CIFAR10, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 2, stride=1)   # -> 32x32
        self.layer2 = self._make_layer(128, 2, stride=2)  # -> 16x16
        self.layer3 = self._make_layer(256, 2, stride=2)  # -> 8x8
        self.layer4 = self._make_layer(512, 2, stride=2)  # -> 4x4

        self.maxpool = nn.MaxPool2d(4)  # 全局最大池化，输出 512 * expansion
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.maxpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out
