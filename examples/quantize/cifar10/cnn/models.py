"""
LeNet-5 (适配 CIFAR-10 版本)

原始 LeNet-5 针对 32x32 灰度图设计，这里做两处适配：
  1. 输入通道改为 3（RGB）
  2. 激活函数使用 ReLU（原版为 Sigmoid/Tanh），训练更稳定

网络结构:
  输入: (N, 3, 32, 32)
  C1:  Conv(3,  6, 5x5) -> ReLU -> MaxPool(2x2)   -> (N, 6,  14, 14)
  C3:  Conv(6, 16, 5x5) -> ReLU -> MaxPool(2x2)   -> (N, 16,  5,  5)
  展平:                                              -> (N, 400)
  F4:  Linear(400, 120) -> ReLU
  F5:  Linear(120,  84) -> ReLU
  F6:  Linear( 84,  10)
"""

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    适配 CIFAR-10 的 LeNet-5 模型。

    Args:
        num_classes: 分类数量，默认 10
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # ── 特征提取部分（卷积层） ──────────────────────────
        self.features = nn.Sequential(
            # C1: 卷积层，3 通道 -> 6 通道，5x5 卷积核，输出 28x28
            nn.Conv2d(in_channels=3, out_channels=6,
                      kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),          # BN 稳定训练（原版无此层）
            nn.ReLU(inplace=True),
            # S2: 最大池化，2x2，步长 2，输出 14x14
            nn.MaxPool2d(kernel_size=2, stride=2),

            # C3: 卷积层，6 通道 -> 16 通道，5x5 卷积核，输出 10x10
            nn.Conv2d(in_channels=6, out_channels=16,
                      kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # S4: 最大池化，2x2，步长 2，输出 5x5
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── 分类部分（全连接层） ──────────────────────────
        # 经过两次卷积+池化后尺寸: 16 x 5 x 5 = 400
        self.classifier = nn.Sequential(
            # F4
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            # F5
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            # F6（输出层，不加激活，配合 CrossEntropyLoss）
            nn.Linear(84, num_classes),
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """卷积层用 He 初始化，全连接层用 Xavier 初始化。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状 (N, 3, 32, 32)
        Returns:
            logits，形状 (N, num_classes)
        """
        x = self.features(x)            # 卷积特征提取
        x = torch.flatten(x, 1)          # 展平为 (N, 400)
        x = self.classifier(x)          # 全连接分类
        return x
