"""
CIFAR-10 全连接网络 (FC Network)
输入: 32x32x3 = 3072 维展平向量
结构: FC(3072->2048) -> BN -> ReLU
      FC(2048->1024) -> BN -> ReLU
      FC(1024->256)  -> BN -> ReLU
      FC(256->10)
"""

import torch
import torch.nn as nn


class CIFAR10_FC(nn.Module):

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.5) -> None:
        """
        Args:
            num_classes: 分类数量，CIFAR-10 默认为 10
            dropout_p:   Dropout 概率，用于缓解过拟合
        """
        super().__init__()

        # CIFAR-10 图像尺寸: 3 x 32 x 32，展平后维度为 3072
        input_dim = 3 * 32 * 32  # 3072

        # 第一全连接层: 3072 -> 2048（加宽首层，提升特征提取能力）
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )

        # 第二全连接层: 2048 -> 1024
        self.layer2 = nn.Sequential(
            nn.Linear(2048, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )

        # 第三全连接层: 1024 -> 256
        self.layer3 = nn.Sequential(
            nn.Linear(1024, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )

        # 输出层: 256 -> 10（不加激活函数，配合 CrossEntropyLoss 使用）
        self.classifier = nn.Linear(256, num_classes, bias=True)

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """使用 He 初始化线性层权重，偏置置零。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 将图像展平为一维向量: (N, 3, 32, 32) -> (N, 3072)
        x = x.view(x.size(0), -1)

        x = self.layer1(x)      # (N, 3072) -> (N, 2048)
        x = self.layer2(x)      # (N, 2048) -> (N, 1024)
        x = self.layer3(x)      # (N, 1024) -> (N, 256)
        x = self.classifier(x)  # (N, 256)  -> (N, 10)
        return x
