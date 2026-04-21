"""
MNIST 全连接网络 (FC Network)
输入: 1x28x28 = 784 维展平向量
结构: FC(784->512) -> BN -> ReLU
      FC(512->256) -> BN -> ReLU
      FC(256->10)
"""

import torch
import torch.nn as nn


class MNIST_FC(nn.Module):
    def __init__(self, num_classes: int = 10, dropout_p: float = 0.5) -> None:
        super().__init__()

        # MNIST 图像尺寸: 1 x 28 x 28，展平后维度为 784
        input_dim = 1 * 28 * 28  # 784

        # 第一全连接层: 784 -> 512
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        # 第二全连接层: 512 -> 256
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # 输出层 (第三全连接层): 256 -> 10
        self.classifier = nn.Linear(256, num_classes, bias=True)

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
        # 将图像展平为一维向量: (N, 1, 28, 28) -> (N, 784)
        x = x.view(x.size(0), -1)
        x = self.layer1(x)      # (N, 784) -> (N, 512)
        x = self.layer2(x)      # (N, 512) -> (N, 256)
        x = self.classifier(x)  # (N, 256) -> (N, 10)
        return x
