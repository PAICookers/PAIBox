"""
MNIST LeNet-5 网络 (LeNet-5 Network)
输入: 1x28x28
结构: 
      Conv2d(1->6) -> ReLU -> MaxPool2d
      Conv2d(6->16) -> ReLU -> MaxPool2d
      FC(16*5*5->120) -> ReLU
      FC(120->84) -> ReLU
      FC(84->10)
"""

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # 特征提取部分
        self.features = nn.Sequential(
            # 输入图像尺寸: 1 x 28 x 28
            # Conv1: padding=2，输出尺寸: 6 x 28 x 28
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # Pool1: 输出尺寸: 6 x 14 x 14
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv2: 输出尺寸: 16 x 10 x 10
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            # Pool2: 输出尺寸: 16 x 5 x 5
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 分类器部分 (FC 层)
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """使用 He 初始化卷积层和全连接层权重，偏置置零。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)         # (N, 16, 5, 5)
        x = torch.flatten(x, 1)      # (N, 16*5*5)
        x = self.classifier(x)       # (N, 10)
        return x
