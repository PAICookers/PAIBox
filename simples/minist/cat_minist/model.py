"""
MNIST Cat 网络

输入: 1x28x28
结构:
    Stem: Conv2d(1->8) -> ReLU -> MaxPool2d
    Branch A: Conv2d(8->8) -> ReLU
    Branch B: Conv2d(8->8) -> ReLU
    Cat: channel 维拼接
    Head: MaxPool2d -> Flatten -> FC(16*7*7->128) -> ReLU -> FC(128->10)
"""

import torch
import torch.nn as nn


class MNISTCatNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.branch_a = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.branch_b = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        branch_a = self.branch_a(x)
        branch_b = self.branch_b(x)
        x = torch.cat([branch_a, branch_b], dim=1)
        x = self.head(x)
        return x