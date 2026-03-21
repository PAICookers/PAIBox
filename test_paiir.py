import torch
from torch import Tensor, nn

from paibox.backendv2.mapper import Mapper
from paibox.paiir import compile_to_paiir


def make_img_3ch_32x32() -> Tensor:
    """3-channel 32x32 image, standard larger input."""
    return torch.randn(1, 3, 10, 10)


class SimpleCNN(nn.Module):
    """Conv-ReLU-Conv: minimal ANN without classifier head."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv1.weight.data = torch.arange(1, 10, dtype=torch.float32).reshape(
            1, 1, 3, 3
        )
        self.conv2.weight.data = torch.arange(10, 19, dtype=torch.float32).reshape(
            1, 1, 3, 3
        )
        print("conv1 weight:", self.conv1.weight)
        print("conv2 weight:", self.conv2.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.conv2(x)


graph = compile_to_paiir(SimpleCNN(), make_img_3ch_32x32())

print(graph.summary())

mapper = Mapper()
mapper.compile(graph)
