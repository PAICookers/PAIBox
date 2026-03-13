import pytest
import torch
from paicorelib import DataSign, DataWidth
from spikingjelly.activation_based import neuron as sj
from torch import Tensor, nn

from paibox.backendv2.mapper import Mapper
from paibox.paiir import CompileConfig, PAIIRGraph, compile_to_paiir
from paibox.paiir.exceptions import UnsupportedOpError, UnsupportedOpWarning
from paibox.paiir.op_node import AccumulateOp, ConcatOp, SequentialOp


def make_img_3ch_32x32() -> Tensor:
    """3-channel 32x32 image, standard larger input."""
    return torch.randn(1, 3, 10, 10)


class SimpleCNN(nn.Module):
    """Conv-ReLU-Conv: minimal ANN without classifier head."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.conv2(x)


graph = compile_to_paiir(SimpleCNN(), make_img_3ch_32x32())

print(graph.summary())

mapper = Mapper()
mapper.compile(graph)
