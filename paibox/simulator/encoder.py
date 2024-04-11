import math
from typing import Literal, Optional

import numpy as np

from paibox.mixin import StatusMemory
from paibox.types import SpikeType, WeightType
from paibox.synapses.conv_utils import _conv2d_faster, Size2Type
__all__ = ["LatencyEncoder", "PeriodicEncoder", "PoissonEncoder", "DirectEncoder"]

MAXSEED = np.iinfo(np.uint32).max
MAXINT = np.iinfo(np.int32).max


class Encoder:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = self._get_rng(seed)

    def _get_rng(self, seed: Optional[int] = None) -> np.random.RandomState:
        _seed = np.random.randint(MAXINT) if seed is None else seed
        return np.random.RandomState(_seed)

    def __call__(self, x: np.ndarray, *args, **kwargs) -> SpikeType:
        raise NotImplementedError


class StatelessEncoder(Encoder):
    pass


class StatefulEncoder(Encoder, StatusMemory):
    def __init__(self, T: int, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        super(Encoder, self).__init__()

        if T < 1:
            raise ValueError(f"'T' must be positive, but got {T}.")

        self.T = T
        self.set_memory("spike", None)
        self.set_memory("t", 0)

    def __call__(self, x: Optional[np.ndarray] = None, *args, **kwargs) -> SpikeType:
        # If there is no encoded spike but there is an input, encode the input
        if self.spike is None:
            if x is None:
                raise ValueError("input must be given if 'spike' is None.")

            self.encode(x)

        t = self.t
        self.t += 1

        if self.t >= self.T:
            self.t = 0

        return self.spike[t]

    def encode(self, x: np.ndarray) -> None:
        """Encoding function. Called only if there is no encoded spike."""
        raise NotImplementedError


class PeriodicEncoder(StatefulEncoder):
    def __init__(self, spike: np.ndarray, **kwargs) -> None:
        """Periodic encoder.

        Args:
            - spike: the spike to be encoded. Encode at instantiation, where `T=shape[0]` & `shape_out=shape[1]`.
        """
        super().__init__(spike.shape[0], **kwargs)
        self.spike = spike

    def encode(self, x: np.ndarray) -> None:
        self.spike = x
        self.T = x.shape[0]


class LatencyEncoder(StatefulEncoder):
    def __init__(self, T: int, encoding_func: Literal["linear", "log"]) -> None:
        """Latency encoder.

        Args:
            - T: encoding timestep.
            - encoding_func: encoding function. It can be 'log' or 'linear'.

        NOTE: See details at https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based/2_encoding.html#id5
        """
        super().__init__(T)

        if encoding_func == "log":
            self.alpha = math.exp(T - 1) - 1
        elif encoding_func != "linear":  # `alpha` is not used in method 'linear'.
            raise ValueError("encoding function must be 'linear' or 'log'.")

        self.enc_func = encoding_func

    def encode(self, x: np.ndarray) -> None:
        if self.enc_func == "log":
            t_f = (self.T - 1 - np.log(self.alpha * x + 1)).round().astype(np.int64)
        else:
            t_f = ((self.T - 1.0) * (1.0 - x)).round().astype(np.int64)

        indices = t_f.ravel()
        spike = np.eye(self.T, dtype=np.bool_)[indices]
        # [*, T] -> [T, *]
        self.spike = np.moveaxis(spike, -1, 0)


class PoissonEncoder(StatelessEncoder):
    def __init__(self, seed: Optional[int] = None, **kwargs) -> None:
        """Poisson encoder.

        NOTE: The output shape of the poisson encoder depends on the input shape.
        """
        super().__init__(seed, **kwargs)

    def __call__(self, x: np.ndarray, *args, **kwargs) -> SpikeType:
        return np.less_equal(self.rng.random(x.shape), x).astype(np.bool_)


class DirectEncoder(StatelessEncoder):
    def __init__(self, ksize: WeightType, stride: Size2Type, padding: Size2Type, leak_mem=0.95) -> None:
        super().__init__()
        self.shape = []
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.leak_mem = leak_mem

    def __call__(self, x: np.ndarray, *args, **kwargs):
        xc, xh, xw = x.shape
        cout, cin, kh, kw = self.ksize.shape
        shape = ((xh + self.padding[0] * 2 - kh) // self.stride[0] + 1, (xw + self.padding[1] * 2 - kw) // self.stride[1] + 1)
        static_input = _conv2d_faster(x, out_shape=shape, kernel=self.ksize, stride=self.stride, padding=self.padding, dtype=np.float32)
        mem_conv1 = np.zeros_like(static_input)
        mem_conv1 = (1 - self.leak_mem) * mem_conv1 + self.leak_mem * static_input
        mem_thr = (mem_conv1 / 1.0) - 1.0
        out = np.where(mem_thr > 0, 1, 0).astype(np.bool_)
        return out
