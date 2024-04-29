import math
from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import NDArray

from paibox.components.synapses.conv_types import _KOrder4d, _Size2Type
from paibox.components.synapses.conv_utils import _pair
from paibox.mixin import StatusMemory
from paibox.types import SpikeType

from .utils import _conv2d_faster_fp32

__all__ = ["LatencyEncoder", "PeriodicEncoder", "PoissonEncoder", "Conv2dEncoder"]

"""
    We provide a few simple encoders for you to implement basic coding functions
    without relying on other libraries, such as SpikingJelly. If you need use
    more complex encoders, use them directly.
"""

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
    def __init__(
        self,
        tau: float,
        decay_input: bool,
        v_threshold: float,
        v_reset: float,
    ) -> None:
        super().__init__()

        if tau < 1:
            raise ValueError(f"'tau' must be great or equal to 1, but got {tau}.")

        self.decay_input = decay_input
        self.tau = tau
        self.v_reset = v_reset
        self.v_threshold = v_threshold

        self.v = np.array(v_reset)

    def _lif_activate(self, encoded: NDArray[np.float32]) -> SpikeType:
        self.neuronal_charge(encoded)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

        return spike

    def neuronal_charge(self, x: NDArray[np.float32]) -> None:
        if self.decay_input:
            self.v = self.v + (self.v_reset - self.v + x) / self.tau
        else:
            self.v = self.v + (self.v_reset - self.v) / self.tau + x

    def neuronal_fire(self) -> SpikeType:
        return (self.v - self.v_threshold) > 0

    def neuronal_reset(self, spike: SpikeType) -> None:
        if self.v_reset == 0:
            # soft reset
            self.v = self.v - self.v_threshold * spike
        else:
            # hard reset
            self.v = spike * self.v_reset + (1.0 - spike) * self.v


class Conv2dEncoder(DirectEncoder):
    def __init__(
        self,
        kernel: NDArray[Any],
        stride: _Size2Type = 1,
        padding: _Size2Type = 0,
        kernel_order: _KOrder4d = "OIHW",
        tau: float = 1,
        decay_input: bool = True,
        v_threshold: float = 1,
        v_reset: float = 0,
    ) -> None:
        """Direct encoder with 2D convolution + LIF activation.

        Args:
            - kernel: convolution kernel. Its dimension order is either (O,I,H,W) or (I,O,H,W), depending   \
                on the argument `kernel_order`.
            - stride: the step size of the kernel sliding. It can be a scalar or a tuple of 2 integers.
            - padding: the amount of padding applied to the input. It can be a scalar or a tuple of 2 integers.
            - kernel_order: dimension order of kernel, (O,I,H,W) or (I,O,H,W). (O,I,H,W) stands for     \
                (output channels, input channels, height, width).
            - tau: membrane time constant.
            - decay_input:  whether the input will decay.
            - v_threshold: threshold voltage.
            - v_reset: reset voltage.

        NOTE: We only provide simple LIF activation. It's the same as `SimpleLIFNode` of SpikingJelly, see
            https://spikingjelly.readthedocs.io/zh-cn/latest/sub_module/spikingjelly.activation_based.neuron.html#spikingjelly.activation_based.neuron.SimpleLIFNode
            for details.

            For complex activation, please use LIFNode or other neurons in SpekingJelly. See
            https://spikingjelly.readthedocs.io/zh-cn/latest/sub_module/spikingjelly.activation_based.neuron.html#spikingjelly.activation_based.neuron.LIFNode
            for details.
        """
        if kernel_order not in ("OIHW", "IOHW"):
            raise ValueError(f"kernel order must be 'OIHW' or 'IOHW'.")

        if kernel_order == "IOHW":
            _kernel = np.swapaxes(kernel, 0, 1)
        else:
            _kernel = kernel.copy()

        super().__init__(tau, decay_input, v_threshold, v_reset)

        self.kernel = _kernel
        self.stride = _pair(stride)
        self.padding = _pair(padding)

    def __call__(self, x: np.ndarray, *args, **kwargs) -> SpikeType:
        encoded = _conv2d_faster_fp32(x, self.kernel, self.stride, self.padding)

        return self._lif_activate(encoded)
