from functools import cached_property
from typing import ClassVar

import numpy as np
from paicorelib import WeightWidth as WW

from paibox.base import NeuDyn, SynSys
from paibox.exceptions import RegisterError, ShapeError
from paibox.types import VOLTAGE_DTYPE, DataType, NeuOutType, SynOutType, WeightType

from ..modules import BuildingModule
from ..neuron import Neuron
from ..neuron.utils import get_delay_reg_len
from ..projection import InputProj
from .conv_types import Size1Type, Size2Type, _KOrder3d, _KOrder4d
from .conv_utils import (
    _conv1d_oshape,
    _conv2d_oshape,
    fm_ndim1_check,
    fm_ndim2_check,
    group_ch_check,
)
from .transforms import (
    AllToAll,
    CompareMax,
    ConnType,
    Conv1dForward,
    Conv2dForward,
    Conv2dSemiFoldedForward,
    ConvTranspose1dForward,
    ConvTranspose2dForward,
    Identity,
    MaskedLinear,
    OneToOne,
    Transform,
)

RIGISTER_MASTER_KEY_FORMAT = "{0}.output"


def _check_equal(num_in: int, num_out: int) -> int:
    if num_in != num_out:
        raise ShapeError(
            f"the number of source & destination neurons must be equal: {num_in} != {num_out}."
        )

    return num_in


class FullConnectedSyn(SynSys):
    comm: Transform
    _n_copied: int = 0
    """Counter of copies."""

    synin: np.ndarray
    """The input of the synapse at every timestep."""

    def __init__(
        self, source: NeuDyn | InputProj, target: NeuDyn, name: str | None = None
    ) -> None:
        super().__init__(name)
        self._source = source
        self._target = target

        self.set_memory("_synout", np.zeros((self.num_out,), dtype=VOLTAGE_DTYPE))

        # Register itself with the master nodes of target.
        target.register_master(RIGISTER_MASTER_KEY_FORMAT.format(self.name), self)

        # If the source is `BuildingModule`, register itself with its module interface.
        if isinstance(source, BuildingModule):
            source.register_output(self)

    def __call__(self, *args, **kwargs) -> SynOutType:
        return self.update(*args, **kwargs)

    def update(self, x: NeuOutType | None = None, *args, **kwargs) -> SynOutType:
        # Retrieve the output at [timestamp] of the target neurons
        if self.target.is_working():
            if isinstance(self.source, InputProj):
                synin = self.source.output if x is None else np.atleast_1d(x)
            else:
                idx = self.target.timestamp % get_delay_reg_len(self.source)
                synin = (
                    self.source.delay_registers[idx] if x is None else np.atleast_1d(x)
                )
        else:
            # Retrieve 0 to the target neurons if it is not working
            if isinstance(self.source, InputProj):
                synin = np.zeros_like(
                    self.source.output if x is None else np.atleast_1d(x)
                )
            else:
                synin = np.zeros_like(
                    self.source.delay_registers[0] if x is None else np.atleast_1d(x)
                )

        self.synin = synin
        self._synout = self.comm(synin).ravel()

        return self._synout

    def reset_state(self, *args, **kwargs) -> None:
        self.reset_memory()  # Call reset of `StatusMemory`.

    def __copy__(self) -> "FullConnSyn":
        return self.__deepcopy__()

    def __deepcopy__(self, memo=None) -> "FullConnSyn":
        self._n_copied += 1

        return FullConnSyn(
            self.source,
            self.target,
            self.connectivity,
            ConnType.All2All,
            f"{self.name}_copied_{self._n_copied}",
        )

    def copy(
        self, source: NeuDyn | InputProj | None = None, target: NeuDyn | None = None
    ) -> "FullConnSyn":
        copied = self.__copy__()
        if isinstance(source, (NeuDyn, InputProj)):
            copied.source = source

        if isinstance(target, NeuDyn):
            copied.target = target

        return copied

    @property
    def source(self) -> NeuDyn | InputProj:
        return self._source

    @source.setter
    def source(self, source: NeuDyn | InputProj) -> None:
        """Set a new source neuron."""
        if source.num_out != self.num_in:
            raise RegisterError(
                f"the number of source neurons before and after the change "
                f"is not equal, {source.num_out} != {self.num_in}."
            )

        self._source = source

    @property
    def target(self) -> NeuDyn:
        return self._target

    @target.setter
    def target(self, target: NeuDyn) -> None:
        """Set a new target neuron."""
        if target.num_in != self.num_out:
            raise RegisterError(
                f"the number of source neurons before and after the change "
                f"is not equal, {target.num_in} != {self.num_out}."
            )

        self._target.unregister_master(self.name)

        self._target = target
        # Allow the same target to register again.
        target.register_master(
            RIGISTER_MASTER_KEY_FORMAT.format(self.name), self, strict=False
        )

    @property
    def shape_in(self) -> tuple[int, ...]:
        return self._source.shape_out

    @property
    def shape_out(self) -> tuple[int, ...]:
        return self._target.shape_in

    @property
    def num_in(self) -> int:
        return self._source.num_out

    @property
    def num_out(self) -> int:
        return self._target.num_in

    @property
    def output(self) -> SynOutType:
        return self._synout

    @property
    def weights(self) -> WeightType:
        return self.comm.weights

    @property
    def weight_width(self) -> WW:
        return self.comm._get_weight_width(self.CFLAG_ENABLE_WP_OPTIMIZATION)

    @cached_property
    def connectivity(self) -> WeightType:
        """The connectivity matrix in `bool` or `np.int8` format."""
        return self.comm.connectivity


class FullConnSyn(FullConnectedSyn):
    def __init__(
        self,
        source: NeuDyn | InputProj,
        target: NeuDyn,
        weights: DataType,
        conn_type: ConnType,
        name: str | None = None,
    ) -> None:
        super().__init__(source, target, name)

        if conn_type is ConnType.One2One:
            comm = OneToOne(_check_equal(self.num_in, self.num_out), weights)
        elif conn_type is ConnType.Identity:
            if not isinstance(weights, (int, bool, np.integer)):
                raise TypeError(
                    f"expected type int, bool, np.integer, but got type {type(weights).__name__}."
                )
            comm = Identity(_check_equal(self.num_in, self.num_out), weights)
        elif conn_type is ConnType.All2All:
            comm = AllToAll((self.num_in, self.num_out), weights)
        else:  # MatConn
            if not isinstance(weights, np.ndarray):
                raise TypeError(
                    f"expected type np.ndarray, but got type {type(weights).__name__}."
                )
            if len(self.shape_in) > 2:
                raise ShapeError(
                    f"Expect the shape of source to have no more than 2 dimensions, "
                    f"but got {len(self.shape_in)}."
                )

            comm = MaskedLinear(self.shape_in, self.shape_out, weights)

        self.comm = comm


class Conv1dSyn(FullConnectedSyn):
    comm: Conv1dForward
    _spatial_ndim: ClassVar[int] = 1

    def __init__(
        self,
        source: NeuDyn | InputProj,
        target: Neuron,
        kernel: np.ndarray,
        stride: Size1Type,
        padding: Size1Type,
        dilation: Size1Type,
        groups: int,
        order: _KOrder3d,
        name: str | None = None,
    ) -> None:
        super().__init__(source, target, name)

        if kernel.ndim != self._spatial_ndim + 2:
            raise ShapeError(
                f"convolution kernel dimension must be {self._spatial_ndim + 2}, but got {kernel.ndim}."
            )

        if order == "IOL":
            _kernel = np.swapaxes(kernel, 0, 1)
        else:
            _kernel = kernel

        co, ci_in_grp, k = _kernel.shape
        ci, li = fm_ndim1_check(source.shape_out, "CL")
        (lo,) = _conv1d_oshape((li,), (k,), stride, padding, dilation)

        group_ch_check(ci, co, groups, ci_in_grp)

        if (_output_size := co * lo) != target.num_in:
            raise ShapeError(
                f"output size mismatch: {_output_size} != {target.num_in}."
            )

        self.comm = Conv1dForward((li,), (lo,), _kernel, stride, padding, groups=groups)


class Conv2dSyn(FullConnectedSyn):
    comm: Conv2dForward
    _spatial_ndim: ClassVar[int] = 2

    def __init__(
        self,
        source: NeuDyn | InputProj,
        target: Neuron,
        kernel: np.ndarray,
        stride: Size2Type,
        padding: Size2Type,
        dilation: Size2Type,
        groups: int,
        order: _KOrder4d,
        name: str | None = None,
    ) -> None:
        super().__init__(source, target, name)

        if kernel.ndim != self._spatial_ndim + 2:
            raise ShapeError(
                f"convolution kernel dimension must be {self._spatial_ndim + 2}, but got {kernel.ndim}."
            )

        if order == "IOHW":
            _kernel = np.swapaxes(kernel, 0, 1)
        else:
            _kernel = kernel

        co, ci_in_grp, kh, kw = _kernel.shape
        ci, hi, wi = fm_ndim2_check(source.shape_out, "CHW")
        ho, wo = _conv2d_oshape((hi, wi), (kh, kw), stride, padding, dilation)

        group_ch_check(ci, co, groups, ci_in_grp)

        if (_output_size := co * ho * wo) != target.num_in:
            raise ShapeError(
                f"output size mismatch: {_output_size} ({co}*{ho}*{wo}) "
                f"!= {target.num_in}."
            )

        self.comm = Conv2dForward(
            (hi, wi), (ho, wo), _kernel, stride, padding, groups=groups
        )


class Conv2dSemiFoldedSyn(FullConnectedSyn):
    comm: Conv2dSemiFoldedForward
    _spatial_ndim: ClassVar[int] = 1

    def __init__(
        self,
        source: NeuDyn | InputProj,
        target: Neuron,
        kernel: np.ndarray,
        stride: Size2Type,
        padding: Size2Type,
        groups: int,
        order: _KOrder3d,
        name: str | None = None,
    ) -> None:
        super().__init__(source, target, name)

        if kernel.ndim != self._spatial_ndim + 2:
            raise ShapeError(
                f"convolution kernel dimension must be {self._spatial_ndim + 2}, but got {kernel.ndim}."
            )

        if order == "IOL":
            _kernel = np.swapaxes(kernel, 0, 1)
        else:
            _kernel = kernel.copy()

        # O,I,H
        co, ci_in_grp, kh = _kernel.shape
        # I,H
        assert len(source.shape_out) == 2
        ci, hi = source.shape_out
        ho = (hi + 2 * padding[0] - kh) // stride[0] + 1

        if ci != (_cur_in_ch := groups * ci_in_grp):
            in_ch_mismatch_text = f"input channels mismatch: {ci} != {_cur_in_ch}"
            in_ch_mismatch_text += f" ({groups}*{ci_in_grp})." if groups > 1 else "."
            raise ShapeError(in_ch_mismatch_text)

        if (_output_size := co * ho) != target.num_in:
            raise ShapeError(
                f"output size mismatch: {_output_size} ({co}*{ho}) != {target.num_in}."
            )

        self.comm = Conv2dSemiFoldedForward(
            (ci, hi), (co, ho), _kernel, stride, padding, groups=groups
        )


class ConvTranspose1dSyn(FullConnectedSyn):
    _spatial_ndim: ClassVar[int] = 1

    def __init__(
        self,
        source: NeuDyn | InputProj,
        target: Neuron,
        kernel: np.ndarray,
        stride: tuple[int],
        padding: tuple[int],
        dilation: tuple[int],
        output_padding: tuple[int],
        order: _KOrder3d,
        name: str | None = None,
    ) -> None:
        super().__init__(source, target, name)

        if kernel.ndim != self._spatial_ndim + 2:
            raise ShapeError(
                f"convolution kernel dimension must be {self._spatial_ndim + 2}, but got {kernel.ndim}."
            )

        if order == "IOL":
            _kernel = np.swapaxes(kernel, 0, 1)
        else:
            _kernel = kernel.copy()

        # O,I,L
        co, in_channels, k = _kernel.shape
        # C,L
        ci, li = fm_ndim1_check(source.shape_out, "CL")
        lo = (
            (li - 1) * stride[0]
            - 2 * padding[0]
            + dilation[0] * (k - 1)
            + output_padding[0]
            + 1
        )

        if ci != in_channels:
            raise ShapeError(f"input channels mismatch: {ci} != {in_channels}.")

        if (_output_size := co * lo) != target.num_in:
            raise ShapeError(
                f"output size mismatch: {_output_size} != {target.num_in}."
            )

        self.comm = ConvTranspose1dForward(
            (li,), (lo,), _kernel, stride, padding, output_padding=output_padding
        )


class ConvTranspose2dSyn(FullConnectedSyn):
    _spatial_ndim: ClassVar[int] = 2

    def __init__(
        self,
        source: NeuDyn | InputProj,
        target: Neuron,
        kernel: np.ndarray,
        stride: Size2Type,
        padding: Size2Type,
        dilation: Size2Type,
        output_padding: Size2Type,
        order: _KOrder4d,
        name: str | None = None,
    ) -> None:
        super().__init__(source, target, name)

        if kernel.ndim != self._spatial_ndim + 2:
            raise ShapeError(
                f"convolution kernel dimension must be {self._spatial_ndim + 2}, but got {kernel.ndim}."
            )

        if order == "IOHW":
            _kernel = np.swapaxes(kernel, 0, 1)
        else:
            _kernel = kernel.copy()

        # O,I,H,W
        co, in_channels, kh, kw = _kernel.shape
        # C,H,W
        ci, hi, wi = fm_ndim2_check(source.shape_out, "CHW")
        ho = (
            (hi - 1) * stride[0]
            - 2 * padding[0]
            + dilation[0] * (kh - 1)
            + output_padding[0]
            + 1
        )
        wo = (
            (wi - 1) * stride[1]
            - 2 * padding[1]
            + dilation[1] * (kw - 1)
            + output_padding[1]
            + 1
        )

        if ci != in_channels:
            raise ShapeError(f"input channels mismatch: {ci} != {in_channels}.")

        if (_output_size := co * ho * wo) != target.num_in:
            raise ShapeError(
                f"output size mismatch: {_output_size} != {target.num_in}."
            )

        self.comm = ConvTranspose2dForward(
            (hi, wi),
            (ho, wo),
            _kernel,
            stride,
            padding,
            output_padding=output_padding,
        )


class MaxPoolSyn(FullConnectedSyn):
    """Max pooling synapses. Only used when input width is 8-bit."""

    comm: CompareMax

    def __init__(
        self,
        source: NeuDyn | InputProj,
        target: Neuron,
        weights: DataType = 1,
        name: str | None = None,
    ) -> None:
        super().__init__(source, target, name)
        self.comm = CompareMax((self.num_in, self.num_out), weights)
