from typing import ClassVar, Optional, Union

import numpy as np
from paicorelib import HwConfig
from paicorelib import WeightPrecision as WP

from paibox.base import NeuDyn, SynSys
from paibox.exceptions import RegisterError, ShapeError
from paibox.types import DataArrayType, SynOutType, WeightType

from ..modules import BuildingModule
from ..neuron import Neuron
from ..projection import InputProj
from .conv_types import _KOrder3d, _KOrder4d
from .conv_utils import _fm_ndim1_check, _fm_ndim2_check
from .transforms import (
    AllToAll,
    ConnType,
    Conv1dForward,
    Conv2dForward,
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

    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        target: NeuDyn,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)

        self._source = source
        self._target = target

        self.set_memory("_synout", np.zeros((self.num_out,), dtype=np.int32))

        # Register itself with the master nodes of target.
        target.register_master(RIGISTER_MASTER_KEY_FORMAT.format(self.name), self)

        # If the source is `BuildingModule`, register itself with its module interface.
        if isinstance(source, BuildingModule):
            source.register_output(self)

    def __call__(self, *args, **kwargs) -> SynOutType:
        return self.update(*args, **kwargs)

    def update(self, spike: Optional[np.ndarray] = None, *args, **kwargs) -> SynOutType:
        # Retrieve the spike at index `timestamp` of the dest neurons
        if self.dest.is_working():
            if isinstance(self.source, InputProj):
                synin = self.source.output.copy() if spike is None else spike
            else:
                idx = self.dest.timestamp % HwConfig.N_TIMESLOT_MAX
                synin = self.source.output[idx].copy() if spike is None else spike
        else:
            # Retrieve 0 to the dest neurons if it is not working
            synin = np.zeros_like(self.source.spike)

        self._synout = self.comm(synin).ravel().astype(np.int32)
        return self._synout

    def reset_state(self, *args, **kwargs) -> None:
        # TODO Add other initialization methods in the future.
        self.reset_memory()  # Call reset of `StatusMemory`.

    def __copy__(self) -> "FullConnSyn":
        return self.__deepcopy__()

    def __deepcopy__(self, memo=None, _nil=[]) -> "FullConnSyn":
        self._n_copied += 1

        return FullConnSyn(
            self.source,
            self.dest,
            self.connectivity,
            ConnType.All2All,
            f"{self.name}_copied_{self._n_copied}",
        )

    def copy(
        self,
        source: Optional[Union[NeuDyn, InputProj]] = None,
        target: Optional[NeuDyn] = None,
    ) -> "FullConnSyn":
        copied = self.__copy__()
        if isinstance(source, (NeuDyn, InputProj)):
            copied.source = source

        if isinstance(target, NeuDyn):
            copied.target = target

        return copied

    @property
    def source(self) -> Union[NeuDyn, InputProj]:
        return self._source

    @source.setter
    def source(self, source: Union[NeuDyn, InputProj]) -> None:
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
    def dest(self) -> NeuDyn:
        # TODO To maintain compatibility, the dest attribute is preserved.
        # Will be removed in a future version.
        return self._target

    @dest.setter
    def dest(self, target: NeuDyn) -> None:
        self.target = target

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
    def weight_precision(self) -> WP:
        return self.comm._get_wp(self.CFLAG_ENABLE_WP_OPTIMIZATION)

    @property
    def connectivity(self) -> WeightType:
        """The connectivity matrix in `np.bool_` or `np.int8` format."""
        return self.comm.connectivity


class FullConnSyn(FullConnectedSyn):
    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        target: NeuDyn,
        weights: DataArrayType,
        conn_type: ConnType,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(source, target, name)

        if conn_type is ConnType.One2One:
            comm = OneToOne(_check_equal(self.num_in, self.num_out), weights)
        elif conn_type is ConnType.Identity:
            if not isinstance(weights, (int, np.bool_, np.integer)):
                raise TypeError(
                    f"expected type int, np.bool_, np.integer, but got type {type(weights)}."
                )
            comm = Identity(_check_equal(self.num_in, self.num_out), weights)
        elif conn_type is ConnType.All2All:
            comm = AllToAll((self.num_in, self.num_out), weights)
        else:  # MatConn
            if not isinstance(weights, np.ndarray):
                raise TypeError(
                    f"expected type np.ndarray, but got type {type(weights)}."
                )
            if len(self.shape_in) > 2:
                raise ShapeError(
                    f"Expect the shape of source to have no more than 2 dimensions, "
                    f"but got {len(self.shape_in)}."
                )

            comm = MaskedLinear(self.shape_in, self.shape_out, weights)

        self.comm = comm


class Conv1dSyn(FullConnectedSyn):
    _spatial_ndim: ClassVar[int] = 1

    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: Neuron,
        kernel: np.ndarray,
        stride: tuple[int],
        padding: tuple[int],
        dilation: tuple[int],
        order: _KOrder3d,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(source, dest, name)

        if kernel.ndim != self._spatial_ndim + 2:
            raise ShapeError(
                f"convolution kernel dimension must be {self._spatial_ndim + 2}, but got {kernel.ndim}."
            )

        if order == "IOL":
            _kernel = np.swapaxes(kernel, 0, 1)
        else:
            _kernel = kernel.copy()

        # O,I,L
        out_channels, in_channels, kernel_l = _kernel.shape
        # C,L
        in_ch, in_l = _fm_ndim1_check(source.shape_out, "CL")
        out_l = (in_l + 2 * padding[0] - dilation[0] * (kernel_l - 1) - 1) // stride[
            0
        ] + 1

        if in_ch != in_channels:
            raise ShapeError(f"input channels mismatch: {in_ch} != {in_channels}.")

        if (_output_size := out_channels * out_l) != dest.num_in:
            raise ShapeError(f"Output size mismatch: {_output_size} != {dest.num_in}.")

        self.comm = Conv1dForward((in_l,), (out_l,), _kernel, stride, padding)


class Conv2dSyn(FullConnectedSyn):
    _spatial_ndim: ClassVar[int] = 2

    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: Neuron,
        kernel: np.ndarray,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        order: _KOrder4d,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(source, dest, name)

        if kernel.ndim != self._spatial_ndim + 2:
            raise ShapeError(
                f"convolution kernel dimension must be {self._spatial_ndim + 2}, but got {kernel.ndim}."
            )

        if order == "IOHW":
            _kernel = np.swapaxes(kernel, 0, 1)
        else:
            _kernel = kernel.copy()

        # O,I,H,W
        out_channels, in_channels, kernel_h, kernel_w = _kernel.shape
        # C,H,W
        in_ch, in_h, in_w = _fm_ndim2_check(source.shape_out, "CHW")
        out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[
            0
        ] + 1
        out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[
            1
        ] + 1

        if in_ch != in_channels:
            raise ShapeError(f"input channels mismatch: {in_ch} != {in_channels}.")

        if (_output_size := out_channels * out_h * out_w) != dest.num_in:
            raise ShapeError(
                f"Output size mismatch: {_output_size} ({out_channels}*{out_h}*{out_w}) "
                f"!= {dest.num_in}."
            )

        self.comm = Conv2dForward(
            (in_h, in_w), (out_h, out_w), _kernel, stride, padding
        )


class ConvTranspose1dSyn(FullConnectedSyn):
    _spatial_ndim: ClassVar[int] = 1

    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: Neuron,
        kernel: np.ndarray,
        stride: tuple[int],
        padding: tuple[int],
        dilation: tuple[int],
        output_padding: tuple[int],
        order: _KOrder3d,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(source, dest, name)

        if kernel.ndim != self._spatial_ndim + 2:
            raise ShapeError(
                f"convolution kernel dimension must be {self._spatial_ndim + 2}, but got {kernel.ndim}."
            )

        if order == "IOL":
            _kernel = np.swapaxes(kernel, 0, 1)
        else:
            _kernel = kernel.copy()

        # O,I,L
        out_channels, in_channels, kernel_l = _kernel.shape
        # C,L
        in_ch, in_l = _fm_ndim1_check(source.shape_out, "CL")
        out_l = (
            (in_l - 1) * stride[0]
            - 2 * padding[0]
            + dilation[0] * (kernel_l - 1)
            + output_padding[0]
            + 1
        )

        if in_ch != in_channels:
            raise ShapeError(f"input channels mismatch: {in_ch} != {in_channels}.")

        if (_output_size := out_channels * out_l) != dest.num_in:
            raise ShapeError(f"Output size mismatch: {_output_size} != {dest.num_in}.")

        self.comm = ConvTranspose1dForward(
            (in_l,), (out_l,), _kernel, stride, padding, output_padding
        )


class ConvTranspose2dSyn(FullConnectedSyn):
    _spatial_ndim: ClassVar[int] = 2

    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: Neuron,
        kernel: np.ndarray,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        output_padding: tuple[int, int],
        order: _KOrder4d,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(source, dest, name)

        if kernel.ndim != self._spatial_ndim + 2:
            raise ShapeError(
                f"convolution kernel dimension must be {self._spatial_ndim + 2}, but got {kernel.ndim}."
            )

        if order == "IOHW":
            _kernel = np.swapaxes(kernel, 0, 1)
        else:
            _kernel = kernel.copy()

        # O,I,H,W
        out_channels, in_channels, kernel_h, kernel_w = _kernel.shape
        # C,H,W
        in_ch, in_h, in_w = _fm_ndim2_check(source.shape_out, "CHW")
        out_h = (
            (in_h - 1) * stride[0]
            - 2 * padding[0]
            + dilation[0] * (kernel_h - 1)
            + output_padding[0]
            + 1
        )
        out_w = (
            (in_w - 1) * stride[1]
            - 2 * padding[1]
            + dilation[1] * (kernel_w - 1)
            + output_padding[1]
            + 1
        )

        if in_ch != in_channels:
            raise ShapeError(f"input channels mismatch: {in_ch} != {in_channels}.")

        if (_output_size := out_channels * out_h * out_w) != dest.num_in:
            raise ShapeError(f"Output size mismatch: {_output_size} != {dest.num_in}.")

        self.comm = ConvTranspose2dForward(
            (in_h, in_w), (out_h, out_w), _kernel, stride, padding, output_padding
        )
