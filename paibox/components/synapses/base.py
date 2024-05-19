from typing import ClassVar, Optional, Tuple, Union

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
    Conv1dForward,
    Conv2dForward,
    ConvTranspose1dForward,
    ConvTranspose2dForward,
)
from .transforms import GeneralConnType as GConnType
from .transforms import Identity, MaskedLinear, OneToOne, Transform

RIGISTER_MASTER_KEY_FORMAT = "{0}.output"


def _check_equal(num_in: int, num_out: int) -> int:
    if num_in != num_out:
        raise ShapeError(
            f"the number of source & destination neurons must be equal: {num_in} != {num_out}."
        )

    return num_in


class Synapses:
    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: NeuDyn,
        subclass_syn_name: str,
    ) -> None:
        self._source = source
        self._target = dest
        self._child_syn_name = subclass_syn_name
        """The name of subclass `FullConnectedSyn`."""

    @property
    def source(self) -> Union[NeuDyn, InputProj]:
        return self._source

    @source.setter
    def source(self, source: Union[NeuDyn, InputProj]) -> None:
        """Set a new source neuron."""
        if source.num_out != self.num_in:
            raise RegisterError(
                f"the number of source neurons before and after the change"
                f"is not equal: {source.num_out} != {self.num_in}."
            )

        self._source = source

    @property
    def dest(self) -> NeuDyn:
        return self._target

    @dest.setter
    def dest(self, dest: NeuDyn) -> None:
        """Set a new destination neuron."""
        if dest.num_in != self.num_out:
            raise RegisterError(
                f"the number of source neurons before and after the change"
                f"is not equal: {dest.num_in} != {self.num_out}."
            )

        self._target = dest
        # FIXME Because the modification of the synapse destination neuron occurs in the backend,
        # there's no need to register new dest again because simulation will not be done again (maybe).
        # But does it mean that we need to make a copy of the original network and then pass it to
        # the backend?
        dest.register_master(
            RIGISTER_MASTER_KEY_FORMAT.format(self._child_syn_name), self
        )

    @property
    def target(self) -> NeuDyn:
        return self._target

    @property
    def shape_in(self) -> Tuple[int, ...]:
        return self._source.shape_out

    @property
    def shape_out(self) -> Tuple[int, ...]:
        return self._target.shape_in

    @property
    def num_in(self) -> int:
        return self._source.num_out

    @property
    def num_out(self) -> int:
        return self._target.num_in


class FullConnectedSyn(Synapses, SynSys):

    comm: Transform

    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: NeuDyn,
        name: Optional[str] = None,
    ) -> None:
        super(Synapses, self).__init__(name)
        super().__init__(source, dest, self.name)

        self.set_memory("_synout", np.zeros((self.num_out,), dtype=np.int32))

        # Register itself with the master nodes of destination.
        dest.register_master(RIGISTER_MASTER_KEY_FORMAT.format(self.name), self)

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

    def _set_comm(self, comm: Transform) -> None:
        self.comm = comm

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
        dest: NeuDyn,
        weights: DataArrayType,
        conn_type: GConnType,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(source, dest, name)

        if conn_type is GConnType.One2One:
            comm = OneToOne(_check_equal(self.num_in, self.num_out), weights)
        elif conn_type is GConnType.Identity:
            if not isinstance(weights, (int, np.bool_, np.integer)):
                raise TypeError(
                    f"expected type int, np.bool_, np.integer, but got type {type(weights)}."
                )
            comm = Identity(_check_equal(self.num_in, self.num_out), weights)
        elif conn_type is GConnType.All2All:
            comm = AllToAll((self.num_in, self.num_out), weights)
        else:  # MatConn
            if not isinstance(weights, np.ndarray):
                raise TypeError(
                    f"expected type np.ndarray, but got type {type(weights)}."
                )
            comm = MaskedLinear((self.num_in, self.num_out), weights)

        self._set_comm(comm)


class Conv1dSyn(FullConnectedSyn):
    _spatial_ndim: ClassVar[int] = 1

    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: Neuron,
        kernel: np.ndarray,
        stride: Tuple[int],
        padding: Tuple[int],
        dilation: Tuple[int],
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

        comm = Conv1dForward((in_l,), (out_l,), _kernel, stride, padding)

        self._set_comm(comm)


class Conv2dSyn(FullConnectedSyn):
    _spatial_ndim: ClassVar[int] = 2

    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: Neuron,
        kernel: np.ndarray,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
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
            raise ShapeError(f"Output size mismatch: {_output_size} != {dest.num_in}.")

        comm = Conv2dForward((in_h, in_w), (out_h, out_w), _kernel, stride, padding)

        self._set_comm(comm)


class ConvTranspose1dSyn(FullConnectedSyn):
    _spatial_ndim: ClassVar[int] = 1

    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: Neuron,
        kernel: np.ndarray,
        stride: Tuple[int],
        padding: Tuple[int],
        dilation: Tuple[int],
        output_padding: Tuple[int],
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

        comm = ConvTranspose1dForward(
            (in_l,), (out_l,), _kernel, stride, padding, output_padding
        )

        self._set_comm(comm)


class ConvTranspose2dSyn(FullConnectedSyn):
    _spatial_ndim: ClassVar[int] = 2

    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: Neuron,
        kernel: np.ndarray,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        output_padding: Tuple[int, int],
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

        comm = ConvTranspose2dForward(
            (in_h, in_w), (out_h, out_w), _kernel, stride, padding, output_padding
        )

        self._set_comm(comm)
