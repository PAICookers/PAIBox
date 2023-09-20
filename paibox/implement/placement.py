from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from paibox.base import NeuDyn, PAIBoxObject
from paibox.libpaicore.v2 import (
    LCN_EX,
    InputWidthFormatType,
    MaxPoolingEnableType,
    SNNModeEnableType,
    SpikeWidthFormatType,
    WeightPrecisionType,
)
from paibox.libpaicore.v2.reg_model import ParamsReg

from .grouping import GroupedLayer, GroupedSyn
from .identifier import Coord


@dataclass
class CoreAttrs:
    """Parameters of registers."""

    weight_width: WeightPrecisionType = field(
        default=WeightPrecisionType.WEIGHT_WIDTH_1BIT
    )
    lcn_ex: LCN_EX = field(default=LCN_EX.LCN_1X)
    input_width: InputWidthFormatType = field(default=InputWidthFormatType.WIDTH_1BIT)
    spike_wdith: SpikeWidthFormatType = field(default=SpikeWidthFormatType.WIDTH_1BIT)
    neuron_num: int = field(default=512)
    pool_max: MaxPoolingEnableType = field(default=MaxPoolingEnableType.DISABLE)
    tick_wait_start: int = field(default=0)
    tick_wait_end: int = field(default=0)
    snn_en: SNNModeEnableType = field(default=SNNModeEnableType.ENABLE)
    target_lcn: LCN_EX = field(default=LCN_EX.LCN_1X)
    test_chip_addr: Coord = field(default_factory=Coord.default)

    def export_to_model(self) -> ParamsReg:
        return ParamsReg(
            weight_precision=self.weight_width,
            LCN_extension=self.lcn_ex,
            input_width_format=self.input_width,
            spike_width_format=self.spike_wdith,
            neuron_num=self.neuron_num,
            max_pooling_en=self.pool_max,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            snn_mode_en=self.snn_en,
            target_LCN=self.target_lcn,
            test_chip_addr=self.test_chip_addr.address,
        )


class Core(PAIBoxObject):
    """
    The object intends to describe the core of the chip.
    The core attributes include:
        1. core id.
        2. weight ram.
        3. the usages of axons & the usages of actual neurons.
        4. LCN extension.
        5. Target LCN extension, `target_lcn`.
        6. Others are FIXED.
    """

    def __init__(
        self,
        core_id: Coord,
        _weight_matrix: np.ndarray,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            core_id: the core coordinate. Default is `(0, 0)`.
            weight_matrix: the weight matrix. It's different from the binary connectivity.
            use_default_attrs: whether to use the default attributes. Default is `False`.
        """
        super().__init__(name)

        self.core_id = core_id
        self.attrs = CoreAttrs(**kwargs)
        self._weight_matrix = _weight_matrix

    @classmethod
    def build(
        cls,
        weight_matrix: Optional[np.ndarray] = None,
        *,
        lcn_ex: LCN_EX,
        **kwargs,
    ) -> "Core":
        """Build the core called by `Placement.__init__()`."""
        core_id = Coord.default()

        # TODO Consider 1-bit weight.
        if weight_matrix is None:
            _weight_matrix = np.zeros((1152, 512), dtype=np.bool_)
        else:
            _weight_matrix = weight_matrix

        return cls(core_id, _weight_matrix, lcn_ex=lcn_ex, **kwargs)

    def set_attrs(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self.attrs, k, v)

    def set_core_id(self, _id: Coord) -> None:
        self.core_id = _id

    def set_target_lcn(self, target_lcn: LCN_EX) -> None:
        self.attrs.target_lcn = target_lcn

    def set_neuron_used(self) -> None:
        self.attrs.neuron_num = self.n_neuron_used

    @property
    def lcn_ex(self) -> LCN_EX:
        return self.attrs.lcn_ex

    @property
    def weights(self) -> np.ndarray:
        return self._weight_matrix

    @property
    def binary_conn(self) -> np.ndarray:
        """ONLY for `np.bool_` now"""
        assert self.n_neuron_used * self.lcn_ex * 1 <= 512
        o_matrix = np.zeros((1152, 512), dtype=np.bool_)

        for i in range(self.n_neuron_used):
            o_matrix[:, 2 * i] = self.weights[:1152, i]
            o_matrix[:, 2 * i + 1] = np.pad(
                self.weights[1152:, i],
                (0, 2 * 1152 - self.n_axon_used),
                "constant",
                constant_values=0,
            )

        return o_matrix

    @property
    def n_core_per_dentrite(self) -> int:
        return 1 if self.weights.dtype == np.bool_ else 8

    @property
    def n_axon_used(self) -> int:
        return self.weights.shape[0]

    @property
    def n_neuron_used(self) -> int:
        return self.weights.shape[1]

    @property
    def uid(self) -> Coord:
        return self.core_id


def _syns_max_lcn_ex(syns: List[GroupedSyn]) -> LCN_EX:
    """Find the max LCN extenion of grouped post-synapses"""
    return max(syns, key=lambda syn: syn.lcn_ex).lcn_ex


class Placement(GroupedLayer):
    def __init__(
        self,
        myself: NeuDyn,
        pre_syns: List[GroupedSyn],
        post_syns: List[GroupedSyn],
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - grouped_layer: the grouped layer to be placed, \
                including the pre & post grouped synapses.
            - name: the name of the object. Optional.

        Description: The `Placement` includes both the pre & post grouped synapses. \
            It will be used to assign the parameters of CORES & NEURONS.

        NOTE: the grouped post-synapses of the placement is read-only. \
            We can ONLY place the previous grouped synspses.
        """
        super().__init__(myself, pre_syns, name)

        self._post_syns = post_syns
        self.placed: List[Core] = []

        """A list of the number of cores that each synapse needs."""
        self.n_core_placement = [s.n_core for s in self.pre]
        self.weights_placed = self._weights_placement()

        # For every synapse
        for i_syn in range(len(self.pre)):
            syn = self.pre[i_syn]
            for i in range(self.n_core_placement[i_syn]):
                self.placed.append(Core.build(syn.weight_divided[i], lcn_ex=syn.lcn_ex))

        for core in self.placed:
            if len(self.post) > 0:
                core.set_target_lcn(self.post[0].lcn_ex)
            else:
                core.set_target_lcn(LCN_EX.LCN_1X)

            core.set_neuron_used()

    def _weights_placement(self) -> List[List[np.ndarray]]:
        """Gather all the divided weights matrix in grouped synapses & place them.

        [0]: divided of matrix of <Synsyn1>
        [
            On core [0]:...
            On core [1]:...
            On core [2]:...
        ]
        [1]: divided of matrix of <Synsyn2>
        """
        weights = []
        for syn in self.pre:
            weights.append(syn.weight_divided)

        return weights

    @classmethod
    def build(
        cls,
        neurons: NeuDyn,
        grouped_syns: List[GroupedSyn],
        name: Optional[str] = None,
    ) -> "Placement":
        pre, post = cls._find_bi_syns(neurons, grouped_syns)

        # If the post-synapses are more than 1,
        # set the max LCN extension that satisfies all.
        if len(post) > 1:
            max_lcn_ex = _syns_max_lcn_ex(post)
            for syn in post:
                syn.lcn_ex = max_lcn_ex

        return cls(neurons, pre, post, name)

    def __repr__(self) -> str:
        return f"<{self.name} at 0x{id(self):x} of target '{self.myself}'>"

    def __str__(self) -> str:
        return f"<{self.name} of target '{self.myself}'>"

    @property
    def post(self) -> List[GroupedSyn]:
        return self._post_syns

    @property
    def dest(self) -> List[GroupedSyn]:
        return self.post
