from typing import List, Optional

import numpy as np

from paibox.base import PAIBoxObject
from paibox.core.reg_model import ParamsReg
from paibox.core.reg_types import InputWidthFormatType
from paibox.core.reg_types import LCNExtensionType as LCN_EX
from paibox.core.reg_types import (
    MaxPoolingEnableType,
    SNNModeEnableType,
    SpikeWidthFormatType,
    WeightPrecisionType,
)

from .grouping import GroupedLayer
from .identifier import Coord


class Core(PAIBoxObject):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.weight_ram = np.zeros((1152, 512), dtype=np.bool_)

        self.set_default_params()

        for k, v in kwargs:
            setattr(self, k, v)

    @classmethod
    def build(cls, **kwargs):
        return cls(**kwargs)

    def set_default_params(self) -> None:
        self.core_id = Coord.default()
        self.weight_width = WeightPrecisionType.WEIGHT_WIDTH_1BIT
        self.lcn_ex = LCN_EX.LCN_1X
        self.input_width = InputWidthFormatType.WIDTH_1BIT
        self.spike_wdith = SpikeWidthFormatType.WIDTH_1BIT
        self.neuron_num = 512
        self.pool_max = MaxPoolingEnableType.DISABLE
        self.tick_wait_start = 0
        self.tick_wait_end = 0
        self.snn_en = SNNModeEnableType.ENABLE
        self.target_lcn = LCN_EX.LCN_1X
        self.test_chip_addr = Coord.default()

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

    @property
    def lcn(self) -> LCN_EX:
        return self.lcn_ex

    @property
    def connectivity(self) -> np.ndarray:
        return self.weight_ram

    @property
    def uid(self) -> Coord:
        return self.core_id


class Placement(PAIBoxObject):
    def __init__(
        self,
        grouped_layer: GroupedLayer,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(name)
        self.grouped_layer = grouped_layer
        self.myself = grouped_layer.myself
        self.target = grouped_layer.target
        self.n_core = [s.n_core for s in grouped_layer.grouped_syns]

        self.placed: List[Core] = []

        for n in self.n_core:
            for i in range(n):
                self.placed.append(Core.build())

    @classmethod
    def build(
        cls,
        grouped_layer: GroupedLayer,
        name: Optional[str] = None,
        **kwargs,
    ) -> "Placement":
        return cls(grouped_layer, name, **kwargs)

    def __repr__(self) -> str:
        return f"<{self.name} at 0x{id(self):x} of target '{self.myself}'>"

    def __str__(self) -> str:
        return f"<{self.name} of target '{self.myself}'>"

    @property
    def n_core_placement(self) -> int:
        return sum(self.n_core)
