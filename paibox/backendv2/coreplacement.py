from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import numpy as np
from .neuron import NeuronPlacement, OfflineNeuronPlacement
from paicorelib import (
    FRAME_DTYPE,
    CoordXY,
    FrameArrayType,
    OfflineCoreRegV2,
    OfflineFrameGenV2,
    find_coordxy_shortest_path,
)
#from .routing import RoutingGroup
from .weight import Weight


class CorePlacement:
    def __init__(
        self,
    ) -> None:
        self._coord: Optional[CoordXY] = None
        self._core_config: Optional[OfflineCoreRegV2] = None
        self.neus: list[NeuronPlacement] = (
            []
        )  # full or half neu, depending on neu allocation
        self.weights: list[Weight] = (
            []
        )  # weight of each single neu, can reuse for different single neu
        self.neu_weight_map: dict[int, int] = {}  # map from neu index to weight index

    def max_input_num(self) -> int:
        max_input_num = 0
        for weight in self.weights:
            input_num = len(weight.processed_weights)
            max_input_num = max(max_input_num, input_num)
        return max_input_num

    @property
    def core_config(self) -> OfflineCoreRegV2:
        if self._core_config is None:
            raise ValueError("core_config has not been set yet.")
        return self._core_config

    @property
    def coord(self) -> CoordXY:
        if self._coord is None:
            raise ValueError("coord has not been set yet.")
        return self._coord

    @abstractmethod
    def to_frame(self) -> tuple[FrameArrayType, FrameArrayType]:
        pass


class EmptyOfflineCorePlacementV2(CorePlacement):
    def __init__(self):
        super().__init__()


class OfflineCorePlamentV2(CorePlacement):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.neus: list[OfflineNeuronPlacement] = []

    def n_sram_required(self) -> int:
        n_sram = 0
        for neu in self.neus:
            n_sram += neu.n_sram_required()
        for weight in self.weights:
            n_sram += weight.n_sram_required()
        return n_sram

    def to_frame(self) -> tuple[FrameArrayType, FrameArrayType]:
        pkt_offset, _ = find_coordxy_shortest_path(self.coord)

        # frame_type_1: core config
        frame_type1 = OfflineFrameGenV2.gen_config_frame1(
            pkt_offset=pkt_offset,
            core_reg_=self.core_config,
        )

        package_arrays: list[FrameArrayType] = []
        for neu in self.neus:
            package_arrays.append(neu.to_package())

        for weight in self.weights:
            package_arrays.append(weight.to_package())

        packages = np.concatenate(package_arrays, axis=0).astype(FRAME_DTYPE)

        start_frame = OfflineFrameGenV2.gen_config_frame3_pkg_header(
            pkt_offset=pkt_offset,
            start_addr=0,
            n_package=len(packages),
        )

        frame_type3 = np.concatenate([start_frame, packages], axis=0).astype(
            FRAME_DTYPE
        )

        return frame_type1, frame_type3
