from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import numpy as np
from paicorelib import (
    FRAME_DTYPE,
    CoordXY,
    DataWidth,
    FrameArrayType,
    NeuronType,
    OfflineCoreRegV2,
    OfflineFrameGenV2,
    find_coordxy_shortest_path,
)

from .core_config import (
    TEST_DEST_CORE,
    Auto_Core_Config,
    Backend_Core_Config,
    Default_Core_Config,
    Frontend_Core_Config,
    to_core_reg,
)
from .neuron import NeuronPlacement, OfflineNeuronPlacement

# from .routing import RoutingGroup
from .weight import Weight


class CorePlacement:
    def __init__(
        self,
    ) -> None:
        self._coord: Optional[CoordXY] = None
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
    @abstractmethod
    def core_config(self) -> OfflineCoreRegV2:
        pass

    @property
    @abstractmethod
    def output_width(self) -> DataWidth:
        pass

    @property
    def coord(self) -> CoordXY:
        if self._coord is None:
            raise ValueError("coord has not been set yet.")
        return self._coord

    @abstractmethod
    def to_frame(self) -> tuple[FrameArrayType, FrameArrayType]:
        pass

    @abstractmethod
    def set_weight_address(self) -> None:
        pass

    @abstractmethod
    def set_auto_core_config(self) -> None:
        pass


class OfflineCorePlacementV2(CorePlacement):
    def __init__(
        self,
        frontend_core_config: Frontend_Core_Config = Frontend_Core_Config(),
        backend_core_config: Backend_Core_Config = Backend_Core_Config(),
    ) -> None:
        super().__init__()
        self.frontend_core_config: Frontend_Core_Config = frontend_core_config
        self.backend_core_config: Backend_Core_Config = backend_core_config
        self.default_core_config: Default_Core_Config = Default_Core_Config()
        self.auto_core_config: Auto_Core_Config = Auto_Core_Config()
        self.neus: list[OfflineNeuronPlacement] = []

    def n_sram_required(self) -> int:
        n_sram = 0
        for neu in self.neus:
            n_sram += neu.n_sram_required()
        for weight in self.weights:
            n_sram += weight.n_sram_required()
        return n_sram

    @property
    def core_config(self) -> OfflineCoreRegV2:
        core_reg = to_core_reg(
            default_conf=self.default_core_config,
            auto_conf=self.auto_core_config,
            backend_conf=self.backend_core_config,
            frontend_conf=self.frontend_core_config,
            coord=self.coord,
        )
        return core_reg

    @property
    def output_width(self) -> DataWidth:
        return self.frontend_core_config.output_width

    def set_weight_address(self) -> None:
        current_address = 0
        for neu in self.neus:
            current_address += neu.n_sram_required()
        weight_start_address = [current_address]
        for weight in self.weights:
            weight_start_address.append(
                weight_start_address[-1] + weight.n_sram_required()
            )
        for i, neu in enumerate(self.neus):
            weight_idx = self.neu_weight_map[i]
            selected_weight = self.weights[weight_idx]
            neu.neu_attrs_part1.weight_address_start = weight_start_address[weight_idx]
            neu.neu_attrs_part1.weight_address_end = (
                weight_start_address[weight_idx + 1] - 1
            )

    def set_auto_core_config(self) -> None:
        neuron_number = 0
        for neu in self.neus:
            neu_count = 1 if neu.neuron_type == NeuronType.HALF else 2
            neuron_number += neu_count
        self.auto_core_config.neuron_number = neuron_number
        pkt_offset, _ = find_coordxy_shortest_path(self.coord, TEST_DEST_CORE)
        self.auto_core_config.test_core_xy = pkt_offset.z
        self.auto_core_config.test_core_x = pkt_offset.x
        self.auto_core_config.test_core_y = pkt_offset.y

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


class EmptyOfflineCorePlacementV2(OfflineCorePlacementV2):
    pass
