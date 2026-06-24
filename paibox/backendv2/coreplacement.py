from abc import abstractmethod

import numpy as np
from paicorelib import (
    FRAME_DTYPE,
    LCN_EX,
    AddPotentialMode,
    CoordXY,
    CoordZXYOffset,
    CSCAccelerateMode,
    DataWidth,
    FrameArrayType,
    InputCoreType,
    OfflineCoreRegV2,
    OfflineFrameGenV2,
    OnlineCoreRegV2,
    OnlineCoreWorkMode,
    OnlineDataWidth,
    OnlineFrameGenV2,
    OnlineSNNMode,
    OutputCoreType,
    PoolingMode,
    WeightCompressType,
    ZeroOutputMode,
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
from .weight import N_WEIGHTS_PER_SRAM, Weight

SRAM_RECORD_BITS = 128


class CorePlacement:
    def __init__(self) -> None:
        self._coord: CoordXY | None = None
        self.neus: list[NeuronPlacement] = (
            []
        )  # full or half neu, depending on neu allocation
        self.weights: list[Weight] = (
            []
        )  # weight of each single neu, can reuse for different single neu
        self.neu_weight_map: dict[int, int] = {}  # map from neu index to weight index
        self.auto_core_config: Auto_Core_Config = Auto_Core_Config()

    def max_input_num(self) -> int:
        max_input_num = 0
        for weight in self.weights:
            input_num = weight.processed_weights.size
            max_input_num = max(max_input_num, input_num)
        return max_input_num

    @property
    @abstractmethod
    def core_config(self) -> OfflineCoreRegV2 | OnlineCoreRegV2: ...

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
    def to_frame(
        self,
    ) -> tuple[FrameArrayType, FrameArrayType | None, FrameArrayType | None]:
        pass

    @abstractmethod
    def set_weight_address(self) -> None:
        pass

    @abstractmethod
    def set_auto_core_config(self, test_offset: CoordZXYOffset | None = None) -> None:
        pass

    @property
    @abstractmethod
    def n_sram_required(self) -> int:
        pass

    @property
    @abstractmethod
    def neuron_sram_required(self) -> int:
        pass

    @property
    @abstractmethod
    def weight_sram_required(self) -> int:
        pass

    def get_compute_pressure(self) -> int:
        pressure = 0
        for i, neu in enumerate(self.neus):
            fold_number = len(neu.raw_neus)
            weight = self.weights[self.neu_weight_map[i]]
            input_bits = 1 << min(int(weight.input_width), 3)
            weight_bits = 1 << min(int(weight.weight_width), 3)
            sram_record_count = weight.n_sram_required
            if weight.compress:
                slots_with_padding = sram_record_count * N_WEIGHTS_PER_SRAM.get(
                    weight.weight_width, 5
                )
            else:
                slots_with_padding = sram_record_count * (
                    SRAM_RECORD_BITS // weight_bits
                )
            pressure += fold_number * input_bits * weight_bits * slots_with_padding
        return pressure


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
        self.neus: list[OfflineNeuronPlacement] = []

    @property
    def n_sram_required(self) -> int:
        n_sram = 0
        for neu in self.neus:
            n_sram += neu.n_sram_required
        for weight in self.weights:
            n_sram += weight.n_sram_required
        return n_sram

    @property
    def neuron_sram_required(self) -> int:
        n_sram = 0
        for neu in self.neus:
            n_sram += neu.n_sram_required
        return n_sram

    @property
    def weight_sram_required(self) -> int:
        n_sram = 0
        for weight in self.weights:
            n_sram += weight.n_sram_required
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
            current_address += neu.n_sram_required
        weight_start_address = [current_address]
        for weight in self.weights:
            weight_start_address.append(
                weight_start_address[-1] + weight.n_sram_required
            )

        csc_sparse_full_neus: list[OfflineNeuronPlacement] = []
        has_nonzero_init_v = False
        for i, neu in enumerate(self.neus):
            weight_idx = self.neu_weight_map[i]
            neu.neu_attrs_part1.weight_address_start = weight_start_address[weight_idx]
            neu.neu_attrs_part1.weight_address_end = (
                weight_start_address[weight_idx + 1] - 1
            )
            if (
                self.default_core_config.csc_accelerate == CSCAccelerateMode.ENABLE
                and neu.neu_attrs_part2 is not None
                and neu.neu_attrs_part2.weight_compress == WeightCompressType.SPARSE
            ):
                csc_sparse_full_neus.append(neu)
                if neu.neu_attrs_part2.vjt_initial != 0:
                    has_nonzero_init_v = True

        if has_nonzero_init_v:
            self.default_core_config.csc_accelerate = CSCAccelerateMode.DISABLE
            return

        for neu in csc_sparse_full_neus:
            if neu.neu_attrs_part2 is not None:
                neu.neu_attrs_part2.vjt_initial = (
                    neu.neu_attrs_part1.weight_address_start
                )

    def _weight_skews(self, weight_idx: int) -> list[int]:
        skews: list[int] = []
        for neu_idx, neu in enumerate(self.neus):
            if self.neu_weight_map.get(neu_idx) != weight_idx:
                continue

            base_skew = neu.neu_attrs_part1.weight_skew
            skews.append(base_skew)

            if (folded_attrs := neu.folded_neu_attrs_part1) is not None:
                skews.extend(
                    [
                        base_skew + folded_attrs.fold_skew_y,
                        base_skew + folded_attrs.fold_skew_x,
                        base_skew + folded_attrs.fold_skew_xy,
                    ]
                )

        return skews or [0]

    def set_auto_core_config(self, test_offset: CoordZXYOffset | None = None) -> None:
        if test_offset is None:
            test_offset, _ = find_coordxy_shortest_path(TEST_DEST_CORE, self.coord)

        self.auto_core_config.neuron_number = sum(
            neu.n_sram_required for neu in self.neus
        )
        self.auto_core_config.test_core_xy = test_offset.z
        self.auto_core_config.test_core_x = test_offset.x
        self.auto_core_config.test_core_y = test_offset.y

    def to_frame(
        self,
    ) -> tuple[FrameArrayType, FrameArrayType | None, FrameArrayType]:
        pkt_offset, _ = find_coordxy_shortest_path(self.coord)

        # frame_type_1: core config
        frame_type1 = OfflineFrameGenV2.gen_config_frame1(pkt_offset, self.core_config)

        frame_type2: FrameArrayType | None = None
        # frame_type_2: lut config
        if self.frontend_core_config.hw_lut_data is not None:
            # hw_lut_data is already validated for PAICORE 2.5 SRAM packing.
            potential_tensor = self.frontend_core_config.hw_lut_data.thresholds
            activation_tensor = self.frontend_core_config.hw_lut_data.values
            potentials = potential_tensor.numpy()
            activations = activation_tensor.numpy()

            frame_type2 = OfflineFrameGenV2.gen_config_frame2(
                pkt_offset, potentials, activations
            )

        package_arrays: list[FrameArrayType] = []
        for neu in self.neus:
            package_arrays.append(neu.to_package())

        for weight_idx, weight in enumerate(self.weights):
            if weight.compress:
                package_arrays.append(weight.to_package(self._weight_skews(weight_idx)))
            else:
                package_arrays.append(weight.to_package())

        packages = np.concatenate(package_arrays, axis=0).astype(FRAME_DTYPE)

        start_frame = OfflineFrameGenV2.gen_config_frame3_pkg_header(
            pkt_offset, 0, len(packages)
        )

        frame_type3 = np.concatenate([start_frame, packages], axis=0).astype(
            FRAME_DTYPE
        )

        return frame_type1, frame_type2, frame_type3


class EmptyOfflineCorePlacementV2(OfflineCorePlacementV2):
    def to_frame(
        self,
    ) -> tuple[FrameArrayType, FrameArrayType | None, FrameArrayType | None]:
        pkt_offset, _ = find_coordxy_shortest_path(self.coord)

        # frame_type_1: core config
        frame_type1 = OfflineFrameGenV2.gen_config_frame1(pkt_offset, self.core_config)
        return frame_type1, None, None


class EmptyOnlineCorePlacementV2(CorePlacement):
    """Minimal empty online core used only as a global signal relay.

    backendv2 does not yet carry a complete online-core configuration policy.
    The class exposes coord and auto core config state so the planner can reason
    about this fallback, and exports only the online core config frame1 needed
    for global signal send/receive and control routing.
    """

    def _unsupported_empty_online_property(self, name: str) -> NotImplementedError:
        return NotImplementedError(
            f"EmptyOnlineCorePlacementV2.{name} is not implemented. "
            "This placement is only a global signal relay with online frame1 export."
        )

    @property
    def n_sram_required(self) -> int:
        raise self._unsupported_empty_online_property("n_sram_required")

    @property
    def weight_sram_required(self) -> int:
        raise self._unsupported_empty_online_property("weight_sram_required")

    @property
    def neuron_sram_required(self) -> int:
        raise self._unsupported_empty_online_property("neuron_sram_required")

    @property
    def core_config(self) -> OnlineCoreRegV2:
        return OnlineCoreRegV2(
            name=f"empty_online_core_reg_at_({self.coord.x},{self.coord.y})",
            snn_ann=OnlineSNNMode.SNN_LIF,
            max_pooling=PoolingMode.AVERAGE,
            add_potential=AddPotentialMode.NORMAL,
            zero_output=ZeroOutputMode.DISABLE,
            work_mode=OnlineCoreWorkMode.FORWARD_INFERENCE,
            input_core=InputCoreType.OFFLINE,
            input_width=OnlineDataWidth.TYPE_1BIT,
            output_core=OutputCoreType.OFFLINE,
            output_width=OnlineDataWidth.TYPE_1BIT,
            lcn_at=LCN_EX.LCN_1X,
            lcn_mp=LCN_EX.LCN_1X,
            lcn_lg=LCN_EX.LCN_1X,
            target_lcn_at=LCN_EX.LCN_1X,
            target_lcn_mp=LCN_EX.LCN_1X,
            target_lcn_lg=LCN_EX.LCN_1X,
            axon_skew=0,
            neuron_number=0,
            update_number=0,
            csc_accelerate=CSCAccelerateMode.DISABLE,
            scale_in=1.0,
            bias_in=0.0,
            scale_out=1.0,
            bias_out=0.0,
            learning_rate=0.0,
            update_core_xy=0,
            update_core_x=0,
            update_core_y=0,
            test_core_xy=self.auto_core_config.test_core_xy,
            test_core_x=self.auto_core_config.test_core_x,
            test_core_y=self.auto_core_config.test_core_y,
            global_send=self.auto_core_config.global_send,
            global_receive=self.auto_core_config.global_receive,
            thread_number=0,
            busy_cycle=20,
            delay_cycle=20,
            width_cycle=10,
            tick_start=0,
            tick_duration=0,
            tick_initial=0,
        )

    @property
    def output_width(self) -> DataWidth:
        raise self._unsupported_empty_online_property("output_width")

    def set_weight_address(self) -> None:
        return None

    def set_auto_core_config(self, test_offset: CoordZXYOffset | None = None) -> None:
        if test_offset is None:
            test_offset, _ = find_coordxy_shortest_path(TEST_DEST_CORE, self.coord)

        self.auto_core_config.neuron_number = 0
        self.auto_core_config.test_core_xy = test_offset.z
        self.auto_core_config.test_core_x = test_offset.x
        self.auto_core_config.test_core_y = test_offset.y

    def to_frame(
        self,
    ) -> tuple[FrameArrayType, FrameArrayType | None, FrameArrayType | None]:
        pkt_offset, _ = find_coordxy_shortest_path(self.coord)

        # frame_type_1: minimal online core config for global signal relay.
        frame_type1 = OnlineFrameGenV2.gen_config_frame1(pkt_offset, self.core_config)
        return frame_type1, None, None
