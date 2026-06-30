import numpy as np
from paicorelib import (
    FRAME_DTYPE,
    FrameArrayType,
    NeuDestInfoV2,
    NeuronType,
    OfflineFrameGenV2,
    OfflineNeuDestInfoV2,
    OfflineNeuFoldedAttrsV2Part1,
    OfflineNeuFoldedAttrsV2Part2,
    OfflineNeuFullAttrsV2Part1,
    OfflineNeuFullAttrsV2Part2,
)

from .op_node import Neuron


class NeuronPlacement:
    def __init__(self, neu: list[Neuron]) -> None:
        self.raw_neus: list[Neuron] = neu
        self.dest_info: NeuDestInfoV2 | None = None


class OfflineNeuronPlacement(NeuronPlacement):
    def __init__(
        self,
        neu: list[Neuron],
        attrs_part1: OfflineNeuFullAttrsV2Part1,
        attrs_part2: OfflineNeuFullAttrsV2Part2 | None,
        fold_attrs_part1: OfflineNeuFoldedAttrsV2Part1 | None = None,
        fold_attrs_part2s: list[OfflineNeuFoldedAttrsV2Part2] | None = None,
    ) -> None:
        super().__init__(neu)
        self.neu_attrs_part1: OfflineNeuFullAttrsV2Part1 = attrs_part1
        self.neu_attrs_part2: OfflineNeuFullAttrsV2Part2 | None = (
            attrs_part2 if attrs_part1.neuron_type == NeuronType.FULL else None
        )
        self.folded_neu_attrs_part1: OfflineNeuFoldedAttrsV2Part1 | None = (
            fold_attrs_part1
        )
        self.folded_neu_attrs_part2s: list[OfflineNeuFoldedAttrsV2Part2] = (
            list(fold_attrs_part2s) if fold_attrs_part2s is not None else []
        )
        self.n_sram_required: int = self.n_sram_required_()

    def copy_for_readdress(
        self, clear_vjt_initial: bool = False
    ) -> "OfflineNeuronPlacement":
        """Copy this placement before recomputing weight addresses.

        `set_weight_address()` may have written derived weight addresses and CSC
        `vjt_initial` values into attrs. Split cores need fresh attrs but should
        keep the same raw graph neurons.
        """
        old_weight_start = self.neu_attrs_part1.weight_address_start
        attrs_part1 = self.neu_attrs_part1.model_copy(
            update={"weight_address_start": 0, "weight_address_end": 0, "vjt": 0}
        )

        attrs_part2 = (
            self.neu_attrs_part2.model_copy(
                update={
                    "vjt_initial": (
                        0
                        if clear_vjt_initial
                        and self.neu_attrs_part2.vjt_initial == old_weight_start
                        else self.neu_attrs_part2.vjt_initial
                    )
                }
            )
            if self.neu_attrs_part2 is not None
            else None
        )

        folded_part2s = [
            attrs.model_copy(
                update={
                    "fold_vjt_0": 0,
                    "fold_vjt_1": 0,
                    "fold_vjt_2": 0,
                    "fold_vjt_3": 0,
                }
            )
            for attrs in self.folded_neu_attrs_part2s
        ]

        return OfflineNeuronPlacement(
            self.raw_neus,
            attrs_part1,
            attrs_part2,
            (
                self.folded_neu_attrs_part1.model_copy()
                if self.folded_neu_attrs_part1 is not None
                else None
            ),
            folded_part2s,
        )

    def n_sram_required_(self) -> int:
        n_sram = 0
        if self.neu_attrs_part1 is not None:
            n_sram += 1
        if self.neu_attrs_part2 is not None:
            n_sram += 1
        if self.folded_neu_attrs_part1 is not None:
            n_sram += 1
        n_sram += len(self.folded_neu_attrs_part2s)
        return n_sram

    def to_package(self) -> FrameArrayType:
        if self.dest_info is None:
            raise ValueError("dest_info has not been set yet.")
        if not isinstance(self.dest_info, OfflineNeuDestInfoV2):
            raise TypeError("dest_info must be of type OfflineNeuDestInfoV2.")

        half_neu, full_neu, fold_neu = OfflineFrameGenV2.gen_config_frame3_pkg_neu(
            dest_info=self.dest_info,
            full_attrs1=self.neu_attrs_part1,
            full_attrs2=self.neu_attrs_part2,
            folded_attrs1=self.folded_neu_attrs_part1,
            folded_attrs2_=self.folded_neu_attrs_part2s,
        )

        if self.neuron_type == NeuronType.HALF:
            frame_list: FrameArrayType = np.concatenate(
                [half_neu, fold_neu], axis=0
            ).astype(FRAME_DTYPE)
        else:
            frame_list: FrameArrayType = np.concatenate(
                [full_neu, fold_neu], axis=0
            ).astype(FRAME_DTYPE)
        return frame_list

    @property
    def neuron_type(self) -> NeuronType:
        return self.neu_attrs_part1.neuron_type
