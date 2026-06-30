"""Pressure-based core unrolling."""

from collections.abc import Callable, Iterator
from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
from paicorelib import CSCAccelerateMode, NeuronType

from .coreplacement import CorePlacement, OfflineCorePlacementV2
from .neuron import OfflineNeuronPlacement
from .route_solver import OFFLINE_CORE_COORDS
from .routing import RoutingGroup


class PressureUnrollStopReason(Enum):
    """Why the pressure unroll loop stopped."""

    MAX_TRY = "max_try"
    NO_FREE_CORE = "no_free_core"
    NO_PRESSURE = "no_pressure"
    NO_CANDIDATE = "no_candidate"
    ROUTING_FAILED = "routing_failed"


@dataclass(frozen=True)
class CorePressure:
    """Pressure value for one offline core in a layer candidate."""

    rg: RoutingGroup
    index: int
    core: OfflineCorePlacementV2
    pressure: int


@dataclass(frozen=True)
class LayerPressureProfile:
    """Aggregated offline-core pressures for one source layer."""

    label: str
    pressures: list[CorePressure] = field(default_factory=list)

    @property
    def peak_pressure(self) -> int:
        return max((item.pressure for item in self.pressures), default=0)


@dataclass(frozen=True)
class PressureUnrollConfig:
    """Internal knobs for the pressure-unroll heuristic."""

    max_try: int = 4
    """Maximum routing probes; success and failure both consume one."""
    max_split_factor: int | None = 2
    """Cap on pieces per split core; 2 means bisect, None chooses max feasible."""
    core_selection: Literal["peak_ratio", "quantile"] = "peak_ratio"
    peak_ratio: float = 0.8
    """For peak_ratio mode, select cores with pressure >= layer_peak * ratio."""
    quantile: float = 0.5
    """For quantile mode, select cores at or above this quantile; 0.5 is median."""

    def __post_init__(self) -> None:
        if self.max_try < 1:
            raise ValueError("'max_try' must be at least 1.")
        if self.max_split_factor is not None and self.max_split_factor < 2:
            raise ValueError("'max_split_factor' must be at least 2 when set.")
        if self.core_selection not in ("peak_ratio", "quantile"):
            raise ValueError("'core_selection' must be 'peak_ratio' or 'quantile'.")
        if not 0 < self.peak_ratio <= 1:
            raise ValueError("'peak_ratio' must be in (0, 1].")
        if not 0 <= self.quantile < 1:
            raise ValueError("'quantile' must be in [0, 1).")


@dataclass
class UnrollResult:
    """Diagnostic summary of one pressure unroll run."""

    n_try: int = 0
    n_commit: int = 0
    stop_reason: PressureUnrollStopReason | None = None
    peak_drops: list[tuple[str, int, int]] = field(default_factory=list)


SplitCoreGroup = tuple[OfflineCorePlacementV2, ...]
RoutingGroupSplitPlan = dict[int, SplitCoreGroup]
LayerSplitPlan = dict[RoutingGroup, RoutingGroupSplitPlan]


@dataclass(frozen=True)
class NeuronSplitRange:
    """Contiguous source-neuron range assigned to one split core."""

    start: int
    end: int


@dataclass
class PressureUnroller:
    """Greedily split high-pressure layer cores while routing remains feasible.

    Each round recomputes layer pressure, tries candidates ordered by current
    peak pressure, and commits at most one successful layer split.
    """

    routing_groups: list[RoutingGroup]
    routing_fn: Callable[[], None]  # must be a feasibility-only probe
    config: PressureUnrollConfig = field(default_factory=PressureUnrollConfig)

    def run(self) -> UnrollResult:
        """Run greedy split/probe rounds within the route-probe budget.

        Strategy:
        1. Group routing groups by source layer.
        2. Pick the current highest-pressure layer.
        3. Select high-pressure offline cores by peak ratio or quantile.
        4. Split selected cores by neuron count.
        5. Keep the split only if a feasibility-only routing probe succeeds.
        """
        result = UnrollResult()
        route_failed = False

        while result.n_try < self.config.max_try:
            n_free_core = self._get_free_offline_core_count()
            if n_free_core <= 0:
                result.stop_reason = PressureUnrollStopReason.NO_FREE_CORE
                return result

            profiles = self._layer_pressure_profiles()
            if not profiles:
                result.stop_reason = PressureUnrollStopReason.NO_PRESSURE
                return result

            candidates = self._splittable_profiles(profiles, n_free_core)
            if not candidates:
                result.stop_reason = PressureUnrollStopReason.NO_CANDIDATE
                return result

            split_committed = False
            for profile, selected_cores, split_factor in candidates:
                if result.n_try >= self.config.max_try:
                    result.stop_reason = (
                        PressureUnrollStopReason.ROUTING_FAILED
                        if route_failed and result.n_commit == 0
                        else PressureUnrollStopReason.MAX_TRY
                    )
                    return result

                result.n_try += 1
                if self._try_split_layer(profile, selected_cores, split_factor, result):
                    result.n_commit += 1
                    split_committed = True
                    break

                route_failed = True

            if not split_committed:
                result.stop_reason = PressureUnrollStopReason.ROUTING_FAILED
                return result

        result.stop_reason = PressureUnrollStopReason.MAX_TRY
        if route_failed and result.n_commit == 0:
            result.stop_reason = PressureUnrollStopReason.ROUTING_FAILED
        return result

    def _splittable_profiles(
        self, profiles: list[LayerPressureProfile], n_free_core: int
    ) -> list[tuple[LayerPressureProfile, list[CorePressure], int]]:
        """Return layer candidates that fit the current free-core budget."""
        candidates: list[
            tuple[int, int, float, LayerPressureProfile, list[CorePressure], int]
        ] = []
        for profile in profiles:
            selected_cores = self._selected_cores(profile)
            split_factor = self._max_split_factor(selected_cores, n_free_core)
            cores_to_split = self._cores_to_split(
                selected_cores, split_factor, n_free_core
            )
            extra_core = len(cores_to_split) * (split_factor - 1)
            if not cores_to_split or extra_core <= 0:
                continue

            layer_split_plan = self._build_layer_split_plan(
                cores_to_split, split_factor
            )
            peak_after = self._peak_pressure_after_split(
                profile.pressures, layer_split_plan
            )
            peak_drop = profile.peak_pressure - peak_after
            if peak_drop <= 0:
                continue

            # Candidate score tuple:
            # 1. peak_pressure: handle the current bottleneck layer first.
            # 2. -extra_core: prefer the cheaper split when peaks tie.
            # 3. peak_drop / extra_core: then prefer better drop per new core.
            candidates.append(
                (
                    profile.peak_pressure,
                    -extra_core,
                    peak_drop / extra_core,
                    profile,
                    cores_to_split,
                    split_factor,
                )
            )

        return [
            (profile, cores_to_split, split_factor)
            for *_, profile, cores_to_split, split_factor in sorted(
                candidates, key=lambda item: item[:3], reverse=True
            )
        ]

    def _try_split_layer(
        self,
        profile: LayerPressureProfile,
        cores_to_split: list[CorePressure],
        split_factor: int,
        result: UnrollResult,
    ) -> bool:
        """Apply one layer split transaction and keep it only if routing passes."""
        layer_split_plan = self._build_layer_split_plan(cores_to_split, split_factor)

        peak_after = self._peak_pressure_after_split(
            profile.pressures, layer_split_plan
        )
        original_core_placements = {
            rg: rg.core_placements.copy() for rg in layer_split_plan
        }
        for rg, rg_split_plan in layer_split_plan.items():
            rg.core_placements = self._replace_split_cores(rg, rg_split_plan)

        try:
            self.routing_fn()
        except RuntimeError as e:
            print(f"Error occurred while checking routing: {e}")
            for rg, core_placements in original_core_placements.items():
                rg.core_placements = core_placements
            print(
                f"Routing is invalid after unrolling {profile.label}, "
                "reverted changes."
            )
            return False

        print("Routing is still valid after unrolling.")
        for rg_split_plan in layer_split_plan.values():
            for split_group in rg_split_plan.values():
                for core in split_group:
                    core.set_weight_address()
        result.peak_drops.append((profile.label, profile.peak_pressure, peak_after))
        print(
            f"Unrolled {len(cores_to_split)} selected cores in {profile.label} "
            f"with factor {split_factor}; peak pressure dropped from "
            f"{profile.peak_pressure} to {peak_after}."
        )
        return True

    def _get_free_offline_core_count(self) -> int:
        return len(OFFLINE_CORE_COORDS) - sum(
            rg.n_core_required for rg in self.routing_groups
        )

    def _layer_pressure_profiles(self) -> list[LayerPressureProfile]:
        """Build pressure profiles after grouping routing groups by source layer."""
        layers: dict[tuple[int, ...], tuple[str, list[CorePressure]]] = {}
        for rg in self.routing_groups:
            key, label = rg.layer_key()
            layers.setdefault(key, (label, []))[1].extend(
                self._offline_core_pressures(rg)
            )

        profiles: list[LayerPressureProfile] = []
        for label, pressures in layers.values():
            profile = LayerPressureProfile(label, pressures)
            if profile.peak_pressure <= 0:
                continue
            profiles.append(profile)

        return profiles

    def _offline_core_pressures(self, rg: RoutingGroup) -> list[CorePressure]:
        """Compute pressures for offline cores; online cores are ignored."""
        return [
            CorePressure(rg, idx, core, core.get_compute_pressure())
            for idx, core in enumerate(rg.core_placements)
            if isinstance(core, OfflineCorePlacementV2)
        ]

    def _selected_cores(self, profile: LayerPressureProfile) -> list[CorePressure]:
        """Pick splittable cores by the configured pressure threshold."""
        if self.config.core_selection == "peak_ratio":
            threshold = profile.peak_pressure * self.config.peak_ratio
        else:
            threshold = float(
                np.quantile(
                    [core_pressure.pressure for core_pressure in profile.pressures],
                    self.config.quantile,
                )
            )

        selected_cores = [
            core_pressure
            for core_pressure in profile.pressures
            if core_pressure.pressure >= threshold
            and core_pressure.pressure > 0
            and len(core_pressure.core.neus) > 1
        ]
        return sorted(selected_cores, key=lambda item: item.pressure, reverse=True)

    def _max_split_factor(
        self, selected_cores: list[CorePressure], n_free_core: int
    ) -> int:
        """Bound the split factor so at least one selected core can fit."""
        if not selected_cores:
            return 1
        k_max = min(
            n_free_core + 1,
            min(len(core_pressure.core.neus) for core_pressure in selected_cores),
        )
        if self.config.max_split_factor is not None:
            k_max = min(k_max, self.config.max_split_factor)
        return k_max

    def _cores_to_split(
        self, selected_cores: list[CorePressure], split_factor: int, n_free_core: int
    ) -> list[CorePressure]:
        """Return the pressure-sorted subset that the current core budget can split."""
        n_extra_per_core = split_factor - 1
        if n_extra_per_core <= 0:
            return []
        n_split_core = min(len(selected_cores), n_free_core // n_extra_per_core)
        return selected_cores[:n_split_core]

    def _build_layer_split_plan(
        self, selected_cores: list[CorePressure], split_factor: int
    ) -> LayerSplitPlan:
        layer_split_plan: LayerSplitPlan = {}
        for candidate in selected_cores:
            layer_split_plan.setdefault(candidate.rg, {})[candidate.index] = (
                self._split_offline_core(candidate.core, split_factor)
            )
        return layer_split_plan

    def _split_offline_core(
        self, core: OfflineCorePlacementV2, split_factor: int
    ) -> SplitCoreGroup:
        """Split one offline core into split_factor parts by neuron count."""

        def _copy_core(src: OfflineCorePlacementV2) -> OfflineCorePlacementV2:
            new = OfflineCorePlacementV2(
                src.frontend_core_config, src.backend_core_config
            )
            new.default_core_config = copy(src.default_core_config)
            return new

        split_cores = []
        for split_range in self._neuron_split_ranges(len(core.neus), split_factor):
            split_core = _copy_core(core)
            split_core.neus = self._copy_split_neurons(core, split_range)
            self._assign_split_weights(core, split_core, split_range)
            split_cores.append(split_core)

        return tuple(split_cores)

    def _copy_split_neurons(
        self, core: OfflineCorePlacementV2, span: NeuronSplitRange
    ) -> list[OfflineNeuronPlacement]:
        clear_vjt_initial = (
            core.default_core_config.csc_accelerate == CSCAccelerateMode.ENABLE
        )
        split_neus = [
            neu.copy_for_readdress(clear_vjt_initial=clear_vjt_initial)
            for neu in core.neus[span.start : span.end]
        ]
        if not split_neus:
            return split_neus

        if split_neus[0].neuron_type == NeuronType.HALF:
            # A split may start at Part2 of a half neuron. Rebuild that boundary
            # neuron as full by copying Part1 from itself and Part2 from the
            # nearest previous full neuron.
            # F1 H1.1 H1.2 H1.3 ... F2 H2.1 H2.2 | H2.3 H2.4 ... F3 ...
            # F1 H1.1 H1.2 H1.3 ... F2 H2.1 H2.2 + F2 H2.1 H2.2 ... F3 ...
            # F=full neuron, H=half neuron
            inherited = next(
                (
                    neu
                    for neu in reversed(core.neus[: span.start])
                    if neu.neuron_type == NeuronType.FULL
                ),
                None,
            )
            if inherited is None:
                raise ValueError(
                    "can not find a previous full neuron for split-leading half neuron"
                )
            attrs_part1 = split_neus[0].neu_attrs_part1.model_copy(
                update={"neuron_type": NeuronType.FULL}
            )
            attrs_part2 = inherited.copy_for_readdress(
                clear_vjt_initial=clear_vjt_initial
            ).neu_attrs_part2
            split_neus[0] = OfflineNeuronPlacement(
                split_neus[0].raw_neus,
                attrs_part1,
                attrs_part2,
                split_neus[0].folded_neu_attrs_part1,
                split_neus[0].folded_neu_attrs_part2s,
            )
        return split_neus

    def _neuron_split_ranges(
        self, n_items: int, split_factor: int
    ) -> Iterator[NeuronSplitRange]:
        """Yield near-even contiguous source-neuron ranges.

        Example: 10 neurons split by 3 gives ranges [0:4], [4:7], [7:10], so
        the first ranges receive the remainder.
        """
        base_size, remainder = divmod(n_items, split_factor)
        start = 0
        for split_idx in range(split_factor):
            end = start + base_size + (1 if split_idx < remainder else 0)
            yield NeuronSplitRange(start, end)
            start = end

    def _assign_split_weights(
        self,
        source_core: OfflineCorePlacementV2,
        split_core: OfflineCorePlacementV2,
        span: NeuronSplitRange,
    ) -> None:
        source_to_split_weight: dict[int, int] = {}

        def split_weight_index(source_weight_idx: int) -> int:
            if source_weight_idx not in source_to_split_weight:
                # Multiple neurons in one split range may share a source weight.
                source_to_split_weight[source_weight_idx] = len(split_core.weights)
                split_core.weights.append(source_core.weights[source_weight_idx])
            return source_to_split_weight[source_weight_idx]

        for source_neu_idx in range(span.start, span.end):
            split_core.neu_weight_map[source_neu_idx - span.start] = split_weight_index(
                source_core.neu_weight_map[source_neu_idx]
            )

    def _peak_pressure_after_split(
        self, pressures: list[CorePressure], layer_split_plan: LayerSplitPlan
    ) -> int:
        """Return the simulated layer peak after replacing selected cores."""
        pressures_after: list[int] = []
        for core_pressure in pressures:
            rg_split_plan = layer_split_plan.get(core_pressure.rg, {})
            if core_pressure.index in rg_split_plan:
                pressures_after.extend(
                    split_core.get_compute_pressure()
                    for split_core in rg_split_plan[core_pressure.index]
                )
            else:
                pressures_after.append(core_pressure.pressure)
        return max(pressures_after, default=0)

    def _replace_split_cores(
        self, rg: RoutingGroup, rg_split_plan: RoutingGroupSplitPlan
    ) -> list[CorePlacement]:
        """Build the committed placement list for a successful split plan."""
        new_core_placements = rg.core_placements.copy()
        for idx in sorted(rg_split_plan, reverse=True):  # Insert in desc order
            new_core_placements[idx : idx + 1] = list(rg_split_plan[idx])
        return new_core_placements
