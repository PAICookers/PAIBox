import copy
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from paibox.base import DynamicSys, PAIBoxObject
from paibox.context import _FRONTEND_CONTEXT
from paibox.exceptions import SimulationError

from .probe import Probe

__all__ = ["Simulator"]


class Simulator(PAIBoxObject):
    def __init__(
        self,
        target: DynamicSys,
        start_time_zero: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - target: the target network.
            - start_time_zero: whether to start the simulation at time 0. If `False`, \
                it will start & record at time 1. Default is `False`.
        """
        if not isinstance(target, DynamicSys):
            raise SimulationError(
                f"target must be an instance of {DynamicSys.__name__}, but got {target}, {type(target)}."
            )

        super().__init__(name)

        self.target = target
        self.dt = 1
        """Time scale."""
        self._ts = 0
        """Time stamp."""
        self._start_time_zero = start_time_zero
        """Whether to start the simulation at time 0."""

        self._sim_data = dict()
        self._sim_data["ts"] = []  # Necessary key for recording timestamp

        self.data = _SimulationData(self._sim_data)
        self.probes: List[Probe] = []

        self._add_inner_probes()
        self.reset()

    def run(self, duration: int, reset: bool = False, **kwargs) -> None:
        """
        Arguments:
            - duration: duration of the simulation.
            - reset: whether to reset the state of components in the model. Default is `False`.
            - kwargs：determined by the parameter format of the input node. It will be deprecated, \
                please use 'FRONTEND_ENV.save()' instead.
        """
        if kwargs:
            warnings.warn(
                "passing extra arguments through 'run()' will be deprecated. "
                "Use 'FRONTEND_ENV.save()' instead.",
                DeprecationWarning,
            )

        if duration < 1:
            raise SimulationError(f"duration must be positive, but got {duration}.")

        n_steps = self._get_nstep(duration)
        if n_steps < 1:
            raise SimulationError(
                f"the number of simulation steps must be positive, but got {n_steps}."
            )

        indices = np.arange(self._ts, self._ts + n_steps, dtype=np.uint16)

        if reset:
            self.target.reset_state()

        self._run_step(indices, **kwargs)

        self._sim_data["ts"].extend(indices * self.dt)
        self._ts += n_steps

    def reset(self) -> None:
        """Reset simulation data & network."""
        # The global timestep start at 1 if excluding time 0.
        self._ts = 0 if self._start_time_zero else 1
        _FRONTEND_CONTEXT["t"] = self.timestamp

        self.target.reset_state()
        self._reset_probes()

    def add_probe(self, probe: Probe) -> None:
        """Add an external probe into the simulator."""
        if probe not in self.probes:
            self.probes.append(probe)
            self._sim_data[probe] = []

    def remove_probe(self, probe: Probe) -> None:
        """Remove a probe from the simulator."""
        if probe in self.probes:
            self.probes.remove(probe)
            self._sim_data.pop(probe)
        else:
            raise KeyError(f"probe '{probe.name}' does not exist.")

    def _run_step(self, indices: NDArray[np.uint16], **kwargs) -> None:
        for i in range(indices.shape[0]):
            _FRONTEND_CONTEXT["t"] = indices[i]
            self.target.update(**kwargs)
            self._update_probes()

    def _destroy_probes(self):
        self.probes.clear()
        self._sim_data.clear()
        self.data.reset()

    def get_raw(self, probe: Probe) -> List[Any]:
        """Retrieve the raw data.

        Argument:
            - probe: the probe to retrieve.
            - t: retrieve the data at time `t`.

        NOTE: For faster access, use the attribute of `data`.
        """
        return self._sim_data[probe]

    def get_raw_at_t(self, probe: Probe, t: int) -> Any:
        """Retrieve the raw data at time `t`.

        Argument:
            - probe: the probe to retrieve.
            - t: retrieve the data at time `t`.

        NOTE: For faster access, use the attribute of `data`.
        """
        t_start = 0 if self._start_time_zero else 1
        t_index = t if self._start_time_zero else t - 1

        if not t_start <= t < self.timestamp:  # [t_start, timestamp)
            raise IndexError(f"time {t} is out of range [{t_start}, {self.timestamp}).")

        return self._sim_data[probe][t_index]

    def _reset_probes(self) -> None:
        """Reset the probes."""
        for probe in self.probes:
            self._sim_data[probe].clear()

        self.data.reset()

    def _get_nstep(self, duration: int) -> int:
        return int(duration / self.dt)

    def _update_probes(self) -> None:
        """Update probes."""
        for probe in self.probes:
            t = getattr(probe.target, probe.attr)
            data = t.copy() if hasattr(t, "copy") else copy.copy(t)  # Shallow copy

            self._sim_data[probe].append(data)

    def _add_inner_probes(self) -> None:
        # Find probes at all levels.
        probe_nodes = (
            self.target.nodes(include_self=False, find_recursive=True)
            .subset(Probe)
            .unique()
        )

        for probe in probe_nodes.values():
            # Store the probe instances
            self.probes.append(probe)
            self._sim_data[probe] = []

    @property
    def timestamp(self) -> int:
        """Timestamp of simulator. Simulation at this time is not finished."""
        return self._ts


class _SimulationData(dict):
    """Data structure used to retrieve and access the simulation data."""

    def __init__(self, raw: Dict[Probe, List[Any]]) -> None:
        super().__init__()
        self.raw = raw
        self._cache = {}

    def __getitem__(self, key) -> Any:
        """
        Return simulation data for ``key`` object.

        For speed reasons, the simulator uses Python lists for Probe data and we
        want to return NumPy arrays.
        """
        if key not in self._cache or len(self._cache[key]) != len(self.raw[key]):
            val = self.raw[key]
            if isinstance(val, list):
                val = np.asarray(val)
                val.setflags(write=False)

            self._cache[key] = val

        return self._cache[key]

    def __iter__(self):
        return iter(self.raw)

    def __len__(self) -> int:
        return len(self.raw)

    def __repr__(self) -> str:
        return repr(self.raw)

    def __str__(self) -> str:
        return str(self.raw)

    def reset(self) -> None:
        self._cache.clear()
