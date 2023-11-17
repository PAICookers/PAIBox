from typing import Any, Dict, List

import numpy as np

from paibox.base import DynamicSys, PAIBoxObject
from paibox.exceptions import SimulationError

from .probe import Probe

__all__ = ["Simulator"]


class Simulator(PAIBoxObject):
    def __init__(
        self,
        target: DynamicSys,
    ) -> None:
        """
        Arguments:
            - target: the target network.
        """
        super().__init__()

        self.target = target
        self.dt = 1
        """Time scale."""
        self._ts = 0
        """Current time."""

        self._sim_data = dict()
        self.data = SimulationData(self._sim_data)
        self.probes: List[Probe] = []

        self._add_inner_probes()
        self.reset()

    def run(self, duration: int, reset: bool = True, **kwargs) -> None:
        """
        Arguments:
            - duration: duration of the simulation.
            - reset: whether to reset the model state.
        """
        if duration < 0:
            raise SimulationError(f"duration should be > 0, but got {duration}")

        n_steps = self._get_nstep(duration)
        if n_steps == 0:
            raise SimulationError(f"Step of simulation should be > 0, but got {n_steps}")

        indices = np.arange(self._ts, self._ts + n_steps, dtype=np.int16)

        if reset:
            self.target.reset_state()

        self.run_step(n_steps, **kwargs)

        self._sim_data["ts"] = indices * self.dt
        self._ts += n_steps

    def run_step(self, n_steps: int, **kwargs) -> None:
        for step in range(n_steps):
            self.step(step, **kwargs)

    def step(self, *args, **kwargs) -> None:
        self.target.update(*args, **kwargs)
        self._update_probe()

    def reset(self) -> None:
        self._ts = 0
        self._clear_probes()

    def add_probe(self, probe: Probe) -> None:
        if probe not in self.probes:
            self.probes.append(probe)
            self._sim_data[probe] = []

    def remove_probe(self, probe: Probe) -> None:
        if probe in self.probes:
            self.probes.remove(probe)
            self._sim_data.pop(probe)
        else:
            raise KeyError(f"Probe {probe.name} does not exist.")

    def get_raw(self, probe: Probe) -> List[Any]:
        """Retrieve the raw data.

        Argument:
            - probe: the probe to retrieve.
            - t: retrieve the data at time `t`.

        NOTE: For faster access, use the `data` attribute.
        """
        return self._sim_data[probe]

    def get_raw_at_t(self, probe: Probe, t: int) -> Any:
        """Retrieve the raw data at time `t`.

        Argument:
            - probe: the probe to retrieve.
            - t: retrieve the data at time `t`.

        NOTE: For faster access, use the `data` attribute.
        """
        if t >= self.time:
            raise IndexError(f"Time {t} is out of range {self.time-1}.")

        return self._sim_data[probe][t]

    def _clear_probes(self) -> None:
        for probe in self.probes:
            self._sim_data[probe] = []

        self.data.reset()

    def _get_nstep(self, duration: int) -> int:
        return int(duration / self.dt)

    def _update_probe(self) -> None:
        for probe in self.probes:
            # Shallow copy
            t = getattr(probe.target, probe.attr)
            data = t.copy() if hasattr(t, "copy") else t

            self._sim_data[probe].append(data)

    def _add_inner_probes(self) -> None:
        probe_nodes = (
            self.target.nodes(level=1, include_self=False).subset(Probe).unique()
        )

        for probe in probe_nodes.values():
            # Store the probe instances
            self.probes.append(probe)
            self._sim_data[probe] = []

    @property
    def time(self) -> int:
        return self._ts


class SimulationData(dict):
    """Data structure used to retrieve and access the simulation data."""

    def __init__(self, raw: Dict[Probe, List[Any]]) -> None:
        super().__init__()
        self.raw = raw
        self._cache = {}

    def __getitem__(self, key):
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
