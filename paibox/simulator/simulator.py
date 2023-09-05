from typing import Any, List
import numpy as np

from paibox.base import DynamicSys
from paibox.collector import Collector
from paibox.mixin import singleton
from .probe import Probe


@singleton
class Simulator:
    def __init__(
        self,
        target: DynamicSys,
        dt: int = 1,
        t0: int = 0,
    ) -> None:
        """
        Arguments:
            - target: the target network.
            - dt: the time step.
            - t0: the initial time.
        """
        self.target = target

        # Timescale
        self.dt = dt
        self.t0 = t0
        # Inner status of the simulator. Time scale unit.
        self._ts1 = 0

        self._sim_data = dict()
        self.data = SimulationData(self._sim_data)
        self.probes: List[Probe] = []

        probe_nodes = (
            self.target.nodes(level=1, include_self=False).subset(Probe).unique()
        )

        if len(tuple(probe_nodes)) > 0:
            self._build_probes(probe_nodes)

        self.reset()

    def run(self, duration: int, *args, **kwargs) -> None:
        """
        Arguments:
            - duration: duration of the simulation.
        """
        if duration < 0:
            # TODO
            raise ValueError

        steps = self._get_time_step(duration)
        if steps == 0:
            # TODO
            raise ValueError

        indices = np.arange(self._ts1, self._ts1 + steps, self.dt, dtype=np.int16)
        
        self.run_step(steps, *args, **kwargs)
        
        self._sim_data["ts"] = indices * self.dt + self.t0
        self._ts1 += steps

    def run_step(self, steps: int, *args, **kwargs) -> None:
        for i in range(steps):
            self.step(*args, **kwargs)

    def step(self, *args, **kwargs) -> None:
        self.target.update(*args, **kwargs)

        self._update_probe()

    def reset(self) -> None:
        self._ts1 = 0
        self.clear_probes()

    def clear_probes(self):
        for probe in self.probes:
            self._sim_data[probe] = []

        self.data.reset()

    def _get_time_step(self, duration: int) -> int:
        return int(duration / self.dt)

    def _update_probe(self) -> None:
        for probe in self.probes:
            # Shallow copy
            t = getattr(probe.target, probe.attr)
            data = t.copy() if hasattr(t, "copy") else t

            self._sim_data[probe].append(data)

    def _build_probes(self, c: Collector) -> None:
        for probe in c.values():
            # Store the probe instances
            self.probes.append(probe)
            self._sim_data.update({probe: []})

    def add_probe(self, probe: Probe) -> None:
        if probe not in self.probes:
            self.probes.append(probe)
            self._sim_data.update({probe: []})
        else:
            # TODO
            raise ValueError(f"Probe {probe} already exists.")


class SimulationData(dict):
    """Data structure used to retrive and access the simulation data."""

    def __init__(self, raw: Any) -> None:
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
