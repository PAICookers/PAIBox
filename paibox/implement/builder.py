from collections import defaultdict
from typing import Any, Dict, Union

from paibox.base import PAIBoxObject, DynamicSys
from paibox.network import DynamicGroup
from paibox.neuron import Neuron
from paibox.synapses import Synapses


class Builder:
    def __init__(self) -> None:
        self._nodes: Dict[str, Neuron] = dict()

        self._graph: Dict[str, Dict[str, Union[Synapses, None]]] = defaultdict(dict)
        self._pred_dg: Dict[str, Dict[str, Synapses]] = defaultdict(dict)
        self._succ_dg: Dict[str, Dict[str, Synapses]] = defaultdict(dict)

    def build_graph(self, network: DynamicGroup) -> None:
        """Build the directed graph given a network.

        Arguments:
            - network: a DynamicGroup.

        For a single-layer network:
            INPUT -> S0 -> N1 -> S1 -> N2
                            |          -> S3 ->
                            ------------> S2 -> N3 -> OUTPUT

            _succ_dg: {
                INPUT: {S0},
                N1: {S1, S2},
                N2: {S3},
                N3: {OUTPUT?}
            }

            _pred_dg: {
                INPUT: {},
                N1: {S0},
                N2: {S1},
                N3: {S2, S3}
            }

            _nodes: {
                "n1": <n1>
            }
        """
        self.network = network

        for name, proc in self.network.__dict__.items():
            if isinstance(proc, Synapses):
                u, v = self._find_name(proc.source), self._find_name(proc.dest)

                # Add nodes
                if u not in self._nodes:
                    self._pred_dg[u] = dict()
                    self._succ_dg[u] = dict()
                    self._nodes[u] = self.network.__dict__[u]

                if v not in self._nodes:
                    self._pred_dg[v] = dict()
                    self._succ_dg[v] = dict()
                    self._nodes[v] = self.network.__dict__[v]

                # Add edges
                self._succ_dg[u][v] = proc
                self._pred_dg[v][u] = proc

        for k, v in self._nodes.items():
            print(k, v)

    def usage_constraint(self) -> Dict[DynamicSys, Dict[str, Any]]:
        """Calculate the resource usages of the network, as a constraint for core mapping.

            1. The number of synapses the neuron or group is connected with.
            2. The LCN extension of each neuron or group.
            3. Weight width of the neuron or group(1-bit in global config NOW)

        TODO: What if...

            4. Weight width of the neuron is not 8-bit.
            5. Input/Spike width is not 8-bit
        """
        neuron_attrs = dict()

        for node in self._nodes:
            if node not in neuron_attrs:
                neuron_attrs[node] = dict()

            neuron_attrs[node]["connected"] = node.num

        print(f"Usage: {neuron_attrs}")

        return neuron_attrs

    def _find_name(self, component: PAIBoxObject) -> str:
        for k, v in self.network.__dict__.items():
            if component is v:
                return k

        raise ValueError(f"Component {component} not found in the network.")

    def _find_src_syn(self, node: DynamicSys):
        pass

    def _find_dest_syn(self, node: DynamicSys):
        pass
