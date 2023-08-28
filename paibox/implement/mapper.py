from collections import defaultdict
from typing import Any, List, Dict

from paibox.core.reg_types import LCNExtensionType
from paibox.neuron import Neuron
from .builder import Builder
from paibox.network import Network
from paibox.collector import Collector


class Mapper:
    def __init__(self, network: Network) -> None:
        self.builder = Builder()
        self.builder.build_graph(network)
        self.core_group = Collector()

    def assemble(self):
        """
        Prerequisites:
        1. Global config:
            - All weights is 1-bit
            - Pure SNN mode. SNN_EN enbales, input and spike width are 1-bit. Max pooling is disabled.


        1. Calculate the number of types of cores needed according to the neuron settings.


        2.

        """
        usages = self.builder.usage_constraint()

        """1. Split based on weight width"""

        """2. Split based on LCN"""
        group2 = self._group_by_lcn(usages)

        """3. Split based on number of axons"""
        self.group3 = self._group_by_axons(group2)

    def _group_by_lcn(self, groups: Dict[Neuron, Dict[str, Any]]) -> Dict[LCNExtensionType, List[Neuron]]:
        result = defaultdict(list)
        
        for k, v in groups.items():
            result[v["lcn"]].append(k)

        return result

    def _group_by_axons(self, groups: Dict[LCNExtensionType, List[Neuron]]):
        result = defaultdict(list)
        r: List[List[Neuron]] = [[]]

        for k, v in groups.items():
            for node in v:
                _sum = 0
                for node in r[-1]:
                    _sum += node.shape_in

                if _sum + node.shape_in <= 1152:
                    r[-1].append(node)
                else:
                    r.append([node])

            result[k] = r

        return result


class AssembleGroup(dict):
    pass
