from collections import defaultdict
import numpy as np
from typing import Any, Dict

from paibox.collector import Collector
from paibox.core.reg_types import LCNExtensionType as LCN_EX
from paibox.network import Network

from .builder import Builder


class Mapper:
    def __init__(self, network: Network) -> None:
        self.builder = Builder()
        self.builder.build_graph(network)
        self.core_group = Collector()

    def do_grouping(self):
        """
        Prerequisites:
        0. Global config:
            - All weights is 1-bit
            - Pure SNN mode. SNN_EN enbales, input and spike width are 1-bit. Max pooling is disabled.

        1. Simplified situation level 0:
            - Every layer is considered seperately.
            - The number of a group of neurons <= 512.

        2. TODO Level 1:
            - The number of a group of neurons may > 512 and the LCN extension of every neuron is LCN_1X.

        """
        # Get the usage constraints of neurons.
        neurons_usages = self.builder.usage_constraint()

        """1. Group by layer"""
        # Do nothing

        """2. Group neurons into cores based on LCN extension & the number of it."""
        for layer_usages in neurons_usages.values():
            group1 = self._grouping_greedy_lcnx_opt(layer_usages)

        """3. Split based on number of axons"""

    def _grouping_greedy_lcnx_opt(self, groups: Dict[str, Any]):
        """LCN extension optimization strategy for grouping neurons.

        Description:
            Always looking for the minimum LCN extension that the group of axons is **ALL** satisfied.

        TODO Summary the constant in this function such as 1152, 512, etc.
        """
        axons_in_layer: np.ndarray = groups["axons_each"]
        n_neuron: int = groups["neuron_num"]

        indices = np.argsort(axons_in_layer, kind="heapsort")
        axons_in_layer.sort(kind="heapsort")

        lcn_each_in_layer = []

        for i in range(n_neuron):
            lcn_ex = np.ceil(axons_in_layer[i] / 1152) - 1
            lcn_each_in_layer.append(LCN_EX(lcn_ex))

        # Traverse the lcn and put them in core
        def _get_limit(indice: int):
            """Get the limit when axon is [indice]"""
            lcn_ex_max_in_core = lcn_each_in_layer[indice]
            n_max_in_core = int(512 / (2**lcn_ex_max_in_core))
            axons_max_in_core = 1152 * (lcn_ex_max_in_core - LCN_EX.LCN_1X + 1)

            return lcn_ex_max_in_core, n_max_in_core, axons_max_in_core

        i = i_last = 0
        axons_grouped = []  # The length of it is the number of cores needed.

        while i < n_neuron:
            i_last = i
            axons_group_sum = axons_in_layer[i_last]
            l, n, a = _get_limit(i_last)

            while (i - i_last) + 1 < n and axons_group_sum + axons_in_layer[i] < a:
                """
                When the axon with index i is successfully grouped, check whether the next axon exists.
                If so, update the sum of axons & limit. Otherwise, break the loop.
                """
                if i == len(axons_in_layer) - 1:
                    break

                i += 1
                axons_group_sum += axons_in_layer[i]
                l, n, a = _get_limit(i)

            axons_grouped.append(
                # Slice the array [i_last: i+1], which means [i_last, i].
                AxonGroup(l, axons_in_layer[i_last : i + 1], indices[i_last : i + 1])
            )
            i += 1
            
        return axons_grouped

    # def _group_by_axons(self, groups: Dict[LCN_EX, List[Neuron]]):
    #     result = defaultdict(list)
    #     r: List[List[Neuron]] = [[]]

    #     for k, v in groups.items():
    #         for node in v:
    #             _sum = 0
    #             for node in r[-1]:
    #                 _sum += node.shape_in

    #             if _sum + node.shape_in <= 1152:
    #                 r[-1].append(node)
    #             else:
    #                 r.append([node])

    #         result[k] = r

    #     return result


def group_by(_dict: Dict, keyfunc=lambda item: item):
    """Groups the given list or dictionary by the value returned by ``keyfunc``."""
    d = defaultdict(list)

    for item in _dict.values():
        d[keyfunc(item)].append(item)

    return d


class AssembleGroup(dict):
    pass
