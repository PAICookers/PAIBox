import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Union, cast

from paicorelib import HwConfig

from paibox.base import DataFlowFormat
from paibox.collector import Collector
from paibox.components import FullConnectedSyn, InputProj, NeuModule, Neuron
from paibox.components.functional import LinearSemiFolded
from paibox.exceptions import (
    GraphBuildError,
    GraphConnectionError,
    GraphNotSupportedError,
)
from paibox.network import DynSysGroup
from paibox.utils import check_elem_unique

from .context import _BACKEND_CONTEXT
from .graph_utils import reverse_edges, toposort, get_node_degrees
from .placement import CoreBlock
from .routing import RoutingGroup
from .segment_utils import get_neu_segments
from .succ_group import *
from .types import *

__all__ = ["PAIGraph"]


@dataclass
class PAIGraph:
    """Directed graph of PAIBox. We treat networks as one whole graph. \
        In the graph, synapses are edges while neurons are nodes.
    """

    _raw_networks: tuple[DynSysGroup, ...] = field(default_factory=tuple)
    """All networks are seen as one graph."""
    _raw_nodes: Collector[NodeName, NodeType] = field(default_factory=Collector)
    """Raw nodes in the networks."""
    _raw_edges: Collector[EdgeName, EdgeType] = field(default_factory=Collector)
    """Raw edges in the graph."""
    _raw_fmodules: Collector[NodeName, NeuModule] = field(default_factory=Collector)
    """Raw functional modules in the graph."""

    nodes: dict[NodeName, NodeAttr] = field(default_factory=dict)
    """General nodes in the graph."""
    edges: dict[EdgeName, EdgeAttr] = field(default_factory=dict)
    """General edges in the graph."""

    inodes: Collector[NodeName, SourceNodeType] = field(default_factory=Collector)
    """Input nodes in the graph."""
    onodes: Collector[NodeName, DestNodeType] = field(default_factory=Collector)
    """Output nodes in the graph."""

    ordered_nodes: list[NodeName] = field(default_factory=list)
    """Nodes in topological sort order."""

    succ_dg: dict[NodeName, dict[NodeName, EdgeAttr]] = field(default_factory=dict)
    """Successor edges & nodes of every node in the graph."""

    pred_dg: dict[NodeName, dict[NodeName, EdgeAttr]] = field(default_factory=dict)
    """Predecessor edges & nodes of every node in the graph."""

    degree_of_nodes: dict[NodeName, NodeDegree] = field(default_factory=dict)
    """A dictionary of in/out-degree tuple of nodes."""

    """Status options"""
    has_built: bool = field(default=False)

    def clear(self, total: bool = True) -> None:
        """Clear the PAIGraph."""
        self.has_built = False

        self.nodes.clear()
        self.edges.clear()
        self.inodes.clear()
        self.onodes.clear()
        self.ordered_nodes.clear()
        self.succ_dg.clear()
        self.pred_dg.clear()
        self.degree_of_nodes.clear()

        if total:
            self._raw_networks = ()
            self._raw_nodes.clear()
            self._raw_edges.clear()
            self._raw_fmodules.clear()

    def build(self, *networks: DynSysGroup, **build_options) -> None:
        self.clear()

        if not check_elem_unique(networks):
            raise GraphBuildError("duplicated networks are not allowed.")

        self._raw_networks = networks
        self._pre_build(**build_options)

        nodes: Collector[NodeName, NodeType] = Collector()
        edges: Collector[EdgeName, EdgeType] = Collector()
        fm: Collector[NodeName, NeuModule] = Collector()

        for subnet in self._raw_networks:
            fm += subnet.nodes().subset(NeuModule).unique()
            nodes += (
                subnet.nodes().include(InputProj, Neuron).exclude(NeuModule).unique()
            )
            edges += subnet.nodes().subset(FullConnectedSyn).unique()

        self._raw_nodes += nodes.val_on_condition(
            lambda node: not node.__gh_build_ignore__
        )
        self._raw_edges += edges.val_on_condition(
            lambda edge: not edge.__gh_build_ignore__
        )
        self._raw_fmodules = fm

        self._update_graph(**build_options)

    def _pre_build(self, **build_options) -> None:
        """Preprocessing before obtaining the topology."""
        # Check the hardware resource limits of operators in the network during the build phase.
        build_options.setdefault("check_before_compile", True)

        # Build functional modules for each network.
        for network in self._raw_networks:
            if network.is_composed_of_semi_folded_ops():
                modules = network.components.subset(NeuModule)
                succ_dg_semi_ops = {
                    name: [t.name for t in op.target] for name, op in modules.items()
                }
                pred_dg_semi_ops = reverse_edges(succ_dg_semi_ops)

                # XXX Networks consisting entirely of semi-folded operators require some additional topology
                # checks. These additional checks may be removed as more network structures will be supported.

                # Currently, `LinearSemiFolded` is at the end of the network, since it will change the form of
                # the input dataflow, and its effective output is at the same time.
                semi_linears = modules.subset(LinearSemiFolded)
                if not all(
                    len(succ_dg_semi_ops[linear]) == 0 for linear in semi_linears
                ):
                    raise GraphNotSupportedError(
                        "currently, the semi-folded linear can only be used as output of the network."
                    )

                ordered_nodes = [modules[name] for name in toposort(succ_dg_semi_ops)]
                network.build_modules(pred_dg_semi_ops, ordered_nodes, **build_options)
            else:
                network.build_modules(**build_options)

    def _update_graph(self, **build_options) -> None:
        self.clear(total=False)

        # TODO Check isolated nodes in _raw_nodes
        for node in self._raw_nodes:
            self.succ_dg[node] = dict()
            self.pred_dg[node] = dict()

        for name, syn in self._raw_edges.items():
            u, v = syn.source.name, syn.dest.name
            if u not in self._raw_nodes:
                raise GraphConnectionError(
                    f"the source neuron {u} of {syn.name} is not included in the graph."
                )

            if v not in self._raw_nodes:
                raise GraphConnectionError(
                    f"the dest neuron {v} of {syn.name} is not included in the graph."
                )

            _edge_attr = EdgeAttr(edge=syn, distance=syn.source.delay_relative)
            self.edges[name] = _edge_attr
            self.succ_dg[u][v] = _edge_attr
            self.pred_dg[v][u] = _edge_attr

        # Check the uniqueness of the successors/predecessors of each node.
        for n in self.succ_dg:
            check_elem_unique(self.succ_dg[n])

        for n in self.pred_dg:
            check_elem_unique(self.pred_dg[n])

        self.degree_of_nodes = get_node_degrees(self.succ_dg)

        # `InputProj` nodes are input nodes definitely.
        self.inodes = self._raw_nodes.subset(InputProj)

        # By default, nodes with out-degree = 0 are considered as output nodes.
        # TODO A node with out-degree can also be an output node. However, no network for now has this topology.
        self.onodes = Collector(
            {
                k: cast(DestNodeType, v)
                for k, v in self._raw_nodes.items()
                if self.degree_of_nodes[k].out_degree == 0
            }
        ).not_subset(InputProj)

        for name, node in self._raw_nodes.items():
            self.nodes[name] = NodeAttr(
                node, self._node_pos(name), self.degree_of_nodes[name]
            )

        self.ordered_nodes = toposort(self.succ_dg)
        self.has_built = True

    def untwist_branch_nodes(self) -> None:
        # FIXME Input nodes may need to be excluded from the nodes to be traversed?
        for node_nn in filter(
            lambda node: self.degree_of_nodes[node].out_degree > 1,
            reversed(self.ordered_nodes),
        ):
            # succ_dg will be updated in _copy_node, so use the copy of succ_dg.
            for succ_nn in self.succ_dg[node_nn].copy():
                # Checking the out-degree of node_nn every time is necessary.
                # The out-degree of node_nn will be changed after coping.
                if (
                    self.degree_of_nodes[succ_nn].in_degree > 1
                    and self.degree_of_nodes[node_nn].out_degree > 1
                ):
                    node = self._raw_nodes[node_nn]
                    self._copy_node(
                        node, keep_pred_conn=True, grab_succ_nodes=succ_nn, update=False
                    )

        self._update_graph()

    def topo_support_check(self) -> None:
        # _degree_check(self.degree_of_nodes, self.succ_dg)

        # Only support output nodes with <= 1152 neurons so far.
        if any(
            onode.num_out > HwConfig.N_FANIN_PER_DENDRITE_MAX
            for onode in self.onodes.values()
        ):
            raise GraphNotSupportedError(
                f"only output nodes with no more than {HwConfig.N_FANIN_PER_DENDRITE_MAX} "
                f"neurons are supported."
            )

    def _node_pos(self, node: NodeName) -> NodePosition:
        if node in self.inodes:
            return NodePosition.INPUT
        elif node in self.onodes:
            return NodePosition.OUTPUT
        else:
            return NodePosition.MEMBER

    def build_check(self) -> None:
        if not self.has_built:
            raise GraphBuildError("the graph hasn't been built yet.")

    def graph_partition(self) -> list[MergedSuccGroup]:
        """Graph partition."""
        # Build the `SuccGroup` for each node in the graph.
        succ_grps: list[SuccGroup] = []
        for nn in self.ordered_nodes:
            if succ_nodes := self.succ_dg[nn]:
                succ_grps.append(SuccGroup(e.edge for e in succ_nodes.values()))

        def dfs(sgrp: SuccGroup, msgrp: MergedSuccGroup) -> None:
            # Union-find sets. If the nodes of two `succ_grps` have intersection, merge them.
            for other_sgrp in succ_grps:
                if other_sgrp not in visited and not set(sgrp.nodes).isdisjoint(
                    other_sgrp.nodes
                ):
                    visited.add(other_sgrp)
                    msgrp.add_group(other_sgrp)
                    dfs(other_sgrp, msgrp)

        # Merge
        merged_sgrps: list[MergedSuccGroup] = []
        visited: set[SuccGroup] = set()
        for sgrp in succ_grps:
            if sgrp not in visited:
                m = MergedSuccGroup([sgrp])
                visited.add(sgrp)
                dfs(sgrp, m)
                merged_sgrps.append(m)

        return merged_sgrps

    def multicast_optim(
        self,
        core_blocks: list[CoreBlock],
        routing_groups: list[RoutingGroup],
        optim_nodes: tuple[NodeName, ...] = (),
    ) -> bool:
        """Multicast optimization.

        NOTE: Only applies to a node that only has 2 successors, and they belong to the same core block.
        """
        raise NotImplementedError

        "the following code is not used, but it may be useful in the future."
        ONLY_SUPPORT_N_SUCC = 2

        def _roundup_to_pow2(n: int) -> int:
            assert n > 0
            return 1 if n < 1 else 2 ** math.ceil(math.log(n, 2))

        is_optimized = False

        if optim_nodes == ():
            _optim_nodes = list(reversed(self.ordered_nodes))
        else:
            _optim_nodes = optim_nodes

        # visit ordered nodes for end to front
        for node_name in filter(lambda node: isinstance(node, Neuron), _optim_nodes):
            node = self._raw_nodes[node_name]

            succ_nn = list(self.succ_dg[node_name].keys())
            if len(succ_nn) != ONLY_SUPPORT_N_SUCC:
                continue

            succ_cbs = get_succ_cb_by_node(node, core_blocks)
            pred_cbs = get_pred_cb_by_node(node, core_blocks)

            # the node to be optimized can only has one successor core block & predecessor core block.
            if len(succ_cbs) != 1 or len(pred_cbs) != 1:
                continue

            succ_cb = succ_cbs[0]
            pred_cb = pred_cbs[0]

            if set(d.name for d in succ_cb.dest) != set(succ_nn):
                continue

            pred_rg = self._find_rg_by_cb(pred_cb, routing_groups)
            succ_rg = self._find_rg_by_cb(succ_cb, routing_groups)

            # The expected previous core block will add a new replicated node.
            pred_cb_dest = pred_cb.dest.copy()
            pred_cb_dest.append(node.copy())

            n_core_required_after_copy = len(
                get_neu_segments(
                    pred_cb_dest,
                    pred_cb.n_fanout,
                    pred_cb.n_neuron_repl,
                    _BACKEND_CONTEXT.cflags["grouping_optim_target"],
                )
            )
            pred_rg_n_core = pred_rg.n_core_required
            pred_rg_n_core_after_copy = (
                pred_rg_n_core - pred_cb.n_core_required + n_core_required_after_copy
            )

            n_core_after_split = [0] * ONLY_SUPPORT_N_SUCC
            for i in range(ONLY_SUPPORT_N_SUCC):
                dest = [self._raw_nodes[succ_nn[i]]]
                n_core_after_split[i] = len(
                    get_neu_segments(
                        dest,  # type: ignore
                        succ_cb.n_fanout,
                        succ_cb.n_neuron_repl,
                        _BACKEND_CONTEXT.cflags["grouping_optim_target"],
                    )
                )

            # 2^log2(#N of source rg) + 2^log2(#N of dest rg)
            n_core_before = _roundup_to_pow2(pred_rg_n_core) + _roundup_to_pow2(
                succ_rg.n_core_required
            )
            # 2^log2(#N of source rg after copy) + sum(2^log2(#N of dest rg[i]))
            n_core_after = _roundup_to_pow2(pred_rg_n_core_after_copy) + sum(
                _roundup_to_pow2(n) for n in n_core_after_split
            )

            # TODO actually here is: n_core_after < n_core_before
            if True:
                if not is_optimized:
                    is_optimized = True

                self._copy_node(node, keep_pred_conn=True, grab_succ_nodes=succ_nn[-1])

        return is_optimized

    def _copy_node(
        self,
        node: NodeType,
        *,
        keep_pred_conn: bool = False,
        keep_succ_conn: bool = False,
        grab_pred_nodes: Union[NodeName, Sequence[NodeName]] = (),
        grab_succ_nodes: Union[NodeName, Sequence[NodeName]] = (),
        update: bool = True,
    ) -> NodeType:
        def _copy_pred_conn(
            copied: DestNodeType, pred_nodes: dict[NodeName, EdgeAttr], orig_ind: int
        ) -> None:
            if copied.name not in self.pred_dg.keys():
                self.pred_dg[copied.name] = dict()
            for pred_nn, pred_edge_attr in pred_nodes.items():
                copied_edge = pred_edge_attr.edge.copy(target=copied)
                self._raw_edges[copied_edge.name] = copied_edge
                # If don't _update_graph(), update partial information:
                # 1. Add the copied node & its incomming edges to succ_dg.
                # 2. Update the in-degree of copied node = in-degree of the original node.
                # 3. The out-degree of predecessors +1.
                # 1/2 -> A -> ...
                # ---
                # 1/2 -> A -> ...
                # 1/2 -> A'-> ...
                if not update:
                    copied_edge_attr = EdgeAttr(copied_edge, pred_edge_attr.distance)
                    self.succ_dg[pred_nn][copied.name] = copied_edge_attr
                    self.pred_dg[copied.name][pred_nn] = copied_edge_attr
                    self.degree_of_nodes[pred_nn].out_degree += 1

            if not update:
                self.degree_of_nodes[copied.name].in_degree = orig_ind

        def _copy_succ_conn(
            copied: DestNodeType, succ_nodes: dict[NodeName, EdgeAttr], orig_oud: int
        ) -> None:
            for succ_nn, succ_edge_attr in succ_nodes.items():
                copied_edge = succ_edge_attr.edge.copy(source=copied)
                self._raw_edges[copied_edge.name] = copied_edge
                # If don't _update_graph(), update partial information:
                # 1. Add the copied node & its outcoming edges to pred_nodes_dict & pred_dg.
                # 2. Update the out-degree of copied node = out-degree of the original node.
                # 3. The in-degree of successors +1.
                # ... -> A -> 1/2
                # ---
                # ... -> A -> 1/2
                # ... -> A'-> 1/2
                if not update:
                    copied_edge_attr = EdgeAttr(copied_edge, succ_edge_attr.distance)
                    self.pred_dg[succ_nn][copied.name] = copied_edge_attr
                    self.succ_dg[copied.name] = {succ_nn: copied_edge_attr}
                    self.degree_of_nodes[succ_nn].in_degree += 1

            if not update:
                self.degree_of_nodes[copied.name].out_degree = orig_oud

        pred_nodes = self.pred_dg[node.name]
        succ_nodes = self.succ_dg[node.name]

        copied = node.copy()
        self._raw_nodes[copied.name] = copied

        if not update:
            self.degree_of_nodes[copied.name] = NodeDegree()

        if keep_pred_conn:
            orig_ind = self.degree_of_nodes[node.name].in_degree
            _copy_pred_conn(copied, pred_nodes, orig_ind)
        else:
            if isinstance(grab_pred_nodes, NodeName):
                grab_pred_nodes = (grab_pred_nodes,)

            if any(nn not in pred_nodes for nn in grab_pred_nodes):
                raise ValueError(
                    f"not all nodes in 'grab_pred_nodes' are in node {node.name}'s predecessors. "
                    f"Got {', '.join(grab_pred_nodes)}, but predecessors are {', '.join(pred_nodes)}."
                )

            if copied.name not in self.pred_dg.keys():
                self.pred_dg[copied.name] = dict()

            for pred_nn in grab_pred_nodes:
                pred_edge = pred_nodes[pred_nn].edge
                pred_edge.target = copied
                # If don't _update_graph(), update partial information:
                # 1. Remove the original connection & add the copied node & the edge
                # with the modified target to succ_nodes_dict & succ_dg.
                # 2. Update the in-degree of copied node = len(grab).
                # 3. The out-degree of predecessors keep the same.
                # 1/2/3 -> A -> ...
                # ---
                # 1     -> A -> ...
                # 2/3   -> A'-> ...
                if not update:
                    _orig_edge_attr = self.succ_dg[pred_nn].pop(node.name)
                    new_edge_attr = EdgeAttr(pred_edge, _orig_edge_attr.distance)

                    self.succ_dg[pred_nn][copied.name] = new_edge_attr
                    self.pred_dg[copied.name][pred_nn] = new_edge_attr

            if not update:
                self.degree_of_nodes[node.name].in_degree -= len(grab_pred_nodes)
                self.degree_of_nodes[copied.name].in_degree = len(grab_pred_nodes)

        if keep_succ_conn:
            orig_oud = self.degree_of_nodes[node.name].out_degree
            _copy_succ_conn(copied, pred_nodes, orig_oud)
        else:
            if isinstance(grab_succ_nodes, NodeName):
                grab_succ_nodes = (grab_succ_nodes,)

            if any(nn not in succ_nodes for nn in grab_succ_nodes):
                raise ValueError(
                    f"not all nodes in 'grab_succ_nodes' are in node {node.name}'s successors."
                    f"Got {', '.join(grab_succ_nodes)}, but successors are {', '.join(succ_nodes)}."
                )

            for succ_nn in grab_succ_nodes:
                succ_edge = succ_nodes[succ_nn].edge
                succ_edge.source = copied
                # If don't _update_graph(), update partial information:
                # 1. Remove the original connection & add the copied node & the edge
                # with the modified target to succ_dg.
                # 2. Update the out-degree of copied node = len(grab).
                # 3. The in-degree of successors keep the same.
                # ... -> A -> 1/2/3
                # ---
                # ... -> A -> 1
                # ... -> A'-> 2/3
                if not update:
                    self.succ_dg[node.name].pop(succ_nn)
                    _orig_edge_attr = self.pred_dg[succ_nn].pop(node.name)
                    new_edge_attr = EdgeAttr(succ_edge, _orig_edge_attr.distance)
                    self.succ_dg[copied.name] = {succ_nn: new_edge_attr}
                    self.pred_dg[succ_nn][copied.name] = new_edge_attr
                    # self.pred_dg = reverse_edges2(self.succ_dg)

            if not update:
                self.degree_of_nodes[node.name].out_degree -= len(grab_succ_nodes)
                self.degree_of_nodes[copied.name].out_degree = len(grab_succ_nodes)

        if update:
            self._update_graph()

        return copied

    @staticmethod
    def _find_rg_by_cb(
        cb: CoreBlock, routing_groups: list[RoutingGroup]
    ) -> RoutingGroup:
        """Find which routing group the target core block is in."""
        _rgs = [rg for rg in routing_groups if cb in rg.core_blocks]

        if len(_rgs) != 1:
            raise GraphConnectionError(
                f"the core block can only be assigned to 1 routing group, but got {len(_rgs)}."
            )

        return routing_groups[0]

    def get_global_t_1st_vld(self) -> int:
        """Return the timestamp when the compiled network outputs the first valid data from the view of external.

        NOTE: If there are more than one output nodes, the first valid data will be the one with the smallest timestamp.
        """
        self.build_check()
        return min(
            n._oflow_format.get_global_t_1st_vld(n.tick_wait_start)
            for n in self.onodes.values()
        )

    def get_output_flow_format(self) -> dict[NodeName, DataFlowFormat]:
        """Return the output data flow format of the compiled in global time.

        NOTE: There may be multiple different data streams in the network.
        """
        self.build_check()
        return {
            n.name: n._oflow_format.local2global(n.tick_wait_start)
            for n in self.onodes.values()
        }

    @property
    def graph_name_repr(self) -> str:
        _prefix = "graph_of_"
        return _prefix + "_and_".join(network.name for network in self._raw_networks)
