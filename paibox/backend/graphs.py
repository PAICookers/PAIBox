import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from paibox.base import DataFlowFormat
from paibox.collector import Collector
from paibox.components import (
    FullConnectedSyn,
    InputProj,
    NeuModule,
    Neuron,
    OnlineNeuron,
)
from paibox.components.functional import LinearSemiFolded
from paibox.exceptions import (
    GraphBuildError,
    GraphConnectionError,
    GraphNotSupportedError,
    PAIBoxWarning,
)
from paibox.network import DynSysGroup
from paibox.utils import check_elem_unique

from .graph_utils import (
    get_node_degrees,
    get_pred_dg_by_succ_dg,
    iter_toposort,
    prune_disconn_graph,
    reverse_edges,
    toposort,
)
from .group import BaseGroup, DataGroup, InhiGroup, MergedGroup
from .placement import CoreBlock
from .routing import RoutingGroup
from .types import (
    DestNodeType,
    EdgeAttr,
    EdgeName,
    EdgeType,
    NodeDegree,
    NodeName,
    NodeType,
    SourceNodeType,
)

__all__ = ["PAIGraph"]


NodeAdjDictType = dict[NodeName, dict[NodeName, EdgeAttr]]


@dataclass
class PAIGraph:
    """Directed graph of PAIBox. We treat networks as one whole graph. In the graph, synapses are   \
        edges and neurons are nodes.
    """

    target_networks: tuple[DynSysGroup, ...] = field(default_factory=tuple)
    """The networks that are actually used in the graph."""

    nodes: Collector[NodeName, NodeType] = field(default_factory=Collector)
    """Valid nodes in the graph."""
    edges: Collector[EdgeName, EdgeType] = field(default_factory=Collector)
    """Valid edges in the graph."""

    inodes: Collector[NodeName, SourceNodeType] = field(default_factory=Collector)
    """Valid input nodes in the graph. Can be recalculated."""
    onodes: Collector[NodeName, DestNodeType] = field(default_factory=Collector)
    """Valid output nodes in the graph. Can be recalculated."""

    succ_dg: NodeAdjDictType = field(default_factory=dict)
    """Valid successor edges & nodes of every node in the graph. Can be recalculated."""
    pred_dg: NodeAdjDictType = field(default_factory=dict)
    """Valid predecessor edges & nodes of every node in the graph. Can be recalculated."""

    degree_of_nodes: dict[NodeName, NodeDegree] = field(default_factory=dict)
    """A dictionary of in/out-degree tuple of nodes. Can be recalculated"""

    has_built: bool = field(default=False)

    _build_options: dict[str, Any] = field(default_factory=dict)
    """Building options."""

    def clear(self) -> None:
        """Clear the PAIGraph."""
        self.has_built = False
        self._build_options.clear()

        self.target_networks = ()
        self.nodes.clear()
        self.edges.clear()
        self.inodes.clear()
        self.onodes.clear()
        self.succ_dg.clear()
        self.pred_dg.clear()
        self.degree_of_nodes.clear()

    def build(self, *networks: DynSysGroup, **build_options) -> None:
        # Check the hardware resource limits of operators in the network during the build phase.
        # self._build_options.setdefault("check_before_compile", True)

        # Prune networks with no input nodes before building the graph.
        # self._build_options.setdefault("ignore_no_inp_subgraph", True)

        # Update building options provided by the user.
        self._build_options.update(build_options)

        self.clear()

        if not networks:
            raise GraphBuildError("no networks are provided.")

        if not check_elem_unique(networks):
            raise GraphBuildError("duplicated networks are not allowed.")

        # The networks will be modified in pre-build phase.
        self._pre_build(networks, **self._build_options)

        # Collect the raw information of the graph with the given networks.
        # TODO For debugging, disable the check. Add an env variable to distinguish between debug & production use.
        self.target_networks = networks
        # self._filter_out_no_inp_networks(networks)

        _nodes: Collector[NodeName, NodeType] = Collector()
        _edges: Collector[EdgeName, EdgeType] = Collector()

        for nw in self.target_networks:
            _nodes += nw.nodes().include(InputProj, Neuron).exclude(NeuModule).unique()
            _edges += nw.nodes().subset(FullConnectedSyn).unique()

        raw_nodes = _nodes.val_on_condition(lambda node: not node.__gh_build_ignore__)
        raw_edges = _edges.val_on_condition(lambda edge: not edge.__gh_build_ignore__)
        raw_succ_dg = self._build_succ_dg(raw_nodes, list(raw_edges.values()))

        # `InputProj` nodes are input nodes definitely.
        self.inodes = raw_nodes.subset(InputProj)

        # Filter out the subgraphs that are not connected to the input nodes.
        if build_options.get("ignore_no_inp_subgraph", False):
            pruned_succ_dg, _ = prune_disconn_graph(
                raw_succ_dg, list(self.inodes.keys())
            )
            # Remove the disconnected nodes then get the valid nodes.
            nodes = raw_nodes.key_on_condition(lambda node: node in pruned_succ_dg)
        else:
            pruned_succ_dg = raw_succ_dg
            nodes = raw_nodes

        # Finally, collect the valid graph information.
        self.nodes = nodes
        self.edges = Collector(
            {
                edge.name: edge
                for edge in raw_edges.values()
                if edge.source.name in self.nodes and edge.target.name in self.nodes
            }
        )
        self.succ_dg = cast(NodeAdjDictType, pruned_succ_dg)
        self.pred_dg = get_pred_dg_by_succ_dg(self.succ_dg)
        self.degree_of_nodes = get_node_degrees(self.succ_dg)
        self.onodes = self._collect_onodes(self.nodes, self.degree_of_nodes)

        # Check the uniqueness of the successors of each node.
        for n in self.succ_dg:
            assert check_elem_unique(self.succ_dg[n])

        self.has_built = True

    def _pre_build(self, networks: tuple[DynSysGroup, ...], **build_options) -> None:
        """Preprocessing before obtaining the topology."""
        # Build functional modules for each network.
        for nw in networks:
            if nw.is_composed_of_semi_folded_ops():
                modules = nw.components.subset(NeuModule)
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
                nw.build_modules(pred_dg_semi_ops, ordered_nodes, **build_options)
            else:
                nw.build_modules(**build_options)

    @staticmethod
    def _filter_out_no_inp_networks(
        networks: tuple[DynSysGroup, ...],
    ) -> tuple[DynSysGroup, ...]:
        """Filter out the networks that have no input nodes."""
        target = []

        for nw in networks:
            if len(nw.nodes().subset(InputProj)) > 0:
                target.append(nw)
            else:
                warnings.warn(
                    f"Network {nw.name} has no input node, filtered out.",
                    PAIBoxWarning,
                )

        return tuple(target)

    @staticmethod
    def _build_succ_dg(
        nodes: Iterable[NodeName], edges: Iterable[EdgeType]
    ) -> NodeAdjDictType:
        succ_dg: NodeAdjDictType = {n: dict() for n in nodes}  # record all nodes

        for edge in edges:
            u, v = edge.source.name, edge.target.name

            if u not in nodes:
                raise GraphConnectionError(
                    f"the source neuron {u} of {edge.name} is not included in the graph."
                )

            if v not in nodes:
                raise GraphConnectionError(
                    f"the target neuron {v} of {edge.name} is not included in the graph."
                )

            succ_dg[u][v] = EdgeAttr(edge, edge.source.delay_relative)

        for n in succ_dg:
            assert check_elem_unique(succ_dg[n])

        return succ_dg

    @staticmethod
    def _collect_onodes(
        nodes: Collector[NodeName, NodeType], degrees: dict[NodeName, NodeDegree]
    ) -> Collector[str, DestNodeType]:
        return Collector(
            {
                k: cast(DestNodeType, v)
                for k, v in nodes.items()
                if degrees[k].out_degree == 0
            }
        ).not_subset(
            InputProj
        )  # Exclude isolated input nodes

    def _update_graph(self, **build_options) -> None:
        """Called after the computation graph(`nodes` or `edges`) has been modified."""
        self.inodes = self.nodes.subset(InputProj)
        self.succ_dg = self._build_succ_dg(self.nodes, list(self.edges.values()))
        self.pred_dg = get_pred_dg_by_succ_dg(self.succ_dg)
        self.degree_of_nodes = get_node_degrees(self.succ_dg)
        self.onodes = self._collect_onodes(self.nodes, self.degree_of_nodes)

    def untwist_branch_nodes(self) -> None:
        # FIXME Input nodes may need to be excluded from the nodes to be traversed?
        ordered_nodes = toposort(self.succ_dg)
        for node_nn in filter(
            lambda node: self.degree_of_nodes[node].out_degree > 1,
            reversed(ordered_nodes),
        ):
            # succ_dg will be updated in _copy_node, so use the copy of succ_dg.
            for succ_nn in self.succ_dg[node_nn].copy():
                # Checking the out-degree of node_nn every time is necessary.
                # The out-degree of node_nn will be changed after coping.
                if (
                    self.degree_of_nodes[succ_nn].in_degree > 1
                    and self.degree_of_nodes[node_nn].out_degree > 1
                ):
                    node = self.nodes[node_nn]
                    self._copy_node(
                        node, keep_pred_conn=True, grab_succ_nodes=succ_nn, update=False
                    )

        self._update_graph()

    def topo_support_check(self) -> None:
        # _degree_check(self.degree_of_nodes, self.succ_dg)

        # Only support output nodes with <= 1152 neurons so far.
        # if any(
        #     onode.num_out > HwConfig.N_FANIN_PER_DENDRITE_MAX
        #     for onode in self.onodes.values()
        # ):
        #     raise GraphNotSupportedError(
        #         f"only output nodes with no more than {HwConfig.N_FANIN_PER_DENDRITE_MAX} "
        #         f"neurons are supported."
        #     )
        pass

    def build_check(self) -> None:
        if not self.has_built:
            raise GraphBuildError("the graph hasn't been built yet.")

    def graph_partition(self) -> list[MergedGroup]:
        """Graph partition."""
        # Build the `SuccGroup` for each node in the graph.
        grps: list[BaseGroup] = []
        for nn in iter_toposort(self.succ_dg):
            if succ_nodes := self.succ_dg[nn]:
                succ_edges = [e_attr.edge for e_attr in succ_nodes.values()]
                grps.append(DataGroup(succ_edges))

        for node in self.nodes.subset(Neuron).values():
            if node.online:
                node = cast(OnlineNeuron, node)
                grps.append(InhiGroup(list(node.lateral_inhi_target)))

        def dfs(sgrp: BaseGroup, msgrp: MergedGroup) -> None:
            # Union-find sets. If the nodes of two `succ_grps` have intersection, merge them.
            for other_sgrp in grps:
                if other_sgrp not in visited and not set(sgrp.nodes).isdisjoint(
                    other_sgrp.nodes
                ):
                    visited.add(other_sgrp)
                    msgrp.add_group(other_sgrp)
                    dfs(other_sgrp, msgrp)

        # Merge
        merged_grps: list[MergedGroup] = []
        visited: set[BaseGroup] = set()
        for sgrp in grps:
            if sgrp not in visited:
                m = MergedGroup([sgrp])
                visited.add(sgrp)
                dfs(sgrp, m)
                merged_grps.append(m)

        return merged_grps

    def _copy_node(
        self,
        node: NodeType,
        *,
        keep_pred_conn: bool = False,
        keep_succ_conn: bool = False,
        grab_pred_nodes: NodeName | Sequence[NodeName] = (),
        grab_succ_nodes: NodeName | Sequence[NodeName] = (),
        update: bool = True,
    ) -> NodeType:
        def _copy_pred_conn(
            copied: DestNodeType, pred_nodes: dict[NodeName, EdgeAttr], orig_ind: int
        ) -> None:
            if copied.name not in self.pred_dg.keys():
                self.pred_dg[copied.name] = dict()
            for pred_nn, pred_edge_attr in pred_nodes.items():
                copied_edge = pred_edge_attr.edge.copy(target=copied)
                self.edges[copied_edge.name] = copied_edge
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
                self.edges[copied_edge.name] = copied_edge
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
        self.nodes[copied.name] = copied

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

    def get_neu_by_name(self, name: NodeName) -> DestNodeType | None:
        for neu in self.nodes.exclude(InputProj):
            if name == neu:
                return cast(DestNodeType, self.nodes[neu])

        return None

    def get_synapse_by_name(self, name: EdgeName) -> EdgeType | None:
        for syn in self.edges:
            if name == syn:
                return self.edges[syn].edge

        return None

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

        if len(self.onodes) == 0:
            return 0

        return min(
            n._oflow_format.get_global_t_1st_vld(n.tick_wait_start)
            for n in self.onodes.values()
        )

    def get_output_flow_format(self) -> dict[NodeName, DataFlowFormat]:
        """Return the output data flow format of the compiled in global time.

        NOTE: There may be multiple different data streams in the nw.
        """
        self.build_check()

        if len(self.onodes) == 0:
            return {}

        return {
            n.name: n._oflow_format.local2global(n.tick_wait_start)
            for n in self.onodes.values()
        }

    @property
    def graph_name_repr(self) -> str:
        _prefix = "graph_of_"
        return _prefix + "_and_".join(nw.name for nw in self.target_networks)
