from typing import Optional

import pytest
from paicorelib import HwConfig

import paibox as pb
from paibox.backend.graph_utils import *
from paibox.backend.types import *
from paibox.components import Neuron
from paibox.exceptions import GraphBuildError, GraphConnectionError, NotSupportedError


class TestPAIGraph:
    def test_build_duplicated_networks(self, build_example_net2):
        net = build_example_net2
        mapper = pb.Mapper()

        with pytest.raises(GraphBuildError):
            mapper.build(net, net, net)

    def test_output_nodes_with_more_than_1152(self, monkeypatch, build_example_net2):
        net = build_example_net2

        # Change the #N of neurons of the output node
        assert 1200 > HwConfig.N_FANIN_PER_DENDRITE_MAX
        monkeypatch.setattr(net.n2, "_n_neuron", 1200)

        mapper = pb.Mapper()
        mapper.build(net)

        with pytest.raises(NotSupportedError):
            mapper.compile()

    def test_prebuild_topo_info(self, build_FModule_ConnWithInput_Net):
        net = build_FModule_ConnWithInput_Net
        mapper = pb.Mapper()
        mapper.build(net)

        assert len(mapper.graph._raw_nodes) == 5
        assert len(mapper.graph._raw_edges) == 4

    def test_prebuild_gh_build_ignore(
        self, monkeypatch, build_FModule_ConnWithInput_Net
    ):
        net = build_FModule_ConnWithInput_Net
        mapper = pb.Mapper()

        monkeypatch.setattr(net.n1, "__gh_build_ignore__", True)

        with pytest.raises(GraphConnectionError):
            mapper.build(net)

        monkeypatch.setattr(net.n1, "__gh_build_ignore__", False)
        monkeypatch.setattr(net.s2, "__gh_build_ignore__", True)
        monkeypatch.setattr(net.n2, "__gh_build_ignore__", True)

        mapper.build(net)
        assert net.s2.name not in mapper.graph._raw_edges
        assert net.n2.name not in mapper.graph._raw_nodes

    @pytest.mark.parametrize("no_twisted_branch", [True, False])
    def test_untwist_branch_nodes1(
        self, ensure_dump_dir, build_Network_branch_nodes, no_twisted_branch
    ):
        net: pb.Network = build_Network_branch_nodes

        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile(no_twisted_branch=no_twisted_branch)
        mapper.export(fp=ensure_dump_dir)

        if no_twisted_branch:
            assert (
                len(mapper.graph.nodes)
                == len(net.nodes(level=1).include(Neuron, pb.InputProj)) + net.n_copy
            )
        else:
            assert len(mapper.graph.nodes) == len(
                net.nodes(level=1).include(Neuron, pb.InputProj)
            )


class TestGroupEdges:
    @staticmethod
    def graph_partition_proto(
        succ_edges: dict[NodeName, dict[NodeName, EdgeName]],
        degree: dict[NodeName, NodeDegree],
        *,
        ordered_nodes: Optional[list[NodeName]] = None,
    ) -> list[set[EdgeName]]:
        gh_parts = []
        rgid = 0
        seen_nodes: set[NodeName] = set()

        pred_dg = get_pred_dg_by_succ_dg(succ_edges)

        if isinstance(ordered_nodes, list):
            # In topological sorting.
            ordered = ordered_nodes
        else:
            # Without sorting.
            ordered = list(succ_edges.keys())

        for node in ordered:
            if node in seen_nodes:
                continue

            if degree[node].out_degree == 0:
                seen_nodes.add(node)
                continue

            succ_nodes: set[NodeName] = set()
            other_involved_nodes: set[NodeName] = set()
            succ_nodes_candid: set[NodeName] = set(succ_edges[node].keys())
            partitioned_nodes = set([node])

            while len(succ_nodes_candid) > 0:
                succ_nodes.update(succ_nodes_candid)

                for candid in succ_nodes_candid:
                    if degree[candid].in_degree > 1:
                        coming_nodes = set(pred_dg[candid].keys()) - seen_nodes
                        other_involved_nodes |= coming_nodes

                other_involved_nodes -= partitioned_nodes
                partitioned_nodes |= other_involved_nodes
                succ_nodes_candid.clear()

                for other_node in other_involved_nodes:
                    other_candid = set(succ_edges[other_node].keys()) - succ_nodes
                    succ_nodes_candid |= other_candid

            seen_nodes |= partitioned_nodes

            succ_edges_set: set[EdgeName] = set()
            succ_nodes_set: set[NodeName] = set()

            for _node in partitioned_nodes:
                succ_edges_set.update(e for e in succ_edges[_node].values())
                succ_nodes_set.update(n for n in succ_edges[_node])

            gh_parts.append(succ_edges_set)

            rgid += 1

        return gh_parts

    @pytest.mark.parametrize(
        "succ_edges, expectation",
        [
            (
                {
                    "inp1": {"n1": "s1"},
                    "n1": {"n2": "s2", "n3": "s3"},
                    "n2": {"n3": "s4"},
                    "n3": {"n4": "s5"},
                    "n4": {},
                },
                [{"s1"}, {"s2", "s3", "s4"}, {"s5"}],
            ),
            (
                {
                    "inp1": {"n1": "s1"},
                    "n1": {"n2": "s2", "n3": "s3"},
                    "n2": {"n4": "s4"},
                    "n3": {"n4": "s5"},
                    "n4": {},
                },
                [{"s1"}, {"s2", "s3"}, {"s4", "s5"}],
            ),
            (
                {
                    "inp1": {"n1": "s1"},
                    "inp2": {"n4": "s2"},
                    "n1": {"n2": "s3"},
                    "n2": {"n3": "s4"},
                    "n3": {},
                    "n4": {"n5": "s5"},
                    "n5": {"n3": "s6"},
                },
                [{"s1"}, {"s2"}, {"s3"}, {"s5"}, {"s4", "s6"}],
            ),
            (
                {
                    "n1": {"n2": "s1", "n3": "s2"},
                    "n2": {"n4": "s3"},
                    "n3": {"n4": "s4"},
                    "n4": {"n5": "s5", "n6": "s6"},
                    "n5": {},
                    "n6": {},
                    "n7": {"n6": "s7"},
                },
                [{"s1", "s2"}, {"s3", "s4"}, {"s5", "s6", "s7"}],
            ),
        ],
    )
    def test_graph_partition(self, succ_edges, expectation):
        """
        Test #1:
            INP1 -> N1 -> N2 -> N3 -> N4
                       ------->

        Test #2:
            INP1 -> N1 -> N2 -> N4
                       -> N3 ->

        Test #3:
            INP1 -> N1 -> N2 -> N3
            INP2 -> N4 -> N5 ->

        Test #4:
            N1 -> N2 -> N4 -> N5
               -> N3 ->    -> N6
                        N7 ->
        """
        degrees = get_node_degrees(succ_edges)
        ordered_nodes = toposort(succ_edges)
        partitioned_edges = self.graph_partition_proto(
            succ_edges, degrees, ordered_nodes=ordered_nodes
        )

        # without considering the order in the list
        assert set(frozenset(e) for e in partitioned_edges) == set(
            frozenset(e) for e in expectation
        )

    @pytest.mark.parametrize(
        "succ_edges, expected",
        [
            (
                {
                    "inp1": {"n1": "s1"},
                    "n1": {"n2": "s2"},
                    "n2": {"n1": "s3", "n3": "s4"},
                    "n3": {},
                },
                {
                    "inp1": (0, 1),
                    "n1": (2, 1),
                    "n2": (1, 2),
                    "n3": (1, 0),
                },
            ),
            (
                {
                    "inp1": {"n1": "s1"},
                    "n1": {"n2": "s2", "n3": "s3"},
                    "n2": {"n3": "s4"},
                    "n3": {"n4": "s5", "n5": "s6"},
                    "n4": {"n6": "s7"},
                    "n5": {"n6": "s8"},
                    "n6": {},
                },
                {
                    "inp1": (0, 1),
                    "n1": (1, 2),
                    "n2": (1, 1),
                    "n3": (2, 2),
                    "n4": (1, 1),
                    "n5": (1, 1),
                    "n6": (2, 0),
                },
            ),
            (
                {
                    "n1": {"n3": "s1", "n4": "s2"},
                    "n2": {"n4": "s3", "n5": "s4"},
                    "n3": {},
                    "n4": {},
                    "n5": {},
                },
                {
                    "n1": (0, 2),
                    "n2": (0, 2),
                    "n3": (1, 0),
                    "n4": (2, 0),
                    "n5": (1, 0),
                },
            ),
        ],
    )
    def test_get_node_degrees(self, succ_edges, expected):
        degrees = get_node_degrees(succ_edges)

        for k, d in expected.items():
            assert degrees[k].in_degree == d[0]
            assert degrees[k].out_degree == d[1]

    def test_group_edges_with_constrs(
        self, monkeypatch, build_network_with_branches_4bit
    ):
        net = build_network_with_branches_4bit

        mapper = pb.Mapper()
        mapper.clear()
        mapper.build(net)
        mapper.compile(no_twisted_branch=False)

        # In this case, N2 & N3 should be together.
        pos_n2 = pos_n3 = 0
        for i, cb in enumerate(mapper.core_blocks):
            _g_with_name = [e.target.name for e in cb.obj]
            if "s2" in _g_with_name:
                pos_n2 = i
                break

        for i, cb in enumerate(mapper.core_blocks):
            _g_with_name = [e.target.name for e in cb.obj]
            if "s3" in _g_with_name:
                pos_n3 = i
                break

        assert pos_n2 == pos_n3

        # In this case, N2 & N3 should be split.
        monkeypatch.setattr(net.n2, "_tws", 2)
        monkeypatch.setattr(net.n3, "_tws", 3)

        mapper.clear()
        mapper.build(net)
        mapper.compile(no_twisted_branch=False)

        pos_n2 = pos_n3 = 0
        for i, part in enumerate(mapper.core_blocks):
            _g_with_name = [e.target.name for e in part.obj]
            if "s2" in _g_with_name:
                pos_n2 = i
                break

        for i, part in enumerate(mapper.core_blocks):
            _g_with_name = [e.target.name for e in part.obj]
            if "s3" in _g_with_name:
                pos_n3 = i
                break

        assert pos_n2 != pos_n3
