from collections import defaultdict
from collections.abc import Sequence
from typing import Optional

import pytest
from paicorelib import HwConfig

import paibox as pb
from paibox.backend.graphs import get_node_degrees, get_pred_dg_by_succ_dg, toposort
from paibox.backend.graphs_types import *
from paibox.components import Neuron
from paibox.exceptions import GraphBuildError, GraphConnectionError, NotSupportedError

from .conftest import TestData


class TestTopoSort:
    @pytest.mark.parametrize(
        TestData.toposort_data["args"],
        TestData.toposort_data["data"],
        ids=TestData.toposort_data["ids"],  # type:ignore
    )
    def test_toposort(self, nodes):
        """
        Test #1: one input 1
            INP1 -> N1 -> N2 -> N3
            N1 -> N4
            N4 -> N2

        Test #2: one input 2
            INP1 -> N1 -> N2 -> N3 -> N4
            N1 -> N5 -> N6 -> N7 -> N4
            N5 -> N3
            N3 -> N6

        Test #3: more than one input
            INP1 -> N1 -> N2 -> N3
            INP2 -> N4 -> N5 -> N3
            N2 -> N4

        Test #4: one input 3
            INP1 -> N1 -> N2 -> N4
            N1 -> N3 -> N4

        Test #5: headless neuron 1
            INP1 -> N1 -> N2 -> N4
            N3 -> N2
        """
        ordered = toposort(nodes)
        assert len(ordered) == len(nodes)

    @pytest.mark.parametrize(
        "edges",
        [
            {
                "inp1": {"n1"},
                "n1": {"n2"},
                "n2": {"n3", "n4"},
                "n3": {},
                "n4": {"n1"},
            },
            {"inp1": {"n1"}, "n1": {"n2"}, "n2": {"n3"}, "n3": {"n2", "n4"}, "n4": {}},
            {"inp1": {"n1"}, "inp2": {"n2"}, "n1": {"n3"}, "n2": {"n1"}, "n3": {"n2"}},
            {
                "inp1": {"n1"},
                "inp2": {"n2"},
                "inp3": {"n2"},
                "n1": {"n3"},
                "n2": {"n1"},
                "n3": {"n4"},
                "n4": {"n2"},
            },
        ],
    )
    def test_toposort_has_cycle(self, edges):
        """
        Test #1: 1 input
        INP1 -> N1 -> N2 -> N3
        N4 -> N1
        N2 -> N4

        Test #2: 1 input
        INP1 -> N1 -> N2 -> N3 -> N4
        N3 -> N2

        Test #3: 2 inputs
        INP1 -> N1 -> N3 -> N2
        INP2 -> N2

        Test #4: 3 inputs
        INP1 -> N1 -> N3 -> N4 -> N2
        INP2 -> N2 -> N1
        INP3 -> N2
        """
        with pytest.raises(NotSupportedError):
            ordered = toposort(edges)


@pytest.mark.skip(reason="Not implemented")
@pytest.mark.parametrize(
    "constrs, expected",
    [
        ([{"1", "2", "3"}, {"4", "5"}], ["1", "2", "3"]),
        ([{"1", "2"}, {"3", "4"}, {"5", "6"}], ["1", "2"]),
    ],
)
def test_bounded_by(constrs, expected):
    def _bounded_by_proto(node: str, constrs: Sequence[Sequence[str]]) -> list[str]:
        for constr in constrs:
            for bounded_node in constr:
                if node == bounded_node:
                    return list(set(constr))

        return []

    result = _bounded_by_proto("1", constrs)

    assert set(result) == set(expected)


@pytest.mark.skip(reason="Not implemented")
@pytest.mark.parametrize(
    "constrs, expected",
    [
        ({"1": {"2", "3", "4"}, "2": {"3", "4"}}, ["2", "3", "4"]),
        ({"1": {"2", "3"}, "4": {"1"}}, ["2", "3", "4"]),
        ({"2": {"1", "3"}, "4": {"1"}}, ["2", "4"]),
    ],
)
def test_conflicted_by(constrs, expected):
    def _conflicted_by_proto(node: str, constrs: dict[str, Sequence[str]]) -> list[str]:
        c = set(constrs.get(node, []))

        for k, v in constrs.items():
            for conf_node in v:
                if node == conf_node:
                    c.add(k)

        return list(c)

    result = _conflicted_by_proto("1", constrs)

    assert set(result) == set(expected)


@pytest.mark.skip(reason="Not implemented")
@pytest.mark.parametrize(
    "constrs, expected",
    [
        (
            [{"1", "2", "3"}, {"4", "5"}, {"6", "7", "1"}, {"4", "8"}],
            [frozenset(["1", "2", "3", "6", "7"]), frozenset(["4", "5", "8"])],
        ),
        (
            [{"1", "2", "3"}, {"4", "5"}, {"6", "7"}, {"4", "8"}],
            [
                frozenset(["1", "2", "3"]),
                frozenset(["6", "7"]),
                frozenset(["4", "5", "8"]),
            ],
        ),
        (
            [{"1", "2", "3"}, {"4", "5"}, {"6", "7"}, {"4", "1", "8"}],
            [
                frozenset(["1", "2", "3", "4", "5", "8"]),
                frozenset(["6", "7"]),
            ],
        ),
    ],
)
def test_bounded_nodes_check(constrs, expected):
    def _bounded_nodes_check_proto(
        constrs: list[Sequence[str]],
    ) -> list[frozenset[str]]:
        seen = {}
        need_update_nodes = []

        for bounded_set in constrs:
            for node in bounded_set:
                if node in seen:
                    need_update_nodes.append(node)
                    seen[node].update(bounded_set)
                else:
                    seen[node] = bounded_set

        _constr = []

        for bounded in constrs:
            for node in need_update_nodes:
                flag = True
                if node in bounded:
                    flag = False
                    break

            if flag:
                # Unique set
                _constr.append(frozenset(bounded))

        while need_update_nodes:
            for node in need_update_nodes:
                _constr.append(frozenset(seen[node]))
                need_update_nodes.remove(node)

        return _constr

    result = _bounded_nodes_check_proto(constrs)

    # Inconvenient to compare the elements in the list of sets.
    assert len(result) == len(expected)


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

        monkeypatch.setattr(net.n1, "__gh_build_ignore__", True)

        mapper = pb.Mapper()

        with pytest.raises(GraphConnectionError):
            mapper.build(net)

    def test_untwist_branch_nodes1(self, ensure_dump_dir, build_Network_branch_nodes):
        net: pb.Network = build_Network_branch_nodes

        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir)

        assert (
            len(mapper.graph.nodes)
            == len(net.get_components(level=1).include(Neuron, pb.InputProj))
            + net.n_copy
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
        "succ_edges",
        [
            {
                "inp1": {"n1": "s1"},
                "n1": {"n2": "s2"},
                "n2": {"n1": "s3", "n3": "s4"},
                "n3": {},
            },
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
                "n1": {"n3": "s1", "n4": "s2"},
                "n2": {"n4": "s3", "n5": "s4"},
                "n3": {},
                "n4": {},
                "n5": {},
            },
        ],
    )
    def test_get_node_degrees(self, succ_edges):
        degrees = get_node_degrees(succ_edges)
        print()

    def test_group_edges_with_constrs(
        self, monkeypatch, build_network_with_branches_4bit
    ):
        net = build_network_with_branches_4bit

        mapper = pb.Mapper()
        mapper.clear()
        mapper.build(net)
        partitioned_edges = mapper.graph.graph_partition()

        # In this case, N2 & N3 should be together.
        pos_n2 = pos_n3 = 0
        for i, part in enumerate(partitioned_edges):
            _g_with_name = [e.name for e in part.edges]
            if "s2" in _g_with_name:
                pos_n2 = i
            if "s3" in _g_with_name:
                pos_n3 = i

        assert pos_n2 == pos_n3
        assert pos_n2 != 0

        # In this case, N2 & N3 should be split.
        monkeypatch.setattr(net.n2, "_tws", 2)
        monkeypatch.setattr(net.n3, "_tws", 3)

        mapper.clear()
        mapper.build(net)
        partitioned_edges = mapper.graph.graph_partition()

        pos_n2 = pos_n3 = 0
        for i, part in enumerate(partitioned_edges):
            _g_with_name = [e.name for e in part.edges]
            if "s2" in _g_with_name:
                pos_n2 = i
            if "s3" in _g_with_name:
                pos_n3 = i

        assert pos_n2 != pos_n3


class TestDAGPathDistance:
    @staticmethod
    def get_longest_path_proto(
        edges_with_d: dict[NodeName, dict[NodeName, int]], ordered_nodes: list[NodeName]
    ) -> tuple[list[NodeName], int]:
        """Get the longest path in the DAG.

        Args:
            - edges_with_d: a list of directed edges with distance.
            - ordered_nodes: nodes in topological sorting.

        Return: the longest distance in the graph.
        """
        distances: dict[NodeName, int] = {node: 0 for node in ordered_nodes}
        pred_nodes: dict[NodeName, NodeName] = defaultdict()

        for node in ordered_nodes:
            for neighbor in edges_with_d[node]:
                d = edges_with_d[node][neighbor]
                if distances[node] + d > distances[neighbor]:
                    distances[neighbor] = distances[node] + d
                    pred_nodes[neighbor] = node

        # When there are more than one output nodes
        # with same distance, choose the first one.
        node = max(
            filter(lambda node: len(edges_with_d[node]) == 0, distances),
            key=lambda node: distances.get(node, 0),
        )

        distance = distances[node]
        path = [node]

        while path[-1] in pred_nodes:
            path.append(pred_nodes[path[-1]])

        path.reverse()
        return path, distance

    @pytest.mark.parametrize(
        TestData.get_longest_path_data["args"],
        TestData.get_longest_path_data["data"],
        ids=TestData.get_longest_path_data["ids"],  # type:ignore
    )
    def test_get_longest_path_proto(self, edges, expected_path, expected_distance):
        ordered = toposort(edges)
        path, distance = self.get_longest_path_proto(edges, ordered)

        assert path == expected_path
        assert distance == expected_distance

    @staticmethod
    def get_shortest_path_proto(
        edges_with_d: dict[NodeName, dict[NodeName, int]],
        ordered_nodes: list[NodeName],
        input_nodes: list[NodeName],
    ) -> tuple[list[NodeName], int]:
        """Get the shortest path in the DAG.

        Args:
            - edges_with_d: a list of directed edges with distance.
            - ordered_nodes: nodes in topological sorting.
            - input_nodes: input nodes.

        Return: the shortest distance in the graph.
        """
        distances: dict[NodeName, int] = defaultdict(lambda: 999)
        pred_nodes: dict[NodeName, NodeName] = defaultdict()

        # set initial value for all inputs nodes.
        if input_nodes:
            for inode in input_nodes:
                distances[inode] = 0
        else:
            distances[ordered_nodes[0]] = 0

        for node in ordered_nodes:
            for neighbor in edges_with_d[node]:
                d = edges_with_d[node][neighbor]
                if distances[node] + d < distances[neighbor]:
                    distances[neighbor] = distances[node] + d
                    pred_nodes[neighbor] = node

        # When there are more than one output nodes
        # with same distance, choose the first one.
        node = min(
            filter(lambda node: len(edges_with_d[node]) == 0, distances),
            key=lambda node: distances.get(node, 0),
        )

        distance = distances[node]
        path = [node]

        while path[-1] in pred_nodes:
            path.append(pred_nodes[path[-1]])

        path.reverse()
        return path, distance

    @pytest.mark.parametrize(
        TestData.get_shortest_path_data["args"],
        TestData.get_shortest_path_data["data"],
        ids=TestData.get_shortest_path_data["ids"],  # type:ignore
    )
    def test_get_shortest_path_proto(
        self, edges, inodes, expected_path, expected_distance
    ):
        ordered = toposort(edges)
        path, dist = self.get_shortest_path_proto(edges, ordered, inodes)

        assert path == expected_path
        assert dist == expected_distance
