from typing import Optional

import pytest

import paibox as pb
from paibox.backend.graphs import *
from paibox.backend.graphs import _degree_check
from paibox.exceptions import NotSupportedError

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

    def test_untwist_branch_nodes1(self, build_Network_branch_nodes1):
        net = build_Network_branch_nodes1

        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert len(mapper.graph.nodes) == 9

    def test_untwist_branch_nodes2(self, build_Network_branch_nodes2):
        net = build_Network_branch_nodes2

        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert 1


class TestGroupEdges:
    @staticmethod
    def group_edges_proto(
        succ_edges: dict[NodeName, dict[NodeName, EdgeName]],
        degree: dict[NodeName, NodeDegree],
        *,
        ordered_nodes: Optional[list[NodeName]] = None,
    ) -> list[set[EdgeName]]:
        """Group all edges according to a certain rule.

        Args:
            - edges: a list of edges.
            - succ_edges: a dictionary recording previous nodes and edges.
            - degree: the in/out-degree of nodes.
            - ordered_nodes: nodes in topological sorting. Optional.

        Returns:
            - A list of set of grouped edges.
        """

        def _find_pred_edges_proto(
            succ_edges: dict[NodeName, dict[NodeName, EdgeName]], target_node: NodeName
        ) -> set[EdgeName]:
            pred = set()

            for succ_node in filter(
                lambda node: target_node in node, succ_edges.values()
            ):
                pred.add(succ_node[target_node])

            return pred

        gathered = []
        seen_edges = set()

        if isinstance(ordered_nodes, list):
            # In topological sorting.
            ordered = ordered_nodes
        else:
            # Without sorting.
            ordered = list(succ_edges.keys())

        for node in ordered:
            if degree[node].in_degree > 1:
                edge_group = _find_pred_edges_proto(succ_edges, node)
                # edge_group2 = edge_group.copy()

                # # Remove edges if it is already traversed
                # for e in edge_group:
                #     if e in seen_edges:
                #         edge_group2.remove(e)
                comming_edges = edge_group.difference(seen_edges)

                seen_edges.update(comming_edges)
                gathered.append(comming_edges)

            if degree[node].out_degree > 1:
                edge_group = set(e for e in succ_edges[node].values())

                if edge_group not in gathered:
                    seen_edges.update(edge_group)
                    gathered.append(edge_group)

            elif degree[node].out_degree > 0:
                succ_node = list(succ_edges[node].keys())[0]
                # Check the in-degree of the only following node.
                if degree[succ_node].in_degree == 1:
                    gathered.append({succ_edges[node][succ_node]})
            else:
                # out-degree = 0, do nothing.
                continue

        return gathered

    @pytest.mark.parametrize(
        "succ_edges",
        [
            # This structure is filtered.
            # (
            #     {
            #         "inp1": {"n1": "s1"},
            #         "n1": {"n2": "s2", "n4": "s3"},
            #         "n2": {"n3": "s4"},
            #         "n3": {},
            #         "n4": {"n2": "s5"},
            #     },
            # ),
            {
                "inp1": {"n1": "s1"},
                "n1": {"n2": "s2", "n3": "s3"},
                "n2": {"n4": "s4"},
                "n3": {"n4": "s5"},
                "n4": {},
            },
            {
                "inp1": {"n1": "s1"},
                "inp2": {"n4": "s2"},
                "n1": {"n2": "s3"},
                "n2": {"n3": "s4"},
                "n3": {},
                "n4": {"n5": "s5"},
                "n5": {"n3": "s6"},
            },
        ],
        ids=["topo_2", "topo_3"],
    )
    def test_group_edges_ordered(self, succ_edges):
        """
        Test #1:
            INP1 -> N1    ->    N2 -> N3
                    N1 -> N4 -> N2
        FIXME Not supported

        Test #2:
            INP1 -> N1 -> N2 -> N4
            N1 -> N3 -> N4

        Test #3:
            INP1 -> N1 -> N2 -> N3
            INP2 -> N4 -> N5 -> N3
        """
        degrees = get_node_degrees(succ_edges)
        ordered_nodes = toposort(succ_edges)

        _degree_check(degrees, succ_edges)

        gathered = self.group_edges_proto(
            succ_edges, degrees, ordered_nodes=ordered_nodes
        )
        print()

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
        self, monkeypatch, get_mapper, build_network_with_branches_4bit
    ):
        net = build_network_with_branches_4bit

        mapper: pb.Mapper = get_mapper
        mapper.clear()
        mapper.build(net)
        grouped_edges, _ = mapper.graph.group_edges()

        # In this case, N2 & N3 should be together.
        pos_n2 = pos_n3 = 0
        for i, g in enumerate(grouped_edges):
            _g_with_name = [e.name for e in g]
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
        grouped_edges, _ = mapper.graph.group_edges()

        pos_n2 = pos_n3 = 0
        for i, g in enumerate(grouped_edges):
            _g_with_name = [e.name for e in g]
            if "s2" in _g_with_name:
                pos_n2 = i
            if "s3" in _g_with_name:
                pos_n3 = i

        assert pos_n2 != pos_n3


class TestDAGPathDistance:
    """Consider DAG only."""

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
