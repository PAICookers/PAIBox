from contextlib import nullcontext
import graphlib
import pytest
import random
from paibox.backend.graph_utils import *
from paibox.exceptions import GraphHasCycleError

from .conftest import TestData


def _generate_random_dag(num_nodes: int):
    assert num_nodes > 0
    nodes = list(str(i) for i in range(num_nodes))

    graph = {node: [] for node in nodes}

    for i in range(num_nodes - 1):
        # Choose the random number of successors for each node.
        n_succ = random.randint(0, num_nodes - i - 1)
        succ_node = random.sample(nodes[i + 1 :], n_succ)

        graph[nodes[i]].extend(succ_node)

    return graph


class TestTopoSort:
    def test_reverse_edges(self):
        edges = {"a": {"b", "c"}, "b": {"d"}, "c": {"d"}, "d": {}}
        expected = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}

        r = reverse_edges(edges)

        assert r == expected

        # Not all nodes are in the keys
        incomplete_edges = {"a": {"b", "c"}, "b": {"d"}, "c": {"d"}}

        with pytest.raises(KeyError):
            r = reverse_edges(incomplete_edges)

    @pytest.mark.parametrize(
        TestData.toposort_data["args"],
        TestData.toposort_data["data"],
        ids=TestData.toposort_data["ids"],  # type: ignore
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
        with pytest.raises(GraphHasCycleError):
            ordered = toposort(edges)

    @pytest.mark.parametrize(
        # Using test cases with a unique topological sort order, otherwise
        # check the length of the returned ordered list only.
        "graph",
        [
            {
                "inp1": {"n1"},
                "n1": {"n2", "n4"},
                "n2": {"n3"},
                "n3": {},
                "n4": {"n2"},
            },
            {"1": ["2", "3"], "2": ["4", "5"], "3": ["2"], "4": ["5"], "5": []},
        ],
    )
    def test_iter_toposort(self, graph):
        ts = graphlib.TopologicalSorter(graph)
        expected = list(reversed([*ts.static_order()]))

        assert list(iter_toposort(graph)) == expected

    def test_if_graph_has_cycle(self):
        n_vertices = [8, 10, 15, 20, 50, 80, 100]

        for n_v in n_vertices:
            has_cycle = False

            graph = _generate_random_dag(n_v)

            ts = graphlib.TopologicalSorter(graph)
            try:
                ts.static_order()
            except graphlib.CycleError:
                has_cycle = True

            if has_cycle:
                context = pytest.raises(GraphHasCycleError)
            else:
                context = nullcontext()

            with context:
                list(toposort(graph))

            with context:
                list(iter_toposort(graph))


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


@pytest.mark.parametrize(
    "graph, expected_n_cycles",
    [
        ({"n1": ["n2", "n4"], "n2": ["n3"], "n3": ["n4", "n5"]}, 0),
        ({"n1": ["n2"], "n2": ["n3"], "n3": ["n4", "n5"], "n4": ["n1"], "n5": []}, 1),
        # c1: 2-3-4-2, c2: 2-3-4-5-2
        (
            {
                "n1": ["n2"],
                "n2": ["n3"],
                "n3": ["n4"],
                "n4": ["n2", "n5"],
                "n5": ["n2"],
            },
            2,
        ),
        # c1: 2-3-5-6-4-2
        (
            {
                "n1": ["n2", "n3", "n5"],
                "n2": ["n3", "n7"],
                "n3": ["n5", "n7"],
                "n4": ["n2", "n7"],
                "n5": ["n6"],
                "n6": ["n4"],
                "n7": [],
            },
            1,
        ),
        (
            {
                "inp1": {"n1"},
                "n1": {"n2", "n4"},
                "n2": {"n3"},
                "n3": {},
                "n4": {"n2"},
            },
            0,
        ),
    ],
)
def test_find_cycles(graph, expected_n_cycles):
    cycles = find_cycles(graph)
    assert len(cycles) == expected_n_cycles


@pytest.mark.parametrize(
    "groups, expected_n_mergred",
    [
        ([["n5", "n3", "n4"], ["n2", "n3", "n4"], ["n1", "n7", "n6"]], 2),
        (
            [
                ["n1", "n2", "n3"],
                ["n3", "n6"],
                ["n4", "n5"],
                ["n7"],
                ["n2", "n8"],
                ["n4", "n9", "n10"],
            ],
            3,
        ),
    ],
)
def test_merge_overlapping_sets(groups, expected_n_mergred):
    consolidated = merge_overlapping_sets(groups)
    assert len(consolidated) == expected_n_mergred
