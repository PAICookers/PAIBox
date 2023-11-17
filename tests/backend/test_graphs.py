import pytest

from paibox.backend.graphs import *
from paibox.exceptions import NotSupportedError


class TestTopoSort:
    @pytest.mark.parametrize(
        "edges",
        [
            (
                {
                    "inp1": {"n1"},
                    "n1": {"n2", "n4"},
                    "n2": {"n3"},
                    "n3": {},
                    "n4": {"n2"},
                }
            ),
            (
                {
                    "inp1": {"n1"},
                    "n1": {"n2", "n5"},
                    "n2": {"n3"},
                    "n3": {"n4", "n6"},
                    "n4": {},
                    "n5": {"n3", "n6"},
                    "n6": {"n7"},
                    "n7": {"n4"},
                }
            ),
            (
                {
                    "inp1": {"n1"},
                    "inp2": {"n4"},
                    "n1": {"n2"},
                    "n2": {"n3"},
                    "n3": {},
                    "n4": {"n5"},
                    "n5": {"n3"},
                }
            ),
            (
                {
                    "inp1": {"n1"},
                    "n1": {"n2", "n3"},
                    "n2": {"n4"},
                    "n3": {"n4"},
                    "n4": {},
                }
            ),
        ],
        ids=["one_input_1", "one_input_2", "multi_inputs_1", "one_input_3"],
    )
    def test_toposort(self, edges):
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
        """
        ordered = toposort(edges)
        assert len(ordered) == len(edges)

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
            ordered = toposort(edges, is_strict=True)


class TestGroupEdges:
    @pytest.mark.parametrize(
        "edges, succ_edges",
        [
            (
                ["s1", "s2", "s3", "s4", "s5"],
                {
                    "inp1": {"n1": "s1"},
                    "n1": {"n2": "s2", "n4": "s3"},
                    "n2": {"n3": "s4"},
                    "n3": {},
                    "n4": {"n2": "s5"},
                },
            ),
            (
                ["s1", "s2", "s3", "s4", "s5"],
                {
                    "inp1": {"n1": "s1"},
                    "n1": {"n2": "s2", "n3": "s3"},
                    "n2": {"n4": "s4"},
                    "n3": {"n4": "s5"},
                    "n4": {},
                },
            ),
            (
                ["s1", "s2", "s3", "s4", "s5", "s6"],
                {
                    "inp1": {"n1": "s1"},
                    "inp2": {"n4": "s2"},
                    "n1": {"n2": "s3"},
                    "n2": {"n3": "s4"},
                    "n3": {},
                    "n4": {"n5": "s5"},
                    "n5": {"n3": "s6"},
                },
            ),
        ],
        ids=["topo_1", "topo_2", "topo_3"],
    )
    def test_group_edges_ordered(self, edges, succ_edges):
        """
        Test #1:
            INP1 -> N1 -> N2 -> N3
            N1 -> N4
            N4 -> N2

        Test #2:
            INP1 -> N1 -> N2 -> N4
            N1 -> N3 -> N4

        Test #3:
            INP1 -> N1 -> N2 -> N3
            INP2 -> N4 -> N5 -> N3
            N2 -> N3
        """
        degrees = get_node_degrees(succ_edges)
        ordered_nodes = toposort(succ_edges)
        gathered = group_edges(edges, succ_edges, degrees, ordered_nodes=ordered_nodes)
        # gathered = group_edges(edges, succ_edges, degrees, ordered_nodes=None)

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
