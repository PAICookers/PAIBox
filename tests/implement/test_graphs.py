import pytest

from paibox.implement.graphs import toposort


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
                # ["inp1", "n1", "n4", "n2", "n3"],
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
                # ["inp1", "n1", "n5", "n2", "n3", "n6", "n7", "n4"],
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
        ],
        ids=["one_input_1", "one_input_2", "multi_inputs_1"],
    )
    def test_toposort_abstract(self, edges):
        """
        Test #1: one input
        INP1 -> N1 -> N2 -> N3
        N1 -> N4
        N4 -> N2

        Test #2: one input
        INP1 -> N1 -> N2 -> N3 -> N4
        N1 -> N5 -> N6 -> N7 -> N4
        N5 -> N3
        N3 -> N6

        Test #2: more than one input
        INP1 -> N1 -> N2 -> N3
        INP2 -> N4 -> N5 -> N3
        N2 -> N4
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
        ],
    )
    def test_toposort_has_cycle(self, edges):
        """
        Test #1: one input
        INP1 -> N1 -> N2 -> N3
        N4 -> N1
        N2 -> N4

        Test #2: one input
        INP1 -> N1 -> N2 -> N3 -> N4
        N3 -> N2

        Test #3: more inputs
        """
        with pytest.raises(ValueError):
            ordered = toposort(edges)
