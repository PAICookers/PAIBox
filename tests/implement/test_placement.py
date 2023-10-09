import numpy as np
import pytest

import paibox as pb
from paibox.implement.placement import (
    RouterTreeNode,
    RouterTreeRoot,
    create_lx_full_tree,
)
from paibox.libpaicore.v2._types import RouterDirection, RouterLevel

from .data import *


def test_weight2binary_connectivity():
    """Test for weight matrix converting to binary connectivity."""
    o = np.zeros((6, 4), np.int8)
    # o = np.array(
    #     [
    #         [1, 2, 3, 4],
    #         [5, 6, 7, 8],
    #         [9, 10, 11, 12],
    #         [13, 14, 15, 16],
    #         [17, 18, 19, 20],
    #         [21, 22, 23, 24],
    #     ],
    #     np.int8,
    # )
    # (8, 2)
    a = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]], np.int8
    )

    b = np.split(a, [6], axis=0)

    assert len(b) == 2

    a_old, a_new = b[0], b[1]

    for i in range(0, 2):
        o[:, 2 * i] = a_old[:, i]
        o[:, 2 * i + 1] = np.pad(
            a_new[:, i], (0, 6 - (8 - 6)), "constant", constant_values=0
        )

    expected = np.array(
        [
            [1, 13, 2, 14],
            [3, 15, 4, 16],
            [5, 0, 6, 0],
            [7, 0, 8, 0],
            [9, 0, 10, 0],
            [11, 0, 12, 0],
        ],
        np.int8,
    )

    assert np.array_equal(o, expected)


@pytest.mark.parametrize(
    "cur_shape, expected_shape",
    [((8, 2), (6, 4)), ((120, 20), (100, 50)), ((1200, 144), (1152, 512))],
)
def test_w2bc_parameterized(cur_shape, expected_shape):
    """When LCN_EX > 1X, do w2bc. Else don't need to do so."""
    cur_total = cur_shape[0] * cur_shape[1]

    cur_matrix = np.random.randint(-128, 128, size=cur_shape, dtype=np.int8)

    o_matrix = np.zeros(expected_shape, dtype=np.int8)

    # Certainty
    assert cur_shape[0] > expected_shape[0]

    for i in range(cur_shape[1]):
        o_matrix[:, 2 * i] = cur_matrix[: expected_shape[0], i]
        o_matrix[:, 2 * i + 1] = np.pad(
            cur_matrix[expected_shape[0] :, i],
            (0, 2 * expected_shape[0] - cur_shape[0]),
            "constant",
            constant_values=0,
        )

    # o_matrix[:, :cur_shape[1]] = cur_matrix[: expected_shape[0],:]
    # o_matrix = np.insert(cur_matrix, slice(1, expected_shape[1], 1), 0, axis=1)

    for i in range(cur_shape[1]):
        assert np.array_equal(cur_matrix[: expected_shape[0], i], o_matrix[:, 2 * i])
        assert np.array_equal(
            cur_matrix[expected_shape[0] :, i],
            o_matrix[: cur_shape[0] - expected_shape[0], 2 * i + 1],
        )


def create_example_tree():
    tree = RouterTreeNode(RouterLevel.L4, tag="L4_1")
    l3_child1 = RouterTreeNode(RouterLevel.L3, tag="L3_1")
    l3_child2 = RouterTreeNode(RouterLevel.L3, tag="L3_2")
    l2_child1 = RouterTreeNode(RouterLevel.L2, tag="L2_1")
    l2_child2 = RouterTreeNode(RouterLevel.L2, tag="L2_2")
    l1_child1 = RouterTreeNode(RouterLevel.L1, tag="L1_1")
    l1_child2 = RouterTreeNode(RouterLevel.L1, tag="L1_2")
    l1_child3 = RouterTreeNode(RouterLevel.L1, tag="L1_3")
    l1_child4 = RouterTreeNode(RouterLevel.L1, tag="L1_4")
    l0_child1 = RouterTreeNode(RouterLevel.L0, tag="L0_1")
    l0_child2 = RouterTreeNode(RouterLevel.L0, tag="L0_2")
    l0_child3 = RouterTreeNode(RouterLevel.L0, tag="L0_3")
    l0_child4 = RouterTreeNode(RouterLevel.L0, tag="L0_4")

    l1_child2.add_child(l0_child1)
    l1_child3.add_child(l0_child2)
    l1_child3.add_child(l0_child3)
    l1_child3.add_child(l0_child4)
    l2_child1.add_child(l1_child1)
    l2_child1.add_child(l1_child2)
    l2_child2.add_child(l1_child3)
    l2_child2.add_child(l1_child4)

    l3_child1.add_child(l2_child1)
    l3_child2.add_child(l2_child2)

    tree.add_child(l3_child1)
    tree.add_child(l3_child2)

    return tree


class TestRouterTreeNode:
    def test_RouterTreeNode(self):
        root = RouterTreeNode(RouterLevel.L5)

        assert root._children == []

        l4_node1 = RouterTreeNode(RouterLevel.L4)
        assert root.add_child(l4_node1) == True

        with pytest.raises(ValueError):
            root.add_child(RouterTreeNode(RouterLevel.L3))

        l4_node2 = RouterTreeNode(RouterLevel.L4)
        l4_node3 = RouterTreeNode(RouterLevel.L4)
        l4_node4 = RouterTreeNode(RouterLevel.L4)
        root.add_child(l4_node2)
        root.add_child(l4_node3)

        assert len(root.children) == 3

        assert root.find_child(l4_node2) == True
        assert root.find_child(l4_node4) == False

        l3_node1 = RouterTreeNode(RouterLevel.L3)

    @pytest.mark.parametrize(
        "level",
        [
            RouterLevel.L5,
            RouterLevel.L4,
            RouterLevel.L3,
            RouterLevel.L2,
            RouterLevel.L1,
        ],
    )
    def test_create_lx_full_tree(self, level):
        def _check_every_child_node(node: RouterTreeNode, level: RouterLevel) -> bool:
            assert node.level == level

            if node.level == RouterLevel.L1:
                assert len(node.children) == 0
                return True
            else:
                assert root.is_full() == True
                assert root.is_empty() == False

            assert len(node.children) == node.node_capacity
            return _check_every_child_node(node.children[-1], RouterLevel(level - 1))

        root = create_lx_full_tree(level)

        assert _check_every_child_node(root, level)

    def test_add_child(self):
        node = RouterTreeNode(RouterLevel.L2)

        for i in range(4):
            child = RouterTreeNode(RouterLevel.L1)
            assert node.add_child(child) == True

        child = RouterTreeNode(RouterLevel.L1)

        assert node.add_child(child) == False

        child1 = RouterTreeNode(RouterLevel.L0)

        with pytest.raises(ValueError):
            node.add_child(child1)

        child2 = RouterTreeNode(RouterLevel.L3)
        with pytest.raises(ValueError):
            node.add_child(child2)

    def test_find_child(self):
        """Example tree.

        L4            L4_1
        L3     L3_1            L3_2
        L2     L2_1            L2_2
        L1  L1_1  L1_2         L1_3      L1_4
        L0        L0_1    L0_2 L0_3 L0_4
        """
        tree = RouterTreeNode(RouterLevel.L4, tag="L4_1")
        l3_child1 = RouterTreeNode(RouterLevel.L3, tag="L3_1")
        l3_child2 = RouterTreeNode(RouterLevel.L3, tag="L3_2")
        l2_child1 = RouterTreeNode(RouterLevel.L2, tag="L2_1")
        l2_child2 = RouterTreeNode(RouterLevel.L2, tag="L2_2")
        l1_child1 = RouterTreeNode(RouterLevel.L1, tag="L1_1")
        l1_child2 = RouterTreeNode(RouterLevel.L1, tag="L1_2")
        l1_child3 = RouterTreeNode(RouterLevel.L1, tag="L1_3")
        l1_child4 = RouterTreeNode(RouterLevel.L1, tag="L1_4")
        l0_child1 = RouterTreeNode(RouterLevel.L0, tag="L0_1")
        l0_child2 = RouterTreeNode(RouterLevel.L0, tag="L0_2")
        l0_child3 = RouterTreeNode(RouterLevel.L0, tag="L0_3")
        l0_child4 = RouterTreeNode(RouterLevel.L0, tag="L0_4")

        assert l1_child2.add_child(l0_child1) == True
        assert l1_child3.add_child(l0_child2) == True
        assert l1_child3.add_child(l0_child3) == True
        assert l1_child3.add_child(l0_child4) == True
        assert l2_child1.add_child(l1_child1) == True
        assert l2_child1.add_child(l1_child2) == True
        assert l2_child2.add_child(l1_child3) == True
        assert l2_child2.add_child(l1_child4) == True

        assert l3_child1.add_child(l2_child1) == True
        assert l3_child2.add_child(l2_child2) == True

        assert tree.add_child(l3_child1) == True
        assert tree.add_child(l3_child2) == True

        assert l1_child1.find_child(l0_child1) == False
        assert l1_child2.find_child(l0_child1) == True
        assert l1_child3.find_child(l0_child2) == True
        assert l2_child1.find_child(l1_child1) == True
        assert l2_child1.find_child(l1_child2) == True
        assert l2_child1.find_child(l1_child3) == False
        assert l2_child2.find_child(l1_child1) == False
        assert l2_child2.find_child(l1_child3) == True
        assert l2_child2.find_child(l1_child4) == True
        assert l3_child1.find_child(l2_child1) == True
        assert l3_child1.find_child(l2_child2) == False
        assert tree.find_child(l3_child1) == True
        assert tree.find_child(l3_child2) == True
        assert tree.find_child(l2_child1) == False

    example_tree = create_example_tree()

    @pytest.mark.parametrize(
        "path, method, expected_tag",
        data_find_node_by_path,
        ids=[
            "L0_1_Y",
            "L1_4_Y",
            "L0_2_Y",
            "L4_1_Y",
            "L4_1_X",
            "L0_1_X",
            "L0_2_X",
            "L0_3_Y",
            "L0_3_X",
            "L0_4_Y",
            "L0_4_X",
        ],
    )
    def test_find_node_by_path(self, path, method, expected_tag):
        find = self.example_tree.find_node_by_path(path, method)

        assert find.tag == expected_tag

    def test_find_node_by_path_illegal(self):
        # Length of path > 4
        path = 5 * [RouterDirection.X0Y0]
        with pytest.raises(ValueError):
            find = self.example_tree.find_node_by_path(path)

        # X0Y1 is out of range on L2-level.
        path = [RouterDirection.X0Y0, RouterDirection.X0Y1]
        with pytest.raises(IndexError):
            find = self.example_tree.find_node_by_path(path)

        # X0Y1 is out of range on L0-level.
        path = [
            RouterDirection.X0Y0,
            RouterDirection.X0Y0,
            RouterDirection.X0Y0,
            RouterDirection.X0Y1,
        ]
        with pytest.raises(IndexError):
            find = self.example_tree.find_node_by_path(path)

    def test_find_node_by_tag(self):
        find1 = self.example_tree.find_node_by_tag("L2_1")
        find2 = self.example_tree.find_node_by_tag("L1_5")
        find3 = self.example_tree.find_node_by_tag("L0_1")
        find4 = self.example_tree.find_node_by_tag("L0_3")

        assert find1 is not None
        assert find2 is None
        assert find3 is not None
        assert find4 is not None

    def test_get_lx_nodes(self):
        root = RouterTreeRoot(empty_root=False)
        nodes_l5 = root.get_lx_nodes(RouterLevel.L5)
        nodes_l4 = root.get_lx_nodes(RouterLevel.L4)
        nodes_l3 = root.get_lx_nodes(RouterLevel.L3)
        nodes_l2 = root.get_lx_nodes(RouterLevel.L2)
        nodes_l1 = root.get_lx_nodes(RouterLevel.L1)
        nodes_l0 = root.get_lx_nodes(RouterLevel.L0)

        assert len(nodes_l5) == 1
        assert len(nodes_l4) == 4
        assert len(nodes_l3) == 16
        assert len(nodes_l2) == 64
        assert len(nodes_l1) == 256
        assert len(nodes_l0) == 0

        root2 = RouterTreeRoot(empty_root=True)
        nodes_l5 = root2.get_lx_nodes(RouterLevel.L5)
        nodes_l4 = root2.get_lx_nodes(RouterLevel.L4)
        nodes_l3 = root2.get_lx_nodes(RouterLevel.L3)
        nodes_l2 = root2.get_lx_nodes(RouterLevel.L2)
        nodes_l1 = root2.get_lx_nodes(RouterLevel.L1)
        nodes_l0 = root.get_lx_nodes(RouterLevel.L0)

        assert len(nodes_l5) == 1
        assert len(nodes_l4) == 0
        assert len(nodes_l3) == 0
        assert len(nodes_l2) == 0
        assert len(nodes_l1) == 0
        assert len(nodes_l0) == 0


class ExampleNet1(pb.Network):
    """Example net.

    N1 -> S1 -> N3
    N2 -> S2 -> N3
    """

    def __init__(self):
        super().__init__()
        self.n1 = pb.neuron.TonicSpiking(400, 3, name="n1")
        self.n2 = pb.neuron.TonicSpiking(400, 3, name="n2")
        self.n3 = pb.neuron.TonicSpiking(400, 4, name="n3")

        self.s1 = pb.synapses.NoDecay(
            self.n1, self.n2, pb.synapses.All2All(), name="s1"
        )
        self.s2 = pb.synapses.NoDecay(
            self.n2, self.n3, pb.synapses.All2All(), name="s2"
        )


class TestRouterTreeRoot:
    root = RouterTreeRoot(empty_root=False, tag="root")
    net1 = ExampleNet1()

    def test_nearest_avail_lx_node(self):
        l1_node = self.root.nearest_avail_lx_node(RouterLevel.L1)

        L0_node1 = RouterTreeNode(RouterLevel.L0, tag="L0_1")
        L0_node2 = RouterTreeNode(RouterLevel.L0, tag="L0_2")
        L0_node3 = RouterTreeNode(RouterLevel.L0, tag="L0_3")
        L0_node4 = RouterTreeNode(RouterLevel.L0, tag="L0_4")

        if l1_node:
            l1_node.add_child(L0_node1)
            l1_node.add_child(L0_node2)
            l1_node.add_child(L0_node3)
            l1_node.add_child(L0_node4)

        l1_node_again = self.root.nearest_avail_lx_node(RouterLevel.L1)
        l2_node = self.root.nearest_avail_lx_node(RouterLevel.L2)

        path1 = self.root.get_L0_node_path(L0_node3)
        path2 = self.root.get_L0_node_path(L0_node2)

    def test_insert_gsyn(self):
        mapper = pb.implement.Mapper()
        mapper.build_graph(self.net1)

        # Group every synapses
        mapper._group_syns()
        mapper._build_gsyn_on_core()

        target_gsyns_on_core = mapper._pred_gsyn_on_core["n2"]

        for gsyn_on_core in target_gsyns_on_core:
            self.root.insert_gsyn_on_core(gsyn_on_core)
