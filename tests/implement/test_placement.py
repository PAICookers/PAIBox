import numpy as np
import pytest

import paibox as pb
from paibox.implement.placement import (
    RouterTreeNode,
    RouterTreeRoot,
    create_lx_full_tree,
)
from paibox.libpaicore.v2._types import RouterLevel


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


def test_RouterTreeNode():
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
    [RouterLevel.L5, RouterLevel.L4, RouterLevel.L3, RouterLevel.L2, RouterLevel.L1],
)
def test_create_lx_full_tree(level):
    def _check_every_child_node(node: RouterTreeNode, level: RouterLevel) -> bool:
        assert node.level == level

        if node.level == RouterLevel.L1:
            assert len(node.children) == 0
            return True
        else:
            assert root.is_full() == True
            assert root.is_empty() == False

        assert len(node.children) == node.capacity
        return _check_every_child_node(node.children[-1], RouterLevel(level - 1))

    root = create_lx_full_tree(level)

    assert _check_every_child_node(root, level)


class TestRouterTreeNode:
    root = RouterTreeRoot()

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
        node = RouterTreeNode(RouterLevel.L3)
        child1 = RouterTreeNode(RouterLevel.L2)
        child2 = RouterTreeNode(RouterLevel.L2)
        child3 = RouterTreeNode(RouterLevel.L2)

        assert node.add_child(child1) == True
        assert node.add_child(child2) == True

        assert node.find_child(child2) == True
        assert node.find_child(child3) == False

    def test_get_lx_nodes(self):
        nodes_l5 = self.root.get_lx_nodes(RouterLevel.L5)
        nodes_l4 = self.root.get_lx_nodes(RouterLevel.L4)
        nodes_l3 = self.root.get_lx_nodes(RouterLevel.L3)
        nodes_l2 = self.root.get_lx_nodes(RouterLevel.L2)
        nodes_l1 = self.root.get_lx_nodes(RouterLevel.L1)
        nodes_l0 = self.root.get_lx_nodes(RouterLevel.L0)

        assert len(nodes_l5) == 1
        assert len(nodes_l4) == 4
        assert len(nodes_l3) == 16
        assert len(nodes_l2) == 64
        assert len(nodes_l1) == 256


class TestRouterTreeRoot:
    root = RouterTreeRoot(empty_root=False, tag="root")

    def test_nearest_avail_lx_node(self):
        l1_node = self.root.nearest_avail_lx_node(RouterLevel.L1)
        
        L0_node1 = RouterTreeNode(RouterLevel.L0, tag="l0_1")
        L0_node2 = RouterTreeNode(RouterLevel.L0, tag="l0_2")
        L0_node3 = RouterTreeNode(RouterLevel.L0, tag="l0_3")
        L0_node4 = RouterTreeNode(RouterLevel.L0, tag="l0_4")
        
        if l1_node:
            l1_node.add_child(L0_node1)
            l1_node.add_child(L0_node2)
            l1_node.add_child(L0_node3)
            l1_node.add_child(L0_node4)
        
        l1_node_again = self.root.nearest_avail_lx_node(RouterLevel.L1)
        l2_node = self.root.nearest_avail_lx_node(RouterLevel.L2)
        
        road = self.root.get_L0_node_road(L0_node3)
        road = self.root.get_L0_node_road(L0_node2)
        
        print("Ok")