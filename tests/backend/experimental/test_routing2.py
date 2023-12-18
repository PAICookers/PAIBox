import pytest
from paicorelib.v2.routing_defs import (
    RoutingDirection,
    RoutingNodeCost,
    RoutingNodeLevel,
    RoutingNodeStatus,
)

import paibox as pb
from paibox.backend.experimental.routing import (
    RoutingNode,
    RoutingRoot,
    create_lx_full_tree,
    get_node_consumption,
)

pytestmark = pytest.mark.skip(reason="Not implemented")


# path, method, expected_tag
data_find_node_by_path = [
    (
        [
            RoutingDirection.X0Y0,
            RoutingDirection.X0Y0,
            RoutingDirection.X0Y1,
            RoutingDirection.X0Y0,
        ],
        "L0_1",
    ),
    (
        [
            RoutingDirection.X0Y1,
            RoutingDirection.X0Y0,
            RoutingDirection.X0Y1,
        ],
        "L1_4",
    ),
    (
        [
            RoutingDirection.X0Y1,
            RoutingDirection.X0Y0,
            RoutingDirection.X0Y0,
            RoutingDirection.X0Y0,
        ],
        "L0_2",
    ),
    (
        [],
        "L4_1",
    ),
    (
        [
            RoutingDirection.X0Y1,
            RoutingDirection.X0Y0,
            RoutingDirection.X0Y0,
            RoutingDirection.X0Y1,
        ],
        "L0_3",
    ),
    (
        [
            RoutingDirection.X0Y1,
            RoutingDirection.X0Y0,
            RoutingDirection.X0Y0,
            RoutingDirection.X1Y0,
        ],
        "L0_4",
    ),
]


def create_example_tree():
    """Example tree.

    L4            L4_1
    L3     L3_1            L3_2
    L2     L2_1            L2_2
    L1  L1_1  L1_2         L1_3      L1_4
    L0        L0_1    L0_2 L0_3 L0_4
    """
    tree = RoutingNode(RoutingNodeLevel.L4, tag="L4_1")
    l3_child1 = RoutingNode(RoutingNodeLevel.L3, tag="L3_1")
    l3_child2 = RoutingNode(RoutingNodeLevel.L3, tag="L3_2")
    l2_child1 = RoutingNode(RoutingNodeLevel.L2, tag="L2_1")
    l2_child2 = RoutingNode(RoutingNodeLevel.L2, tag="L2_2")

    l1_child1 = RoutingNode(RoutingNodeLevel.L1, tag="L1_1")
    l1_child2 = RoutingNode(RoutingNodeLevel.L1, tag="L1_2")
    l1_child3 = RoutingNode(RoutingNodeLevel.L1, tag="L1_3")
    l1_child4 = RoutingNode(RoutingNodeLevel.L1, tag="L1_4")

    l0_child1 = RoutingNode(RoutingNodeLevel.L0, tag="L0_1")
    l0_child2 = RoutingNode(RoutingNodeLevel.L0, tag="L0_2")
    l0_child3 = RoutingNode(RoutingNodeLevel.L0, tag="L0_3")
    l0_child4 = RoutingNode(RoutingNodeLevel.L0, tag="L0_4")

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
        root = RoutingNode(RoutingNodeLevel.L5)

        assert root._children == []

        l4_node1 = RoutingNode(RoutingNodeLevel.L4)
        assert root.add_child(l4_node1) == True

        with pytest.raises(ValueError):
            root.add_child(RoutingNode(RoutingNodeLevel.L3))

        l4_node2 = RoutingNode(RoutingNodeLevel.L4)
        l4_node3 = RoutingNode(RoutingNodeLevel.L4)
        l4_node4 = RoutingNode(RoutingNodeLevel.L4)
        l4_node5 = RoutingNode(RoutingNodeLevel.L4)
        root.add_child(l4_node2)
        root.add_child(l4_node3)

        assert len(root.children) == 3

        root.add_child(l4_node4)

        # Add child to a full tree node.
        assert root.add_child(l4_node5) == False

    @pytest.mark.parametrize(
        "level",
        [
            RoutingNodeLevel.L5,
            RoutingNodeLevel.L4,
            RoutingNodeLevel.L3,
            RoutingNodeLevel.L2,
            RoutingNodeLevel.L1,
        ],
    )
    def test_create_lx_full_tree(self, level):
        def _check_every_child_node(node: RoutingNode, level: RoutingNodeLevel) -> bool:
            assert node.level == level

            if node.level == RoutingNodeLevel.L1:
                assert len(node.children) == node.node_capacity
                return True
            else:
                assert root.is_full() == True
                assert root.is_empty() == False

            assert len(node.children) == node.node_capacity
            return _check_every_child_node(
                node.children[-1], RoutingNodeLevel(level - 1)
            )

        root = create_lx_full_tree(level)

        assert _check_every_child_node(root, level)

    def test_add_child(self):
        node = RoutingNode(RoutingNodeLevel.L2)

        for i in range(4):
            child = RoutingNode(RoutingNodeLevel.L1)
            assert node.add_child(child) == True

        child = RoutingNode(RoutingNodeLevel.L1)

        assert node.add_child(child) == False

        child1 = RoutingNode(RoutingNodeLevel.L0)

        with pytest.raises(ValueError):
            node.add_child(child1)

        child2 = RoutingNode(RoutingNodeLevel.L3)
        with pytest.raises(ValueError):
            node.add_child(child2)

    def test_add_child_for_L0(self):
        node_l0 = RoutingNode(RoutingNodeLevel.L0)
        child1 = RoutingNode(RoutingNodeLevel.L0)

        with pytest.raises(ValueError):
            node_l0.add_child(child1)

    def test_find_child(self):
        """Example tree.

        L4            L4_1
        L3     L3_1            L3_2
        L2     L2_1            L2_2
        L1  L1_1  L1_2         L1_3      L1_4
        L0        L0_1    L0_2 L0_3 L0_4
        """
        tree = RoutingNode(RoutingNodeLevel.L4, tag="L4_1")
        l3_child1 = RoutingNode(RoutingNodeLevel.L3, tag="L3_1")
        l3_child2 = RoutingNode(RoutingNodeLevel.L3, tag="L3_2")
        l2_child1 = RoutingNode(RoutingNodeLevel.L2, tag="L2_1")
        l2_child2 = RoutingNode(RoutingNodeLevel.L2, tag="L2_2")
        l1_child1 = RoutingNode(RoutingNodeLevel.L1, tag="L1_1")
        l1_child2 = RoutingNode(RoutingNodeLevel.L1, tag="L1_2")
        l1_child3 = RoutingNode(RoutingNodeLevel.L1, tag="L1_3")
        l1_child4 = RoutingNode(RoutingNodeLevel.L1, tag="L1_4")
        l0_child1 = RoutingNode(RoutingNodeLevel.L0, tag="L0_1")
        l0_child2 = RoutingNode(RoutingNodeLevel.L0, tag="L0_2")
        l0_child3 = RoutingNode(RoutingNodeLevel.L0, tag="L0_3")
        l0_child4 = RoutingNode(RoutingNodeLevel.L0, tag="L0_4")

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

        # assert l1_child1.find_child(l0_child1) == False
        # assert l1_child2.find_child(l0_child1) == True
        # assert l1_child3.find_child(l0_child2) == True
        # assert l2_child1.find_child(l1_child1) == True
        # assert l2_child1.find_child(l1_child2) == True
        # assert l2_child1.find_child(l1_child3) == False
        # assert l2_child2.find_child(l1_child1) == False
        # assert l2_child2.find_child(l1_child3) == True
        # assert l2_child2.find_child(l1_child4) == True
        # assert l3_child1.find_child(l2_child1) == True
        # assert l3_child1.find_child(l2_child2) == False
        # assert tree.find_child(l3_child1) == True
        # assert tree.find_child(l3_child2) == True
        # assert tree.find_child(l2_child1) == False

    example_tree = create_example_tree()

    @pytest.mark.parametrize(
        "path, expected_tag",
        data_find_node_by_path,
        ids=[
            "L0_1",
            "L1_4",
            "L0_2",
            "L4_1",
            "L0_3",
            "L0_4",
        ],
    )
    def test_find_node_by_path(self, path, expected_tag):
        find = self.example_tree.find_node_by_path(path)

        assert find.tag == expected_tag

    def test_find_node_by_path_illegal(self):
        # Length of path > 4
        path = 5 * [RoutingDirection.X0Y0]
        with pytest.raises(ValueError):
            find = self.example_tree.find_node_by_path(path)

        # X0Y1 is out of range on L2-level.
        path = [RoutingDirection.X0Y0, RoutingDirection.X0Y1]
        with pytest.raises(IndexError):
            find = self.example_tree.find_node_by_path(path)

        # X0Y1 is out of range on L0-level.
        path = [
            RoutingDirection.X0Y0,
            RoutingDirection.X0Y0,
            RoutingDirection.X0Y0,
            RoutingDirection.X0Y1,
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

    def test_get_node_path(self):
        find1 = self.example_tree.find_node_by_tag("L2_1")

        path = self.example_tree.get_node_path(find1)  # type: ignore
        assert path == [RoutingDirection.X0Y0, RoutingDirection.X0Y0]

        find2 = self.example_tree.find_node_by_tag("L2_2")

        path = self.example_tree.get_node_path(find2)  # type: ignore
        assert path == [RoutingDirection.X0Y1, RoutingDirection.X0Y0]

    def test_get_lx_nodes(self):
        root = RoutingRoot(empty_root=False, tag="Root_L5")
        nodes_l5 = root.get_lx_nodes(RoutingNodeLevel.L5)
        nodes_l4 = root.get_lx_nodes(RoutingNodeLevel.L4)
        nodes_l3 = root.get_lx_nodes(RoutingNodeLevel.L3)
        nodes_l2 = root.get_lx_nodes(RoutingNodeLevel.L2)
        nodes_l1 = root.get_lx_nodes(RoutingNodeLevel.L1)
        nodes_l0 = root.get_lx_nodes(RoutingNodeLevel.L0)

        assert len(nodes_l5) == 1
        assert len(nodes_l4) == 4
        assert len(nodes_l3) == 16
        assert len(nodes_l2) == 64
        assert len(nodes_l1) == 256
        assert len(nodes_l0) == 1024

        root2 = RoutingRoot(empty_root=True)
        nodes_l5 = root2.get_lx_nodes(RoutingNodeLevel.L5)
        nodes_l4 = root2.get_lx_nodes(RoutingNodeLevel.L4)
        nodes_l3 = root2.get_lx_nodes(RoutingNodeLevel.L3)
        nodes_l2 = root2.get_lx_nodes(RoutingNodeLevel.L2)
        nodes_l1 = root2.get_lx_nodes(RoutingNodeLevel.L1)
        nodes_l0 = root.get_lx_nodes(RoutingNodeLevel.L0)

        assert len(nodes_l5) == 1
        assert len(nodes_l4) == 0
        assert len(nodes_l3) == 0
        assert len(nodes_l2) == 0
        assert len(nodes_l1) == 0
        assert len(nodes_l0) == 1024

    def test_get_avail_child(self):
        root = create_lx_full_tree(RoutingNodeLevel.L3, "L3")

        node_l0_1 = RoutingNode(RoutingNodeLevel.L0, tag="L0_1")
        node_l0_2 = RoutingNode(RoutingNodeLevel.L0, tag="L0_2")
        node_l0_3 = RoutingNode(RoutingNodeLevel.L0, tag="L0_3")
        node_l0_4 = RoutingNode(RoutingNodeLevel.L0, tag="L0_4")
        node_l0_5 = RoutingNode(RoutingNodeLevel.L0, tag="L0_5")

        node_l1_1 = root.find_node_by_path(
            [RoutingDirection.X0Y0, RoutingDirection.X0Y1]
        )
        node_l1_1.add_child(node_l0_1)
        node_l1_1.add_child(node_l0_2)
        node_l1_1.add_child(node_l0_3)
        node_l1_1.add_child(node_l0_4)

        assert node_l1_1.status == RoutingNodeStatus.ALL_EMPTY

        node_l1_2 = root.find_node_by_path(
            [RoutingDirection.X0Y0, RoutingDirection.X0Y0]
        )
        node_l1_2.add_child(node_l0_5)

        assert node_l1_2.status == RoutingNodeStatus.ALL_EMPTY

        node_l2_1 = root.find_node_by_path([RoutingDirection.X0Y0])

        assert node_l2_1.get_avail_child() == node_l1_2

    def test_find_avail_lx_node(self):
        root = create_lx_full_tree(RoutingNodeLevel.L3, "L3")

        node_l0_1 = RoutingNode(RoutingNodeLevel.L0, tag="L0_1")
        node_l0_2 = RoutingNode(RoutingNodeLevel.L0, tag="L0_2")
        node_l0_3 = RoutingNode(RoutingNodeLevel.L0, tag="L0_3")
        node_l0_4 = RoutingNode(RoutingNodeLevel.L0, tag="L0_4")

        node_l0_5 = RoutingNode(RoutingNodeLevel.L0, tag="L0_5")
        node_l0_6 = RoutingNode(RoutingNodeLevel.L0, tag="L0_6")
        node_l0_7 = RoutingNode(RoutingNodeLevel.L0, tag="L0_7")

        node_l1_1 = root.find_node_by_path(
            [RoutingDirection.X0Y0, RoutingDirection.X0Y0]
        )
        node_l1_1.add_child(node_l0_1)
        node_l1_1.add_child(node_l0_2)
        node_l1_1.add_child(node_l0_3)
        node_l1_1.add_child(node_l0_4)

        node_l1_2 = root.find_node_by_path(
            [RoutingDirection.X0Y0, RoutingDirection.X0Y1]
        )
        node_l1_2.add_child(node_l0_5)
        node_l1_2.add_child(node_l0_6)
        node_l1_2.add_child(node_l0_7)

        node = root._find_lx_node_with_n_child_avail(RoutingNodeLevel.L1, 1)

        assert node is not None
        assert node == node_l1_1

        node = root._find_lx_node_with_n_child_avail(RoutingNodeLevel.L0, 1)
        assert node is not None
        assert node == node_l0_1

        # Add until full
        for i in range(10):
            new_node = RoutingNode(RoutingNodeLevel.L0)
            node = root._find_lx_node_with_n_child_avail(RoutingNodeLevel.L1, 1)

            assert node is not None
            assert node.add_child(new_node)

        assert node_l1_2.status == RoutingNodeStatus.OCCUPIED

        node_l1_3 = root.find_node_by_path(
            [RoutingDirection.X0Y0, RoutingDirection.X1Y0]
        )
        node_l1_4 = root.find_node_by_path(
            [RoutingDirection.X0Y0, RoutingDirection.X1Y1]
        )

        assert node_l1_3.status == RoutingNodeStatus.OCCUPIED
        assert node_l1_4.status == RoutingNodeStatus.OCCUPIED

        node_l1_5 = root.find_node_by_path(
            [RoutingDirection.X0Y1, RoutingDirection.X0Y0]
        )
        assert len(node_l1_5.children) == 1

    def test_n_child_empty(self):
        node_l2 = create_lx_full_tree(RoutingNodeLevel.L2)

        assert node_l2.n_child_empty() == node_l2.node_capacity

        node_l0 = RoutingNode(RoutingNodeLevel.L0)
        assert node_l0.n_child_empty() == node_l0.node_capacity

    def test_is_children_all_status(self):
        node_l1 = create_lx_full_tree(RoutingNodeLevel.L1, root_tag="L1_0")

        assert node_l1.is_children_all_status(RoutingNodeStatus.ALL_EMPTY) == True

        node_l0 = RoutingNode(RoutingNodeLevel.L0)
        node_l0.is_children_all_status(RoutingNodeStatus.ALL_EMPTY)

        path = RoutingDirection.X0Y0
        node = node_l1[path.to_index()]

        node.add_item(1)
        node_l1._status_update()

        assert node_l1.is_children_all_status(RoutingNodeStatus.ALL_EMPTY) == False


class ExampleNet1(pb.Network):
    """Example net.

    INP1- > S1 -> N1 -> S2 -> N2 -> S3 -> N3
    """

    def __init__(self):
        super().__init__()
        self.inp1 = pb.projection.InputProj(
            pb.simulator.processes.Constant(1200, 1), name="inp1"
        )
        self.n1 = pb.neuron.TonicSpiking(1200, 3, name="n1")
        self.n2 = pb.neuron.TonicSpiking(400, 4, name="n2")
        self.n3 = pb.neuron.TonicSpiking(800, 5, name="n3")

        self.s1 = pb.synapses.NoDecay(
            self.inp1, self.n1, pb.synapses.All2All(), name="s1"
        )
        self.s2 = pb.synapses.NoDecay(
            self.n1, self.n2, pb.synapses.All2All(), name="s2"
        )
        self.s3 = pb.synapses.NoDecay(
            self.n2, self.n3, pb.synapses.All2All(), name="s3"
        )


@pytest.fixture
def build_example_net():
    return ExampleNet1()


class TestRouterTreeRoot:
    def test_insert_n_core(self):
        root = create_lx_full_tree(RoutingNodeLevel.L3, "L3_0")

        def insert(n_core, cost):
            level, next_level_n = cost.get_routing_level()
            print(level, next_level_n)
            routing_nodes = root.find_lx_node_for_routing(
                RoutingNodeLevel(level), next_level_n
            )
            if len(routing_nodes) == 0:
                raise ValueError

            n = 0
            for i in range(len(routing_nodes)):
                router_node = routing_nodes[i]

                while router_node.status != RoutingNodeStatus.OCCUPIED:
                    if n < n_core:
                        if not router_node.add_item_to_L0_node(data=n):
                            raise ValueError
                    else:
                        if not router_node.add_item_to_L0_node(data=None):
                            raise ValueError

                    n += 1
                    # Update from the root node.
                    root.node_status_update()

            print("Insert OK")

        n_core1, cost1 = 5, RoutingNodeCost(8, 2, 1, 1, 1)
        # cost2 = RoutingNodeCost(4, 1, 1, 1, 1)
        n_core3, cost3 = 20, RoutingNodeCost(32, 8, 2, 1, 1)

        insert(n_core1, cost1)
        # insert(cost2)
        insert(n_core3, cost3)

        # TODO Debug tree, printing in good format!

        # mapper = pb.Mapper()
        # mapper.build_graph(self.net1)

        # # Group every synapses
        # mapper._group_syns()
        # mapper._build_gsyn_on_core()

        # # Insert the first `gsyn_on_core`.
        # cb_on_core1 = mapper._succ_gsyn_on_core["inp1"]["n1"]
        # mapper.router_tree.insert_gsyn_on_core(*cb_on_core1)

        # # Insert when there are leaves in router tree already.
        # cb_on_core2 = mapper._succ_gsyn_on_core["n1"]["n2"]
        # mapper.router_tree.insert_gsyn_on_core(*cb_on_core2)
