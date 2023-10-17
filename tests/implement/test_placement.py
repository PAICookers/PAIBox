import paibox as pb
import pytest
from paibox.implement.placement import (
    RoutingNode,
    RoutingRoot,
    get_parent,
)
from paibox.libpaicore.v2.route import (
    RoutingDirection,
    RoutingNodeLevel,
    RoutingNodeCost,
)

@pytest.fixture
def build_example_root():
    root = RoutingNode(RoutingNodeLevel.L3, tag="L3")

    node_l2_1 = RoutingNode(RoutingNodeLevel.L2, tag="L2_1")
    node_l2_2 = RoutingNode(RoutingNodeLevel.L2, tag="L2_2")
    node_l2_3 = RoutingNode(RoutingNodeLevel.L2, tag="L2_3")

    node_l1_1 = RoutingNode(RoutingNodeLevel.L1, tag="L1_1")
    node_l1_2 = RoutingNode(RoutingNodeLevel.L1, tag="L1_2")
    node_l1_3 = RoutingNode(RoutingNodeLevel.L1, tag="L1_3")

    node_l2_1.add_child_to(node_l1_1, RoutingDirection.X0Y0)
    node_l2_2.add_child_to(node_l1_2, RoutingDirection.X0Y1)
    node_l2_3.add_child_to(node_l1_3, RoutingDirection.X1Y0)

    root.add_child_to(node_l2_1, RoutingDirection.X0Y0)
    root.add_child_to(node_l2_2, RoutingDirection.X1Y1)
    root.add_child_to(node_l2_3, RoutingDirection.X1Y0)
    
    return root

class TestRouterTree:
    def test_basics(self):
        root = RoutingNode(RoutingNodeLevel.L3, tag="L3")

        node_l2_1 = RoutingNode(RoutingNodeLevel.L2, tag="L2_1")
        node_l2_2 = RoutingNode(RoutingNodeLevel.L2, tag="L2_2")
        node_l2_3 = RoutingNode(RoutingNodeLevel.L2, tag="L2_3")

        assert root.add_child(node_l2_1) == True
        assert root.add_child_to(node_l2_2, RoutingDirection.X1Y1) == True

        node1 = root.create_child(tag="L2_created")  # X0Y1
        assert node1 is not None
        assert root.n_child == 3

        assert root.add_child_to(node_l2_3, RoutingDirection.X1Y1, False) == False
        assert root.add_child_to(node_l2_3, RoutingDirection.X1Y1, True) == True
        assert root.n_child == 3
        assert root.children[RoutingDirection.X1Y1] == node_l2_3

        node2 = root.create_child(False, tag="L2_created2")  # X1Y0
        assert node2 is not None
        assert root.n_child == 4
        assert root.children[RoutingDirection.X1Y0] == node2

        node3 = root.create_child(False, tag="L2_created3")
        assert node3 is None

    def test_find_node_by_path(self):
        root = RoutingNode(RoutingNodeLevel.L3, tag="L3")

        node_l2_1 = RoutingNode(RoutingNodeLevel.L2, tag="L2_1")
        node_l2_2 = RoutingNode(RoutingNodeLevel.L2, tag="L2_2")
        node_l2_3 = RoutingNode(RoutingNodeLevel.L2, tag="L2_3")

        node_l1_1 = RoutingNode(RoutingNodeLevel.L1, tag="L1_1")
        node_l1_2 = RoutingNode(RoutingNodeLevel.L1, tag="L1_2")
        node_l1_3 = RoutingNode(RoutingNodeLevel.L1, tag="L1_3")

        assert node_l2_1.add_child_to(node_l1_1, RoutingDirection.X0Y0) == True
        assert node_l2_2.add_child_to(node_l1_2, RoutingDirection.X0Y1) == True
        assert node_l2_3.add_child_to(node_l1_3, RoutingDirection.X1Y0) == True

        assert root.add_child_to(node_l2_1, RoutingDirection.X0Y0) == True
        assert root.add_child_to(node_l2_2, RoutingDirection.X1Y1) == True
        assert root.add_child_to(node_l2_3, RoutingDirection.X1Y0) == True

        find0 = root[RoutingDirection.X0Y0]
        assert find0 == node_l2_1

        find1 = root.find_node_by_path([RoutingDirection.X0Y0, RoutingDirection.X0Y0])
        assert find1 == node_l1_1

        find2 = root.find_node_by_path([RoutingDirection.X0Y0, RoutingDirection.X0Y1])
        assert find2 is None

        find3 = root.find_node_by_path([RoutingDirection.X1Y0, RoutingDirection.X1Y0])
        assert find3 == node_l1_3

        find4 = root.find_node_by_path([RoutingDirection.X1Y1, RoutingDirection.X1Y0])
        assert find4 is None

    def test_create_lx_full_tree(self):
        root = RoutingNode(RoutingNodeLevel.L3, tag="L3")

        node_l2_1 = RoutingNode.create_lx_full_tree(RoutingNodeLevel.L2, "L2_1")
        node_l2_2 = RoutingNode.create_lx_full_tree(RoutingNodeLevel.L2, "L2_2")
        node_l2_3 = RoutingNode.create_lx_full_tree(RoutingNodeLevel.L2, "L2_3")

        assert root.add_child(node_l2_1) == True
        assert root.add_child(node_l2_2) == True

        assert root.add_child_to(node_l2_3, RoutingDirection.X1Y1, False) == True

        assert root.n_child == 3
        assert RoutingDirection.X1Y0 not in root.children.keys()

    def test_add_L0_for_placing(self):
        subtree = RoutingNode.create_routing_tree(RoutingNodeLevel.L3, 2)

        assert subtree.n_child == 2

        n = 6
        for i in range(n):
            assert subtree.add_L0_for_placing(i) == True

        find_l0_1 = subtree.find_empty_nodes_at_level(RoutingNodeLevel.L0)
        find_l0_2 = subtree.find_nodes_at_level(RoutingNodeLevel.L0, 0)

        find_l1_1 = subtree.find_nodes_at_level(RoutingNodeLevel.L1, 0)
        find_l1_2 = subtree.find_nodes_at_level(RoutingNodeLevel.L1, 2)
        find_l1_3 = subtree.find_nodes_at_level(RoutingNodeLevel.L1, 4)
        find_l1_4 = subtree.find_empty_nodes_at_level(RoutingNodeLevel.L1)

        find_l2 = subtree.find_nodes_at_level(RoutingNodeLevel.L2, 0)
        find_l3 = subtree.find_nodes_at_level(RoutingNodeLevel.L3, 2)

        assert len(find_l0_1) == 0
        assert len(find_l0_2) == n
        assert len(find_l1_1) == 8
        assert len(find_l1_2) == 7
        assert len(find_l1_3) == 6
        assert len(find_l1_4) == 6
        assert len(find_l2) == 2
        assert len(find_l3) == 1

        assert find_l1_1[0].n_child == find_l1_1[0].node_capacity
        assert find_l1_1[1].n_child == n - find_l1_1[0].n_child

    def test_create_routing_tree(self):
        """Test for `create_routing_tree()` & `find_empty_lx_nodes()`."""
        # A L3-level routing tree.
        subtree = RoutingNode.create_routing_tree(RoutingNodeLevel.L3, 2)

        find_l2 = subtree.find_empty_nodes_at_level(RoutingNodeLevel.L2)
        find_l1 = subtree.find_empty_nodes_at_level(RoutingNodeLevel.L1)

        assert len(find_l2) == 1 * 2
        assert len(find_l1) == 1 * 2 * 4

        # A L4-level routing tree.
        subtree = RoutingNode.create_routing_tree(RoutingNodeLevel.L4, 1)

        find_l3 = subtree.find_empty_nodes_at_level(RoutingNodeLevel.L3)
        find_l2 = subtree.find_empty_nodes_at_level(RoutingNodeLevel.L2)
        find_l1 = subtree.find_empty_nodes_at_level(RoutingNodeLevel.L1)

        assert len(find_l3) == 1
        assert len(find_l2) == 1 * 4
        assert len(find_l1) == 1 * 4 * 4

    def test_add_subtree(self):
        root = RoutingNode(RoutingNodeLevel.L4, tag="L4")
        subtree = RoutingNode.create_routing_tree(RoutingNodeLevel.L3, 2)

        n = 6
        for i in range(n):
            assert subtree.add_L0_for_placing(i) == True

        insert = root.add_subtree(subtree)

        assert insert is not None

        subtree2 = RoutingNode.create_routing_tree(RoutingNodeLevel.L3, 4)
        insert = root.add_subtree(subtree2)

        assert insert is not None

        subtree3 = RoutingNode.create_routing_tree(RoutingNodeLevel.L3, 1)
        l2_node = subtree3.find_empty_nodes_at_level(RoutingNodeLevel.L2)[0]
        l2_node.tag = "L2_new"

        insert = root.add_subtree(subtree3)

        assert insert is not None

    def test_get_parent(self):
        root = RoutingNode(RoutingNodeLevel.L3, tag="L3")

        node_l2_1 = RoutingNode(RoutingNodeLevel.L2, tag="L2_1")

        node_l1_1 = RoutingNode(RoutingNodeLevel.L1, tag="L1_1")
        node_l1_2 = RoutingNode(RoutingNodeLevel.L1, tag="L1_2")
        node_l1_3 = RoutingNode(RoutingNodeLevel.L1, tag="L1_3")

        assert node_l2_1.add_child_to(node_l1_1, RoutingDirection.X0Y0) == True
        assert node_l2_1.add_child_to(node_l1_2, RoutingDirection.X0Y1) == True

        assert root.add_child_to(node_l2_1, RoutingDirection.X0Y0) == True

        parent1 = get_parent(root, node_l1_1)

        assert parent1 == node_l2_1

        parent2 = get_parent(root, node_l1_3)
        assert parent2 is None


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
    
    def test_breadth_of_lx_nodes(self, build_example_root):
        root = RoutingRoot()
        
        assert root.add_subtree(build_example_root) == True
        
        nodes_l5 = root.breadth_of_lx_nodes(RoutingNodeLevel.L5)
        nodes_l4 = root.breadth_of_lx_nodes(RoutingNodeLevel.L4)
        nodes_l3 = root.breadth_of_lx_nodes(RoutingNodeLevel.L3)
        nodes_l2 = root.breadth_of_lx_nodes(RoutingNodeLevel.L2)
        nodes_l1 = root.breadth_of_lx_nodes(RoutingNodeLevel.L1)
        nodes_l0 = root.breadth_of_lx_nodes(RoutingNodeLevel.L0)
        
        assert nodes_l5 == 1
        assert nodes_l4 == 1
        assert nodes_l3 == 1
        assert nodes_l2 == 3
        assert nodes_l1 == 3
        assert nodes_l0 == 0
        
    def test_insert_gsyn_on_core_proto(self):
        root = RoutingRoot()

        def _gen_subtree(n_core: int, cost: RoutingNodeCost):
            level, next_level_n = cost.get_routing_level()

            routing_root = RoutingNode.create_routing_tree(level, next_level_n)

            for i in range(cost.n_L0):
                if i < n_core:
                    if not routing_root.add_L0_for_placing(data=i):
                        raise RuntimeError
                else:
                    if not routing_root.add_L0_for_placing(data="occupied"):
                        raise RuntimeError

                i += 1

            return routing_root

        n_core1, cost1 = 5, RoutingNodeCost(8, 2, 1, 1, 1)
        n_core2, cost2 = 3, RoutingNodeCost(4, 1, 1, 1, 1)
        n_core3, cost3 = 20, RoutingNodeCost(32, 8, 2, 1, 1)

        subtree1 = _gen_subtree(n_core1, cost1)
        assert root.add_subtree(subtree1) is not None

        subtree2 = _gen_subtree(n_core2, cost2)
        assert root.add_subtree(subtree2) is not None

        subtree3 = _gen_subtree(n_core3, cost3)
        assert root.add_subtree(subtree3) is not None

    def test_insert_gsyn_on_core(self, build_example_net):
        net = build_example_net

        mapper = pb.implement.Mapper()
        mapper.build_graph(net)

        # Group every synapses
        mapper._group_synapses()
        mapper._build_gsyn_on_core()

        # Insert the first `gsyn_on_core`.
        gsyns_on_core1 = mapper._succ_gsyn_on_core["inp1"]["n1"]
        mapper.routing_tree.insert_gsyn_on_core(*gsyns_on_core1)

        # Insert when there are leaves in router tree already.
        gsyns_on_core2 = mapper._succ_gsyn_on_core["n1"]["n2"]
        mapper.routing_tree.insert_gsyn_on_core(*gsyns_on_core2)
