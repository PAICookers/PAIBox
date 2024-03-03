import pytest
from paicorelib import RoutingCost, RoutingDirection, RoutingLevel

import paibox as pb
from paibox.backend.routing import RoutingCluster, RoutingGroup, RoutingRoot, get_parent


class TestRouterTree:
    def test_basics(self):
        root = RoutingCluster(RoutingLevel.L3, tag="L3")

        cluster_l2_1 = RoutingCluster(RoutingLevel.L2, tag="L2_1")
        cluster_l2_2 = RoutingCluster(RoutingLevel.L2, tag="L2_2")
        cluster_l2_3 = RoutingCluster(RoutingLevel.L2, tag="L2_3")

        assert root.add_child(cluster_l2_1) == True
        assert root.add_child_to(cluster_l2_2, RoutingDirection.X1Y1) == True

        cluster1 = root.create_child(tag="L2_created")  # X0Y1
        assert cluster1 is not None
        assert len(root.children) == 3

        assert root.add_child_to(cluster_l2_3, RoutingDirection.X1Y1, False) == False
        assert root.add_child_to(cluster_l2_3, RoutingDirection.X1Y1, True) == True
        assert len(root.children) == 3
        assert root.children[RoutingDirection.X1Y1] == cluster_l2_3

        cluster2 = root.create_child(False, tag="L2_created2")  # X1Y0
        assert cluster2 is not None
        assert len(root.children) == 4
        assert root.children[RoutingDirection.X1Y0] == cluster2

        cluster3 = root.create_child(False, tag="L2_created3")
        assert cluster3 is None

    def test_clear(self):
        root = RoutingCluster(RoutingLevel.L3, tag="L3")

        cluster_l2_1 = RoutingCluster(RoutingLevel.L2, tag="L2_1")
        cluster_l2_2 = RoutingCluster(RoutingLevel.L2, tag="L2_2")
        cluster_l2_3 = RoutingCluster(RoutingLevel.L2, tag="L2_3")

        cluster_l1_1 = RoutingCluster(RoutingLevel.L1, tag="L1_1")
        cluster_l1_2 = RoutingCluster(RoutingLevel.L1, tag="L1_2")
        cluster_l1_3 = RoutingCluster(RoutingLevel.L1, tag="L1_3")

        assert cluster_l2_1.add_child_to(cluster_l1_1, RoutingDirection.X0Y0) == True
        assert cluster_l2_2.add_child_to(cluster_l1_2, RoutingDirection.X0Y1) == True
        assert cluster_l2_3.add_child_to(cluster_l1_3, RoutingDirection.X1Y0) == True

        assert root.add_child_to(cluster_l2_1, RoutingDirection.X0Y0) == True
        assert root.add_child_to(cluster_l2_2, RoutingDirection.X1Y1) == True
        assert root.add_child_to(cluster_l2_3, RoutingDirection.X1Y0) == True

        cluster_l2_2.clear()
        assert len(cluster_l2_2.children) == 0

        root.clear()
        assert len(root.children) == 0

    def test_find_cluster_by_path(self):
        root = RoutingCluster(RoutingLevel.L3, tag="L3")

        cluster_l2_1 = RoutingCluster(RoutingLevel.L2, tag="L2_1")
        cluster_l2_2 = RoutingCluster(RoutingLevel.L2, tag="L2_2")
        cluster_l2_3 = RoutingCluster(RoutingLevel.L2, tag="L2_3")

        cluster_l1_1 = RoutingCluster(RoutingLevel.L1, tag="L1_1")
        cluster_l1_2 = RoutingCluster(RoutingLevel.L1, tag="L1_2")
        cluster_l1_3 = RoutingCluster(RoutingLevel.L1, tag="L1_3")

        assert cluster_l2_1.add_child_to(cluster_l1_1, RoutingDirection.X0Y0) == True
        assert cluster_l2_2.add_child_to(cluster_l1_2, RoutingDirection.X0Y1) == True
        assert cluster_l2_3.add_child_to(cluster_l1_3, RoutingDirection.X1Y0) == True

        assert root.add_child_to(cluster_l2_1, RoutingDirection.X0Y0) == True
        assert root.add_child_to(cluster_l2_2, RoutingDirection.X1Y1) == True
        assert root.add_child_to(cluster_l2_3, RoutingDirection.X1Y0) == True

        find0 = root[RoutingDirection.X0Y0]
        assert find0 == cluster_l2_1

        find1 = root.find_cluster_by_path(
            [RoutingDirection.X0Y0, RoutingDirection.X0Y0]
        )
        assert find1 == cluster_l1_1

        find2 = root.find_cluster_by_path(
            [RoutingDirection.X0Y0, RoutingDirection.X0Y1]
        )
        assert find2 is None

        find3 = root.find_cluster_by_path(
            [RoutingDirection.X1Y0, RoutingDirection.X1Y0]
        )
        assert find3 == cluster_l1_3

        find4 = root.find_cluster_by_path(
            [RoutingDirection.X1Y1, RoutingDirection.X1Y0]
        )
        assert find4 is None

    def test_get_routing_path(self):
        root = RoutingCluster(RoutingLevel.L3, tag="L3")

        cluster_l2_1 = RoutingCluster(RoutingLevel.L2, tag="L2_1")
        cluster_l2_2 = RoutingCluster(RoutingLevel.L2, tag="L2_2")
        cluster_l2_3 = RoutingCluster(RoutingLevel.L2, tag="L2_3")

        cluster_l1_1 = RoutingCluster(RoutingLevel.L1, tag="L1_1")
        cluster_l1_2 = RoutingCluster(RoutingLevel.L1, tag="L1_2")
        cluster_l1_3 = RoutingCluster(RoutingLevel.L1, tag="L1_3")
        cluster_l1_4 = RoutingCluster(RoutingLevel.L1, tag="L1_4")

        assert cluster_l2_1.add_child_to(cluster_l1_1, RoutingDirection.X0Y0) == True
        assert cluster_l2_2.add_child_to(cluster_l1_2, RoutingDirection.X0Y1) == True
        assert cluster_l2_3.add_child_to(cluster_l1_3, RoutingDirection.X1Y0) == True

        assert root.add_child_to(cluster_l2_1, RoutingDirection.X0Y0) == True
        assert root.add_child_to(cluster_l2_2, RoutingDirection.X1Y1) == True
        assert root.add_child_to(cluster_l2_3, RoutingDirection.X1Y0) == True

        assert root.get_routing_path(cluster_l2_1) == [RoutingDirection.X0Y0]
        assert root.get_routing_path(cluster_l1_3) == [
            RoutingDirection.X1Y0,
            RoutingDirection.X1Y0,
        ]

        assert root.get_routing_path(cluster_l1_3) == [
            RoutingDirection.X1Y0,
            RoutingDirection.X1Y0,
        ]
        assert root.get_routing_path(cluster_l1_4) == None

    def test_create_lx_full_tree(self):
        root = RoutingCluster(RoutingLevel.L3, tag="L3")

        cluster_l2_1 = RoutingCluster.create_lx_full_tree(
            RoutingLevel.L2, root_tag="L2_1"
        )
        cluster_l2_2 = RoutingCluster.create_lx_full_tree(
            RoutingLevel.L2, root_tag="L2_2"
        )
        cluster_l2_3 = RoutingCluster.create_lx_full_tree(
            RoutingLevel.L2, root_tag="L2_3"
        )

        assert root.add_child(cluster_l2_1) == True
        assert root.add_child(cluster_l2_2) == True

        assert root.add_child_to(cluster_l2_3, RoutingDirection.X1Y1, False) == True

        assert len(root.children) == 3
        assert RoutingDirection.X1Y0 not in root.children.keys()

    def test_add_L0_for_placing(self):
        subtree = RoutingCluster.create_routing_tree(RoutingLevel.L3, 2)
        assert len(subtree.children) == 2

        n = 6
        for i in range(n):
            subtree.add_L0_for_placing()

        find_l0_1 = subtree.find_leaf_at_level(RoutingLevel.L0)
        find_l0_2 = subtree.find_clusters_at_level(RoutingLevel.L0, 0)

        find_l1_1 = subtree.find_clusters_at_level(RoutingLevel.L1, 0)
        find_l1_2 = subtree.find_clusters_at_level(RoutingLevel.L1, 2)
        find_l1_3 = subtree.find_clusters_at_level(RoutingLevel.L1, 4)
        find_l1_4 = subtree.find_leaf_at_level(RoutingLevel.L1)

        find_l2 = subtree.find_clusters_at_level(RoutingLevel.L2, 0)
        find_l3 = subtree.find_clusters_at_level(RoutingLevel.L3, 2)

        assert len(find_l0_1) == 0
        assert len(find_l0_2) == n
        assert len(find_l1_1) == 8
        assert len(find_l1_2) == 7
        assert len(find_l1_3) == 6
        assert len(find_l1_4) == 6
        assert len(find_l2) == 2
        assert len(find_l3) == 1

        assert len(find_l1_1[0].children) == find_l1_1[0].node_capacity
        assert len(find_l1_1[1].children) == n - len(find_l1_1[0].children)

    def test_create_routing_tree(self):
        """Test for `create_routing_tree()` & `find_empty_lx_clusters()`."""
        # A L3-level routing tree.
        subtree = RoutingCluster.create_routing_tree(RoutingLevel.L3, 2)

        find_l2 = subtree.find_leaf_at_level(RoutingLevel.L2)
        find_l1 = subtree.find_leaf_at_level(RoutingLevel.L1)

        assert len(find_l2) == 0
        assert len(find_l1) == 8

        # A L4-level routing tree.
        subtree = RoutingCluster.create_routing_tree(RoutingLevel.L4, 1)

        find_l3 = subtree.find_leaf_at_level(RoutingLevel.L3)
        find_l2 = subtree.find_leaf_at_level(RoutingLevel.L2)
        find_l1 = subtree.find_leaf_at_level(RoutingLevel.L1)

        assert len(find_l3) == 0
        assert len(find_l2) == 0
        assert len(find_l1) == 4 * 4

    def test_add_subtree(self):
        root = RoutingCluster(RoutingLevel.L4, tag="L4")
        subtree = RoutingCluster.create_routing_tree(RoutingLevel.L3, 2)

        n = 6
        for i in range(n):
            subtree.add_L0_for_placing()

        insert = root.add_subtree(subtree)

        assert insert == True

        subtree2 = RoutingCluster.create_routing_tree(RoutingLevel.L3, 4)
        insert = root.add_subtree(subtree2)

        assert insert == True

        subtree3 = RoutingCluster.create_routing_tree(RoutingLevel.L3, 1)
        l2_cluster = subtree3.find_clusters_at_level(RoutingLevel.L2)[0]
        l2_cluster.tag = "L2_new"

        insert = root.add_subtree(subtree3)

        assert insert == True

    def test_get_parent(self):
        root = RoutingCluster(RoutingLevel.L3, tag="L3")

        cluster_l2_1 = RoutingCluster(RoutingLevel.L2, tag="L2_1")

        cluster_l1_1 = RoutingCluster(RoutingLevel.L1, tag="L1_1")
        cluster_l1_2 = RoutingCluster(RoutingLevel.L1, tag="L1_2")
        cluster_l1_3 = RoutingCluster(RoutingLevel.L1, tag="L1_3")

        assert cluster_l2_1.add_child_to(cluster_l1_1, RoutingDirection.X0Y0) == True
        assert cluster_l2_1.add_child_to(cluster_l1_2, RoutingDirection.X0Y1) == True

        assert root.add_child_to(cluster_l2_1, RoutingDirection.X0Y0) == True

        parent1 = get_parent(root, cluster_l1_1)

        assert parent1 == cluster_l2_1

        parent2 = get_parent(root, cluster_l1_3)
        assert parent2 is None


class TestRoutingGroup:
    def test_RoutingGroup_instance(self, build_example_net1):
        net = build_example_net1

        mapper = pb.Mapper()
        mapper.build(net)

        # Build the core blocks
        mapper.build_core_blocks()
        mapper.lcn_ex_adjustment()
        mapper.coord_assign()

        # 8+5+4, 8+8+4
        assert mapper.routing_tree.n_L0_clusters >= mapper.n_core_required

    def test_RoutingGroup_instance2(self, monkeypatch, build_example_net2):
        net = build_example_net2

        # N1 & N2 will be split
        monkeypatch.setattr(net.n2, "_tws", 2)

        mapper = pb.Mapper()
        mapper.build(net)

        # Build the core blocks
        mapper.build_core_blocks()
        mapper.lcn_ex_adjustment()
        mapper.coord_assign()

        assert mapper.routing_tree.n_L0_clusters >= mapper.n_core_required

    def test_RoutingGroup_instance3(self, build_example_net4):
        net = build_example_net4

        # N1 & N2 will be together
        mapper = pb.Mapper()
        mapper.build(net)

        # Build the core blocks
        mapper.build_core_blocks()
        mapper.lcn_ex_adjustment()
        mapper.coord_assign()

        assert len(mapper.core_blocks) == 3
        assert mapper.routing_tree.n_L0_clusters >= mapper.n_core_required

    def test_RoutingGroup_instance4(self, monkeypatch, build_example_net4):
        net = build_example_net4

        # N1 & N2 will be split
        monkeypatch.setattr(net.n3, "_tws", 3)

        mapper = pb.Mapper()
        mapper.build(net)

        # Build the core blocks
        mapper.build_core_blocks()
        mapper.lcn_ex_adjustment()
        mapper.coord_assign()

        assert len(mapper.core_blocks) == 4
        assert mapper.routing_tree.n_L0_clusters >= mapper.n_core_required


class TestRouterTreeRoot:
    def test_breadth_of_lx_clusters(self, build_example_root):
        root = RoutingRoot()

        assert root.add_subtree(build_example_root) == True

        clusters_l5 = root.breadth_of_lx_clusters(RoutingLevel.L5)
        clusters_l4 = root.breadth_of_lx_clusters(RoutingLevel.L4)
        clusters_l3 = root.breadth_of_lx_clusters(RoutingLevel.L3)
        clusters_l2 = root.breadth_of_lx_clusters(RoutingLevel.L2)
        clusters_l1 = root.breadth_of_lx_clusters(RoutingLevel.L1)
        clusters_l0 = root.breadth_of_lx_clusters(RoutingLevel.L0)

        assert clusters_l5 == 1
        assert clusters_l4 == 1
        assert clusters_l3 == 1
        assert clusters_l2 == 2
        assert clusters_l1 == 5
        assert clusters_l0 == 0

    def test_insert_coreblock_proto(self):
        root = RoutingRoot()

        def _gen_routing_tree(n_core: int, cost: RoutingCost):
            level = cost.get_routing_level()

            routing_root = RoutingCluster.create_routing_tree(level, cost[level.value])

            for i in range(cost.n_L0):
                if i < n_core:
                    if not routing_root.add_L0_for_placing(data=i):
                        raise RuntimeError
                else:
                    if not routing_root.add_L0_for_placing(data="occupied"):
                        raise RuntimeError

            return routing_root

        n_core1, cost1 = 5, RoutingCost(8, 2, 1, 1, 1)
        n_core2, cost2 = 3, RoutingCost(4, 1, 1, 1, 1)
        n_core3, cost3 = 20, RoutingCost(32, 8, 2, 1, 1)

        subtree1 = _gen_routing_tree(n_core1, cost1)
        assert root.add_subtree(subtree1) == True

        subtree2 = _gen_routing_tree(n_core2, cost2)
        assert root.add_subtree(subtree2) == True

        subtree3 = _gen_routing_tree(n_core3, cost3)
        assert root.add_subtree(subtree3) == True

    def test_insert_routing_group(self, build_example_net1):
        net = build_example_net1

        mapper = pb.Mapper()
        mapper.build(net)

        # Build the core blocks
        mapper.build_core_blocks()
        mapper.lcn_ex_adjustment()

        for cb in mapper.core_blocks:
            cb.group_neurons()

        core_blocks = mapper.core_blocks
        routing_group = RoutingGroup(*core_blocks)

        assert len(routing_group) == len(core_blocks)

        mapper.routing_tree.insert_routing_group(routing_group)

        print()
