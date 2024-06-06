from contextlib import nullcontext

import numpy as np
import pytest
from paicorelib import Coord, RoutingDirection, RoutingLevel

import paibox as pb
from paibox.backend.routing import RoutingCluster, RoutingCoord, RoutingRoot, get_parent
from paibox.exceptions import RoutingError

X0Y0 = RoutingDirection.X0Y0
X1Y0 = RoutingDirection.X1Y0
X0Y1 = RoutingDirection.X0Y1
X1Y1 = RoutingDirection.X1Y1
ANY = RoutingDirection.ANY
L5 = RoutingLevel.L5
L4 = RoutingLevel.L4
L3 = RoutingLevel.L3
L2 = RoutingLevel.L2
L1 = RoutingLevel.L1
L0 = RoutingLevel.L0


class TestRouterTree:
    def test_basics(self):
        root = RoutingCluster(L3, tag="L3")

        cluster_l2_1 = RoutingCluster(L2, tag="L2_1")
        cluster_l2_2 = RoutingCluster(L2, tag="L2_2")
        cluster_l2_3 = RoutingCluster(L2, tag="L2_3")

        assert root.add_child(cluster_l2_1) == True
        assert root.add_child_to(cluster_l2_2, X1Y1) == True

        cluster1 = root.create_child(tag="L2_created")  # X0Y1
        assert cluster1 is not None
        assert len(root.children) == 3

        assert root.add_child_to(cluster_l2_3, X1Y1) == False
        assert len(root.children) == 3

        cluster2 = root.create_child(tag="L2_created2")  # X1Y0
        assert cluster2 is not None
        assert len(root.children) == 4
        assert root.children[X1Y0] == cluster2

        cluster3 = root.create_child(tag="L2_created3")
        assert cluster3 is None

    def test_clear(self):
        root = RoutingCluster(L3, tag="L3")

        cluster_l2_1 = RoutingCluster(L2, tag="L2_1")
        cluster_l2_2 = RoutingCluster(L2, tag="L2_2")
        cluster_l2_3 = RoutingCluster(L2, tag="L2_3")

        cluster_l1_1 = RoutingCluster(L1, tag="L1_1")
        cluster_l1_2 = RoutingCluster(L1, tag="L1_2")
        cluster_l1_3 = RoutingCluster(L1, tag="L1_3")

        assert cluster_l2_1.add_child_to(cluster_l1_1, X0Y0) == True
        assert cluster_l2_2.add_child_to(cluster_l1_2, X0Y1) == True
        assert cluster_l2_3.add_child_to(cluster_l1_3, X1Y0) == True

        assert root.add_child_to(cluster_l2_1, X0Y0) == True
        assert root.add_child_to(cluster_l2_2, X1Y1) == True
        assert root.add_child_to(cluster_l2_3, X1Y0) == True

        cluster_l2_2.clear()
        assert len(cluster_l2_2.children) == 0

        root.clear()
        assert len(root.children) == 0

    def test_remove_child(self, build_example_root):
        root = build_example_root

        assert root.remove_child(X0Y1, strict=True)
        assert X0Y1 not in root

        with pytest.raises(RoutingError):
            root.remove_child(X1Y1, strict=True)

    def test_find_cluster_by_path(self):
        root = RoutingCluster(L3, tag="L3")

        cluster_l2_1 = RoutingCluster(L2, tag="L2_1")
        cluster_l2_2 = RoutingCluster(L2, tag="L2_2")
        cluster_l2_3 = RoutingCluster(L2, tag="L2_3")

        cluster_l1_1 = RoutingCluster(L1, tag="L1_1")
        cluster_l1_2 = RoutingCluster(L1, tag="L1_2")
        cluster_l1_3 = RoutingCluster(L1, tag="L1_3")

        assert cluster_l2_1.add_child_to(cluster_l1_1, X0Y0) == True
        assert cluster_l2_2.add_child_to(cluster_l1_2, X0Y1) == True
        assert cluster_l2_3.add_child_to(cluster_l1_3, X1Y0) == True

        assert root.add_child_to(cluster_l2_1, X0Y0) == True
        assert root.add_child_to(cluster_l2_2, X1Y1) == True
        assert root.add_child_to(cluster_l2_3, X1Y0) == True

        find0 = root[X0Y0]
        assert find0 == cluster_l2_1

        find1 = root.find_cluster_by_path([X0Y0, X0Y0])
        assert find1 == cluster_l1_1

        find2 = root.find_cluster_by_path([X0Y0, X0Y1])
        assert find2 is None

        find3 = root.find_cluster_by_path([X1Y0, X1Y0])
        assert find3 == cluster_l1_3

        find4 = root.find_cluster_by_path([X1Y1, X1Y0])
        assert find4 is None

    def test_get_routing_path(self):
        root = RoutingCluster(L3, tag="L3")

        cluster_l2_1 = RoutingCluster(L2, tag="L2_1")
        cluster_l2_2 = RoutingCluster(L2, tag="L2_2")
        cluster_l2_3 = RoutingCluster(L2, tag="L2_3")

        cluster_l1_1 = RoutingCluster(L1, tag="L1_1")
        cluster_l1_2 = RoutingCluster(L1, tag="L1_2")
        cluster_l1_3 = RoutingCluster(L1, tag="L1_3")
        cluster_l1_4 = RoutingCluster(L1, tag="L1_4")

        assert cluster_l2_1.add_child_to(cluster_l1_1, X0Y0) == True
        assert cluster_l2_2.add_child_to(cluster_l1_2, X0Y1) == True
        assert cluster_l2_3.add_child_to(cluster_l1_3, X1Y0) == True

        assert root.add_child_to(cluster_l2_1, X0Y0) == True
        assert root.add_child_to(cluster_l2_2, X1Y1) == True
        assert root.add_child_to(cluster_l2_3, X1Y0) == True

        assert root.get_routing_path(cluster_l2_1) == [X0Y0]
        assert root.get_routing_path(cluster_l1_3) == [
            X1Y0,
            X1Y0,
        ]

        assert root.get_routing_path(cluster_l1_3) == [
            X1Y0,
            X1Y0,
        ]
        assert root.get_routing_path(cluster_l1_4) == None

    def test_create_lx_full_tree(self):
        root = RoutingCluster(L3, tag="L3")

        cluster_l2_1 = RoutingCluster.create_lx_full_tree(L2, root_tag="L2_1")
        cluster_l2_2 = RoutingCluster.create_lx_full_tree(L2, root_tag="L2_2")
        cluster_l2_3 = RoutingCluster.create_lx_full_tree(L2, root_tag="L2_3")

        assert root.add_child(cluster_l2_1) == True
        assert root.add_child(cluster_l2_2) == True

        assert root.add_child_to(cluster_l2_3, X1Y1, False) == True

        assert len(root.children) == 3
        assert X1Y0 not in root.children.keys()

    def test_add_L0_for_placing(self):
        subtree = RoutingCluster.create_routing_tree(L3, 2)
        assert len(subtree.children) == 2

        n = 6
        for _ in range(n):
            subtree.add_L0_for_placing()

        find_l0_1 = subtree.find_leaf_at_level(L0)
        find_l0_2 = subtree.find_lx_clusters(L0, 0)

        find_l1_1 = subtree.find_lx_clusters(L1, 0)
        find_l1_2 = subtree.find_lx_clusters(L1, 2)
        find_l1_3 = subtree.find_lx_clusters(L1, 4)
        find_l1_4 = subtree.find_leaf_at_level(L1)

        find_l2 = subtree.find_lx_clusters(L2, 0)
        find_l3 = subtree.find_lx_clusters(L3, 2)

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
        subtree = RoutingCluster.create_routing_tree(L3, 2)

        find_l2 = subtree.find_leaf_at_level(L2)
        find_l1 = subtree.find_leaf_at_level(L1)

        assert len(find_l2) == 0
        assert len(find_l1) == 8

        # A L4-level routing tree.
        subtree = RoutingCluster.create_routing_tree(L4, 1)

        find_l3 = subtree.find_leaf_at_level(L3)
        find_l2 = subtree.find_leaf_at_level(L2)
        find_l1 = subtree.find_leaf_at_level(L1)

        assert len(find_l3) == 0
        assert len(find_l2) == 0
        assert len(find_l1) == 4 * 4

    def test_add_subtree(self):
        root = RoutingCluster(L4, tag="L4")
        subtree = RoutingCluster.create_routing_tree(L3, 2)

        n = 6
        for _ in range(n):
            subtree.add_L0_for_placing()

        insert = root.add_subtree(subtree, False)

        assert insert == True

        subtree2 = RoutingCluster.create_routing_tree(L3, 4)
        insert = root.add_subtree(subtree2, False)

        assert insert == True

        subtree3 = RoutingCluster.create_routing_tree(L3, 1)
        l2_cluster = subtree3.find_lx_clusters(L2)[0]
        l2_cluster.tag = "L2_new"

        insert = root.add_subtree(subtree3, False)

        assert insert == True

    def test_get_parent(self):
        root = RoutingCluster(L3, tag="L3")
        cluster_l2_1 = RoutingCluster(L2, tag="L2_1")
        cluster_l1_1 = RoutingCluster(L1, tag="L1_1")
        cluster_l1_2 = RoutingCluster(L1, tag="L1_2")
        cluster_l1_3 = RoutingCluster(L1, tag="L1_3")

        assert cluster_l2_1.add_child_to(cluster_l1_1, X0Y0) == True
        assert cluster_l2_1.add_child_to(cluster_l1_2, X0Y1) == True

        assert root.add_child_to(cluster_l2_1, X0Y0) == True

        parent1 = get_parent(root, cluster_l1_1)

        assert parent1 == cluster_l2_1

        parent2 = get_parent(root, cluster_l1_3)
        assert parent2 is None

    def test_routing_coord(self):
        root = RoutingCluster(L3, tag="L3")
        cluster_l2_1 = RoutingCluster(L2, tag="L2_1")
        cluster_l1_1 = RoutingCluster(L1, tag="L1_1")
        cluster_l1_2 = RoutingCluster(L1, tag="L1_2")
        cluster_l1_3 = RoutingCluster(L1, tag="L1_3")
        cluster_l0_1 = RoutingCluster(L0, tag="L0_1")

        assert cluster_l1_3.add_child_to(cluster_l0_1, X0Y1) == True
        assert cluster_l2_1.add_child_to(cluster_l1_1, X0Y0) == True
        assert cluster_l2_1.add_child_to(cluster_l1_2, X0Y1) == True
        assert cluster_l2_1.add_child_to(cluster_l1_3, X1Y0) == True
        assert root.add_child_to(cluster_l2_1, X1Y1) == True

        assert root.routing_coord == RoutingCoord(X0Y0, X0Y0)
        assert cluster_l1_2.routing_coord == RoutingCoord(X0Y0, X0Y0, X1Y1, X0Y1, ANY)
        assert cluster_l0_1.routing_coord == RoutingCoord(X0Y0, X0Y0, X1Y1, X1Y0, X0Y1)


class TestRoutingGroup:
    def test_RoutingGroup_instance(self, build_example_net1):
        net = build_example_net1

        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        # 8+5+4, 8+8+4
        assert mapper.routing_tree.breadth_of_lx(L0) >= mapper.n_core_required

    def test_RoutingGroup_instance2(self, monkeypatch, build_example_net2):
        net = build_example_net2

        # N1 & N2 will be split
        monkeypatch.setattr(net.n2, "_tws", 2)

        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert mapper.routing_tree.breadth_of_lx(L0) >= mapper.n_core_required

    def test_RoutingGroup_instance3(self, build_example_net4):
        net = build_example_net4

        # N1 & N2 will be together
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert len(mapper.core_blocks) == 3
        assert mapper.routing_tree.breadth_of_lx(L0) >= mapper.n_core_required

    def test_RoutingGroup_instance4(self, monkeypatch, build_example_net4):
        net = build_example_net4

        # N1 & N2 will be split
        monkeypatch.setattr(net.n3, "_tws", 3)

        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert len(mapper.core_blocks) == 4
        assert mapper.routing_tree.breadth_of_lx(L0) >= mapper.n_core_required


class TestRoutingRoot:
    def test_get_n_lxcluster(self, build_example_root, monkeypatch):
        monkeypatch.setattr(
            pb.BACKEND_CONFIG, "target_chip_addr", [Coord(0, 0), Coord(1, 0)]
        )

        root = RoutingRoot(pb.BACKEND_CONFIG.target_chip_addr)

        assert root[0].include_online == True
        assert root[1].include_online == True
        assert root[0].add_subtree(build_example_root, False) == True
        assert root[1].add_subtree(build_example_root, False) == True

        clusters_l5 = root[0].breadth_of_lx(L5)
        clusters_l4 = root[0].breadth_of_lx(L4)
        clusters_l3 = root[0].breadth_of_lx(L3)
        clusters_l2 = root[1].breadth_of_lx(L2)
        clusters_l1 = root[1].breadth_of_lx(L1)
        clusters_l0 = root[1].breadth_of_lx(L0)

        assert clusters_l5 == 1
        assert clusters_l4 == 1
        assert clusters_l3 == 1
        assert clusters_l2 == 2
        assert clusters_l1 == 5
        assert clusters_l0 == 0

        assert root.breadth_of_lx(L1) == 5 * 2
        assert root.breadth_of_lx(L2) == 2 * 2

    @staticmethod
    def _gen_routing_cluster(n_core: int):
        from paicorelib import get_routing_consumption

        cost = get_routing_consumption(n_core)
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

    @staticmethod
    def _gen_random_cores(n_core: int):
        n_core_half = n_core // 2
        cores = []

        for _ in range(n_core_half):
            cores.append(np.random.randint(1, 300, dtype=int))

        for _ in range(n_core - n_core_half):
            cores.append(np.random.randint(100, 600, dtype=int))

        return cores

    @pytest.mark.parametrize(
        "cores, expectation",
        (
            ([10, 20, 30, 40, 100, 200], nullcontext()),
            ([5, 10, 20, 100, 500], pytest.raises(RoutingError)),
        ),
    )
    def test_insert_routing_group_1chip(self, cores, expectation):
        root = RoutingRoot(pb.BACKEND_CONFIG.target_chip_addr)

        with expectation as e:
            for core in cores:
                subtree = self._gen_routing_cluster(core)
                if not root[0].add_subtree(subtree, True):
                    raise RoutingError("Insert failed.")

        huge_core = 500
        subtree = self._gen_routing_cluster(huge_core)
        assert root[0].add_subtree(subtree, True) == False  # Out of resources

    @pytest.mark.parametrize(
        "cores, expectation",
        (
            (
                [64, 128, 64],
                [
                    RoutingCoord(X0Y0, X0Y0),
                    RoutingCoord(X0Y0, X1Y0),
                    RoutingCoord(X0Y0, X1Y1),
                    RoutingCoord(X0Y0, X0Y1),
                ],
            ),
        ),
    )
    def test_insert_routing_group_detail(self, cores, expectation):
        root = RoutingRoot(pb.BACKEND_CONFIG.target_chip_addr)
        index = 0

        for core in cores:
            subtree = self._gen_routing_cluster(core)
            assert root[0].add_subtree(subtree, True) == True

            if len(subtree.children) == 1:
                assert subtree[X0Y0].routing_coord == expectation[index]
                index += 1
            elif len(subtree.children) == 2:
                assert subtree[X0Y0].routing_coord == expectation[index]
                index += 1
                assert subtree[X0Y1].routing_coord == expectation[index]
                index += 1
            elif len(subtree.children) == 4:
                assert subtree[X0Y0].parent.routing_coord == expectation[index]
                index += 1
            else:
                assert False

    @pytest.mark.parametrize(
        "cores, expectation",
        (
            ([200, 400, 600, 800, 1000], nullcontext()),
            ([80, 100, 240, 490, 500, 490, 1000, 1000], nullcontext()),
            ([512, 128, 128, 128, 32, 32, 32, 16, 1000, 1000, 1000], nullcontext()),
            ([200, 400, 600, 800, 1020], pytest.raises(RoutingError)),
            ([80, 100, 240, 490, 490, 500, 1000, 1000], pytest.raises(RoutingError)),
            ([200, 400, 600, 800, 200, 400, 200, 300], pytest.raises(RoutingError)),
        ),
    )
    def test_insert_routing_group_multichip4(self, cores, expectation):
        from paicorelib import Coord, HwConfig

        chip_list = [Coord(1, 1), Coord(1, 2), Coord(2, 1), Coord(2, 2)]
        root = RoutingRoot(chip_list)

        subtrees = []
        for core in cores:
            subtrees.append(self._gen_routing_cluster(core))

        n_wasted = [int(np.power(2, np.ceil(np.log2(core))) - core) for core in cores]

        with expectation as e:
            for i, subtree in enumerate(subtrees):
                check_hit_online = n_wasted[i] <= HwConfig.N_CORE_ONLINE
                flag = False

                for chip_root in root.chip_roots:
                    flag = chip_root.add_subtree(subtree, check_hit_online)
                    if flag:
                        break

                if not flag:
                    raise RoutingError("Insert failed.")
