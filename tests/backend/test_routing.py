import random

import pytest
from paicorelib import Coord, HwConfig, RoutingDirection, RoutingLevel

import paibox as pb
from paibox.backend.routing import RoutingManager, get_unused_lx

from .conftest import gen_random_used_lx

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


class TestRoutingGroup:
    def test_RoutingGroup_instance(self, build_example_net1):
        net = build_example_net1

        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        # 8+5+4, 8+8+4
        assert mapper.routing_mgr.n_core_total >= mapper.n_core_required

    def test_RoutingGroup_instance2(self, monkeypatch, build_example_net2):
        net = build_example_net2

        # N1 & N2 will be split
        monkeypatch.setattr(net.n2, "_tws", 2)

        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert mapper.routing_mgr.n_core_total >= mapper.n_core_required

    def test_RoutingGroup_instance3(self, build_example_net4):
        net = build_example_net4

        # N1 & N2 will be together
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert len(mapper.core_blocks) == 3
        assert mapper.routing_mgr.n_core_total >= mapper.n_core_required

    def test_RoutingGroup_instance4(self, monkeypatch, build_example_net4):
        net = build_example_net4

        # N1 & N2 will be split
        monkeypatch.setattr(net.n3, "_tws", 3)

        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert len(mapper.core_blocks) == 4
        assert mapper.routing_mgr.n_core_total >= mapper.n_core_required


class TestRoutingManager:
    @pytest.mark.parametrize(
        "chip_list, n_cores, expected",
        [
            # expected = [(chip_idx_loc, core_loc), (...), ...]
            (
                [Coord(0, 0)],
                [100, 200, 100, 100],
                [
                    (0, 0),
                    (0, 256),
                    (0, 256 + 256),
                    (0, 256 + 256 + 128),
                ],
            ),
            (
                [Coord(0, 0)],
                [20, 10, 10, 20, 50],
                [(0, 0), (0, 32), (0, 48), (0, 64), (0, 128)],
            ),
            (
                [Coord(0, 0), Coord(1, 0)],
                [200, 120, 100, 200, 400, 200, 10],
                [
                    (0, 0),
                    (0, 256),
                    (0, 256 + 128),
                    (0, 512),
                    (1, 0),
                    (1, 512),
                    (1, 512 + 256),
                ],
            ),
        ],
    )
    def test_get_insert_location(self, chip_list, n_cores, expected, monkeypatch):
        monkeypatch.setattr(pb.BACKEND_CONFIG, "target_chip_addr", chip_list)
        root = RoutingManager(pb.BACKEND_CONFIG.target_chip_addr)
        cores_cost = [1 << (n - 1).bit_length() for n in n_cores]

        for i, (core_incoming, core_req) in enumerate(zip(cores_cost, n_cores)):
            core_loc, chip_idx_loc, routing_path = root.get_insert_location(
                core_incoming, core_req
            )
            assert expected[i][0] == chip_idx_loc
            assert (
                chip_idx_loc * HwConfig.N_CORE_MAX_INCHIP + expected[i][1] == core_loc
            )


@pytest.mark.parametrize("lx", [L4, L3, L2, L1, L0])
def test_get_unused_lx(lx):
    n_lx_max = HwConfig.N_SUB_ROUTING_NODE ** (5 - lx)
    n = random.randint(1, n_lx_max)

    used_lx = gen_random_used_lx(n, lx)
    unused_lx = get_unused_lx(used_lx, lx)

    assert len(unused_lx) == n_lx_max - len(set(used_lx))
