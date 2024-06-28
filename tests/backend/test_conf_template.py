import random

from paicorelib import Coord, HwConfig

import paibox as pb
from paibox.backend.conf_template import (
    export_core_params_json,
    export_core_plm_conf_json,
    export_input_conf_json,
    export_neuconf_json,
    export_output_conf_json,
    export_used_L2_clusters,
)

try:
    import orjson as json

    print("Use orjson")
except ModuleNotFoundError:
    import json

    print("Use json")


class TestConfExport:
    def test_export_core_params_json(self, ensure_dump_dir, MockCoreConfigDict):
        core_params = {
            Coord(1, 1): {
                Coord(0, 0): MockCoreConfigDict,
                Coord(0, 1): MockCoreConfigDict,
            },
            Coord(2, 2): {Coord(0, 0): MockCoreConfigDict},
        }

        export_core_params_json(core_params, ensure_dump_dir)

    def test_NeuronConfig_conf_json(self, ensure_dump_dir, MockNeuronConfig):
        nconf = MockNeuronConfig
        mock_n = pb.IF(1, 1)
        export_neuconf_json({mock_n: nconf}, ensure_dump_dir)

    def test_export_input_conf_json(self, ensure_dump_dir, MockNeuronDest):
        iconf = {"n1": MockNeuronDest}
        export_input_conf_json(iconf, ensure_dump_dir)

    def test_export_output_conf_json(self, ensure_dump_dir, MockNeuronDestInfo):
        oconf = {"n1": {0: MockNeuronDestInfo}}
        export_output_conf_json(oconf, ensure_dump_dir)

    def test_export_core_plm_conf_json(self, ensure_dump_dir, MockCorePlmConfig):
        chip_coord = Coord(1, 1)
        core_coord = Coord(10, 10)

        core_plm_conf = {chip_coord: {core_coord: MockCorePlmConfig}}
        export_core_plm_conf_json(core_plm_conf, ensure_dump_dir)

        with open(ensure_dump_dir / "core_plm.json", "rb") as f:
            core_plm_conf_json = json.loads(f.read())
            assert list(core_plm_conf_json.keys())[0] == str(chip_coord)

    def test_export_used_L2_clusters(self, ensure_dump_dir, monkeypatch):
        from paibox.backend.conf_template import _get_clk_en_L2_dict

        from .conftest import gen_random_used_lx

        clist = [Coord(0, 0), Coord(0, 1), Coord(2, 2)]
        monkeypatch.setattr(pb.BACKEND_CONFIG, "target_chip_addr", clist)

        n_lx_max = HwConfig.N_SUB_ROUTING_NODE ** (5 - 2)
        n = random.randint(1, n_lx_max)
        used_L2 = []

        for _ in range(len(clist)):
            used_L2.append(gen_random_used_lx(n, 2))

        clk_en_L2_dict = _get_clk_en_L2_dict(
            pb.BACKEND_CONFIG.target_chip_addr, used_L2
        )

        export_used_L2_clusters(clk_en_L2_dict, ensure_dump_dir)
