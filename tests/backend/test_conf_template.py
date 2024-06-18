import json

from paicorelib import Coord

import paibox as pb
from paibox.backend.conf_template import (
    export_core_params_json,
    export_core_plm_conf_json,
    export_input_conf_json,
    export_neuconf_json,
    export_output_conf_json,
)

try:
    import orjson

    print("Use orjson")
except ModuleNotFoundError:
    print("Use json")


class TestConfExport:
    def test_export_core_params_json(self, ensure_dump_dir, MockCoreConfigDict):
        core_params = {Coord(0, 0): MockCoreConfigDict}

        export_core_params_json(core_params, ensure_dump_dir)

    def test_NeuronConfig_conf_json(self, ensure_dump_dir, MockNeuronConfig):
        nconf = MockNeuronConfig
        mock_n = pb.IF(1, 1)
        export_neuconf_json({mock_n: nconf}, ensure_dump_dir / "neu_conf.json")

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
        export_core_plm_conf_json(core_plm_conf, ensure_dump_dir / "core_plm.json")

        with open(ensure_dump_dir / "core_plm.json", "r") as f:
            core_plm_conf_json = json.load(f)
            assert list(core_plm_conf_json.keys())[0] == str(chip_coord)
