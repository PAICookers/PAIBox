import json

import pytest
from paicorelib import Coord

from paibox.backend.conf_template import (
    export_core_params_json,
    export_core_plm_conf_json,
    export_input_conf_json,
    export_neuconf_json,
    export_output_conf_json,
)


class TestConfExport:
    @pytest.mark.skip(reason="Already tested in test_export_core_plm_conf_json()")
    def test_export_core_params_json(self, ensure_dump_dir, MockCoreConfigDict):
        core_params = {Coord(0, 0): MockCoreConfigDict}

        export_core_params_json(core_params, ensure_dump_dir)

    @pytest.mark.skip(reason="Already tested in test_export_core_plm_conf_json()")
    def test_NeuronConfig_instance(self, ensure_dump_dir, MockCorePlmConfig):
        nconf = MockCorePlmConfig.neuron_configs
        export_neuconf_json(nconf, ensure_dump_dir / "neu_conf.json")

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
