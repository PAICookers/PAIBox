import json

import pytest

from paibox.libpaicore import *


@pytest.mark.parametrize(
    "coord, params",
    [
        (
            Coord(0, 0),
            {
                "weight_precision": WeightPrecision.WEIGHT_WIDTH_1BIT,
                "lcn_extension": LCN_EX.LCN_2X,
                "input_width_format": InputWidthFormat.WIDTH_1BIT,
                "spike_width_format": SpikeWidthFormat.WIDTH_1BIT,
                "neuron_num": 100,
                "max_pooling_en": MaxPoolingEnable.DISABLE,
                "tick_wait_start": 0,
                "tick_wait_end": 0,
                "snn_mode_en": SNNModeEnable.ENABLE,
                "target_lcn": LCN_EX.LCN_1X,
                "test_chip_addr": Coord(0, 0),
            },
        ),
        (
            Coord(0, 1),
            {
                "weight_precision": WeightPrecision.WEIGHT_WIDTH_1BIT,
                "lcn_extension": LCN_EX.LCN_2X,
                "input_width_format": InputWidthFormat.WIDTH_1BIT,
                "spike_width_format": SpikeWidthFormat.WIDTH_1BIT,
                "neuron_num": 500,
                "max_pooling_en": MaxPoolingEnable.DISABLE,
                "tick_wait_start": 0,
                "tick_wait_end": 0,
                "snn_mode_en": SNNModeEnable.ENABLE,
                "target_lcn": LCN_EX.LCN_2X,
                # "test_chip_addr": Coord(0, 0),
                "unused_key": 999,
            },
        ),
    ],
)
def test_CoreParams_instance(ensure_dump_dir, coord, params):
    params_reg = ParamsReg.model_validate(params, strict=True)

    params_dict = params_reg.model_dump(by_alias=True)
    assert isinstance(params_dict["test_chip_addr"], int)

    with open(ensure_dump_dir / f"reg_model_{coord.address}.json", "w") as f:
        json.dump({coord.address: params_dict}, f, indent=4, ensure_ascii=True)
