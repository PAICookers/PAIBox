import pytest
import json
from pathlib import Path

import paibox as pb
from paibox.libpaicore import NeuronAttrs, NeuronDestInfo


@pytest.fixture(scope="module")
def ensure_dump_dir() -> Path:
    p = Path(__file__).parent / "debug"

    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)

    return p


def test_NeuronParams_instance(ensure_dump_dir):
    n1 = pb.neuron.LIF((100,), 3)

    attrs = NeuronAttrs.model_validate(n1.export_params(), strict=True)

    attrs_dict = attrs.model_dump(by_alias=True)

    with open(ensure_dump_dir / f"ram_model_{n1.name}.json", "w") as f:
        json.dump({n1.name: attrs_dict}, f, indent=4, ensure_ascii=True)


@pytest.mark.parametrize(
    "params",
    [
        {
            "tick_relative": [0] * 100 + [1] * 100,
            "addr_axon": list(range(0, 200)),
            "addr_core_x": 0,
            "addr_core_y": 1,
            "addr_core_x_ex": 0,
            "addr_core_y_ex": 0,
            "addr_chip_x": 0,
            "addr_chip_y": 0,
        },
        {
            "tick_relative": [0] * 100,
            "addr_axon": list(range(0, 100)),
            "addr_core_x": 0,
            "addr_core_y": 1,
            "addr_core_x_ex": 0,
            "addr_core_y_ex": 0,
            "addr_chip_x": 0,
            "addr_chip_y": 0,
        },
    ],
)
def test_NeuronDestInfo_instance(ensure_dump_dir, params):
    dest_info = NeuronDestInfo.model_validate(params, strict=True)

    dest_info_dict = dest_info.model_dump(by_alias=True)

    with open(ensure_dump_dir / f"ram_model_dest.json", "w") as f:
        json.dump(dest_info_dict, f, indent=4, ensure_ascii=True)


@pytest.mark.parametrize(
    "params",
    [
        {
            # Different length
            "tick_relative": [0] * 100,
            "addr_axon": list(range(0, 200)),
            "addr_core_x": 0,
            "addr_core_y": 1,
            "addr_core_x_ex": 0,
            "addr_core_y_ex": 0,
            "addr_chip_x": 0,
            "addr_chip_y": 0,
        },
        {
            # illegal type
            "tick_relative": [0] * 100,
            "addr_axon": ["illegal"] * 100,
            "addr_core_x": 0,
            "addr_core_y": 1,
            "addr_core_x_ex": 0,
            "addr_core_y_ex": 0,
            "addr_chip_x": 0,
            "addr_chip_y": 0,
        },
        {
            "tick_relative": [0] * 100,
            "addr_axon": list(range(0, 100)),
            "addr_core_x": 0,
            "addr_core_y": 1,
            # Missing some key words
            # "addr_core_x_ex": 0,
            # "addr_core_y_ex": 0,
            "addr_chip_x": 0,
            "addr_chip_y": 0,
        },
        {
            # out of range
            "tick_relative": [0] * 99 + [1 << 9],
            "addr_axon": list(range(0, 100)),
            "addr_core_x": 0,
            "addr_core_y": 1,
            "addr_core_x_ex": 0,
            "addr_core_y_ex": 0,
            "addr_chip_x": 0,
            "addr_chip_y": 0,
        },
        {
            "tick_relative": [0] * 100,
            "addr_axon": list(range(0, 100)),
            "addr_core_x": 0,
            # out of range
            "addr_core_y": 32,
            "addr_core_x_ex": 0,
            "addr_core_y_ex": 0,
            "addr_chip_x": 0,
            "addr_chip_y": 0,
        },
        {
            "tick_relative": [0] * 100,
            # out of range
            "addr_axon": [1 << 11] * 100,
            "addr_core_x": 0,
            "addr_core_y": 0,
            "addr_core_x_ex": 0,
            "addr_core_y_ex": 0,
            "addr_chip_x": 0,
            "addr_chip_y": 0,
        },
    ],
)
def test_NeuronDestInfo_instance_illegal(params):
    with pytest.raises(ValueError):
        dest_info = NeuronDestInfo.model_validate(params, strict=True)
