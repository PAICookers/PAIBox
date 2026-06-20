"""backendv2 export-boundary tests for PAICORE 2.5 ANN LUT SRAM data."""

import numpy as np
import torch
from paicorelib import CoordZXYOffset, DataSign, DataWidth, OfflineFrameGenV2, SNNMode
from paicorelib.framelib.frame_defs import OfflineConfigFrame2FormatV2

from paibox.backendv2.op_node import get_frontend_core_conf
from paibox.paiir.ir.calc_params import LutData, OfflineCoreParams
from paibox.paiir.ir.lut_activation import LutCustom, _lookup_hw_lut_data


def _low_range_identity_lut():
    levels = torch.arange(5, dtype=torch.int32)
    return LutCustom.from_intervals(levels, levels, output_signed=False)


def _assert_value_block(
    values: torch.Tensor, start: int, stop: int, value: int
) -> None:
    assert values[start:stop].tolist() == [value] * (stop - start)


def test_frontend_core_conf_accepts_hw_lut_sram_data() -> None:
    lut = _low_range_identity_lut()
    logical_lut = lut.logical_lut_data
    hw_lut_data = lut.to_hw_lut_data(
        output_data_sign=DataSign.UNSIGNED,
        output_width=DataWidth.WIDTH_4BIT,
    )
    core_params = OfflineCoreParams(snn_mode=SNNMode.ANN)
    core_params.tick_start = 1
    core_params.set_output_format((DataSign.UNSIGNED, DataWidth.WIDTH_4BIT))

    conf = get_frontend_core_conf(core_params, hw_lut_data)

    assert conf.hw_lut_data is hw_lut_data
    assert not torch.equal(conf.hw_lut_data.thresholds, logical_lut.thresholds)
    assert torch.equal(
        conf.hw_lut_data.thresholds[torch.tensor([16, 32, 48, 64])],
        torch.tensor([1, 2, 3, 4], dtype=torch.int32),
    )
    _assert_value_block(conf.hw_lut_data.values, 0, 16, 0)
    _assert_value_block(conf.hw_lut_data.values, 16, 32, 1)
    _assert_value_block(conf.hw_lut_data.values, 32, 48, 2)
    _assert_value_block(conf.hw_lut_data.values, 48, 64, 3)
    _assert_value_block(conf.hw_lut_data.values, 64, 256, 4)
    values, indices = _lookup_hw_lut_data(
        conf.hw_lut_data,
        DataWidth.WIDTH_4BIT,
        torch.tensor([1, 2, 3, 4], dtype=torch.int32),
    )
    assert indices.tolist() == [16, 32, 48, 64]
    assert values.tolist() == [1, 2, 3, 4]

    frames = OfflineFrameGenV2.gen_config_frame2(
        CoordZXYOffset(0, 0, 0),
        conf.hw_lut_data.thresholds.numpy(),
        conf.hw_lut_data.values.numpy(),
    )
    payloads = frames[1:]
    decoded_potentials = (
        ((payloads >> OfflineConfigFrame2FormatV2.POTENTIAL_OFFSET) & 0xFFFF_FFFF)
        .astype(np.uint32)
        .view(np.int32)
    )
    decoded_activations = (
        (payloads >> OfflineConfigFrame2FormatV2.ACTIVATION_OFFSET)
        & OfflineConfigFrame2FormatV2.ACTIVATION_MASK
    ).astype(np.uint8)
    decoded_lut = LutData(
        thresholds=torch.from_numpy(decoded_potentials.copy()),
        values=torch.from_numpy(decoded_activations.copy()),
        is_float=False,
    )

    decoded_values, decoded_indices = _lookup_hw_lut_data(
        decoded_lut,
        DataWidth.WIDTH_4BIT,
        torch.tensor([1, 2, 3, 4], dtype=torch.int32),
    )
    assert torch.equal(decoded_lut.thresholds, hw_lut_data.thresholds)
    assert torch.equal(decoded_lut.values, hw_lut_data.values)
    assert decoded_indices.tolist() == [16, 32, 48, 64]
    assert decoded_values.tolist() == [1, 2, 3, 4]
    _assert_value_block(decoded_lut.values, 0, 16, 0)
    _assert_value_block(decoded_lut.values, 16, 32, 1)
    _assert_value_block(decoded_lut.values, 32, 48, 2)
    _assert_value_block(decoded_lut.values, 48, 64, 3)
    _assert_value_block(decoded_lut.values, 64, 256, 4)
