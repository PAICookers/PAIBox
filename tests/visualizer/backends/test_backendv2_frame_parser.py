import numpy as np
import pytest
from paicorelib import AERPacketZXYCopy, CoordXY, CoordZXYOffset, OfflineFrameGenV2
from paicorelib.framelib.frame_defs import FFV2, FrameHeader
from paicorelib.framelib.parser_v2 import (
    FrameParseError,
    decode_core_config,
    parse_frame_stream,
)

from paibox.visualizer.backends.v2.frame_parser import chip_core_role

from ..helpers import make_core_frame


def test_chip_core_roles() -> None:
    assert chip_core_role(0, 0) == "cpu"
    assert chip_core_role(1, 0) == "online"
    assert chip_core_role(8, 1) == "online"
    assert chip_core_role(0, 2) == "offline"


def test_parse_config_frame1_core_config() -> None:
    frames = make_core_frame(
        CoordXY(3, 1),
        global_send=(1 << 6) | (1 << 3),
        global_receive=1 << 2,
    )
    parsed = parse_frame_stream(frames)
    assert parsed.frame_count == 4
    assert (3, 1) in parsed.cores

    core = parsed.cores[(3, 1)]
    config = decode_core_config(core.frame_type1_payloads)
    assert config["neuron_number"] == 4
    assert config["thread_number"] == 3
    assert config["tick_start"] == 2
    assert config["tick_duration"] == 7
    assert config["tick_initial"] == 5
    assert config["global_send"] == (1 << 6) | (1 << 3)
    assert config["global_receive"] == 1 << 2


def test_parse_lut_and_frame3_package_summaries() -> None:
    offset = CoordZXYOffset(0, 2, 2)
    lut = OfflineFrameGenV2.gen_config_frame2(
        offset,
        np.arange(256, dtype=np.int32),
        np.arange(256, dtype=np.int8),
    )
    frame3 = OfflineFrameGenV2.gen_config_frame3_pkg_header(
        offset,
        start_addr=7,
        n_package=2,
        pkt_ncopy=AERPacketZXYCopy(),
    )
    frame3 = np.concatenate([frame3, np.array([0x11, 0x22], dtype=np.uint64)])

    parsed = parse_frame_stream(np.concatenate([lut, frame3]))
    core = parsed.cores[(2, 2)]

    assert len(core.packages) == 2
    assert core.packages[0].frame_type == 2
    assert core.packages[0].package_count == 256
    assert core.packages[1].frame_type == 3
    assert core.packages[1].start_addr == 7
    assert core.packages[1].package_count == 2


def test_invalid_lut_package_fails() -> None:
    offset = CoordZXYOffset(0, 2, 2)
    header = OfflineFrameGenV2.make_package(
        FrameHeader.CONFIG_TYPE2,
        offset,
        AERPacketZXYCopy(),
        0,
        np.array([0x1, 0x2], dtype=np.uint64),
    )

    with pytest.raises(FrameParseError, match="config frame type2 LUT"):
        parse_frame_stream(header)


def test_unsupported_frame_header_fails() -> None:
    bad = np.array(
        [int(FrameHeader.WORK_TYPE1) << FFV2.GENERAL_HEADER_OFFSET], dtype=np.uint64
    )
    with pytest.raises(FrameParseError, match="unsupported frame header"):
        parse_frame_stream(bad)
