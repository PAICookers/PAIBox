from pathlib import Path

import numpy as np
from paicorelib import CoordXY, OfflineFrameGenV2, find_coordxy_shortest_path

from paibox.backendv2.core_config import (
    Auto_Core_Config,
    Backend_Core_Config,
    Default_Core_Config,
    Frontend_Core_Config,
    to_core_reg,
)
from paibox.backendv2.generated.proto.compile_artifacts_pb2 import (
    CompileArtifacts,
    ConfigFrames,
)


def endpoint_map(app: object) -> dict[str, object]:
    """Return registered endpoints keyed by path for direct API tests."""
    return {
        route.path: route.endpoint
        for route in app.routes
        if hasattr(route, "endpoint")
    }


def make_core_frame(
    coord: CoordXY,
    *,
    neuron_number: int = 4,
    global_send: int = 0,
    global_receive: int = 0,
    test_core_xy: int | None = None,
    test_core_x: int | None = None,
    test_core_y: int | None = None,
) -> np.ndarray:
    default_test_offset, _ = find_coordxy_shortest_path(CoordXY(0, 0), coord)
    core_reg = to_core_reg(
        Default_Core_Config(thread_number=3),
        Auto_Core_Config(
            neuron_number=neuron_number,
            test_core_xy=(
                default_test_offset.z if test_core_xy is None else test_core_xy
            ),
            test_core_x=(default_test_offset.x if test_core_x is None else test_core_x),
            test_core_y=(default_test_offset.y if test_core_y is None else test_core_y),
            global_send=global_send,
            global_receive=global_receive,
        ),
        Backend_Core_Config(),
        Frontend_Core_Config(tick_start=2, tick_duration=7, tick_initial=5),
        coord,
    )
    offset, _ = find_coordxy_shortest_path(coord)
    return OfflineFrameGenV2.gen_config_frame1(offset, core_reg)


def write_pb(
    path: Path,
    frames: np.ndarray,
    *,
    root_core_offset: tuple[int, int, int] | None = None,
    word_order: int = ConfigFrames.HIGH_FIRST,
) -> Path:
    artifacts = CompileArtifacts()
    artifacts.schema_version = 1
    thread = artifacts.io_mapping.threads.add()
    thread.thread_id = 3
    if root_core_offset is not None:
        thread.root_core_offset.xy = root_core_offset[0]
        thread.root_core_offset.x = root_core_offset[1]
        thread.root_core_offset.y = root_core_offset[2]
    thread.runtime.timesteps = 20
    thread.runtime.tick_depth = 2
    thread.runtime.sync_steps = 21
    thread.runtime.decode_mode = thread.runtime.STREAM
    core_tick = thread.core_ticks.add()
    core_tick.core_offset.xy = 1
    core_tick.core_offset.x = 2
    core_tick.core_offset.y = 0
    core_tick.tick.tick_start = 2
    core_tick.tick.tick_duration = 7
    core_tick.tick.tick_initial = 5
    core_tick.nodes.append("SequentialOp_0")
    artifacts.config_frames.word_order = word_order
    for frame in frames:
        value = int(frame)
        high, low = (value >> 32) & 0xFFFFFFFF, value & 0xFFFFFFFF
        if word_order == ConfigFrames.HIGH_FIRST:
            artifacts.config_frames.words.extend((high, low))
        else:
            artifacts.config_frames.words.extend((low, high))
    path.write_bytes(artifacts.SerializeToString())
    return path
