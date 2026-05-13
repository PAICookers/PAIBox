# backendv2 proto 手册

本文档说明 `backendv2` 导出的 `proto/config.pb` 的数据结构和应用侧消费方式。目标读者是需要下发配置帧、编码输入工作帧、解析输出工作帧的应用开发人员。

本文档只描述 `compile_artifacts.proto` 当前已经承诺的内容。板端 DMA、UART、PCIe、NoC FIFO、同步控制等运行时协议不属于本 proto 的接口范围。

## 1. 产物文件

`Mapper.compile(...)` 会在导出目录下生成 `proto/` 子目录：

```text
output/
  proto/
    config.pb
    config.json
    compile_artifacts.proto
    compile_artifacts_pb2.py
    compile_artifacts_pb2.pyi
```

| 文件                        | 用途                                                                    |
| --------------------------- | ----------------------------------------------------------------------- |
| `config.pb`                 | 二进制 protobuf，应用程序应优先读取此文件。                             |
| `config.json`               | `config.pb` 的 JSON 展开，供人工检查和调试。                            |
| `compile_artifacts.proto`   | schema 文件。                                                           |
| `compile_artifacts_pb2.py`  | Python 生成代码，仅在 x86 导出且开启 `export_proto_python` 时复制。     |
| `compile_artifacts_pb2.pyi` | Python 类型标注文件，仅在 x86 导出且开启 `export_proto_python` 时复制。 |

正式程序不要把 `config.json` 作为机器接口；它只用于调试和人工核对。

## 2. 读取 `config.pb`

Python 应用可以把 `compile_artifacts_pb2.py` 和 `compile_artifacts_pb2.pyi` 复制到自己的工程目录，再按普通模块导入。

推荐目录形态：

```text
app/
  compile_artifacts_pb2.py
  compile_artifacts_pb2.pyi
  load_config.py
```

读取示例：

```python
from pathlib import Path

from compile_artifacts_pb2 import CompileArtifacts


def load_compile_artifacts(pb_path: str | Path) -> CompileArtifacts:
    artifacts = CompileArtifacts()
    artifacts.ParseFromString(Path(pb_path).read_bytes())
    return artifacts


artifacts = load_compile_artifacts("output/proto/config.pb")
```

非 Python 应用应使用 `compile_artifacts.proto` 为目标语言生成代码。例如 C++：

```bash
protoc --cpp_out=. compile_artifacts.proto
```

## 3. 顶层结构

`config.pb` 的顶层 message 是 `CompileArtifacts`：

```proto
message CompileArtifacts {
    uint32 schema_version = 1;
    IOMapping io_mapping = 2;
    ConfigFrames config_frames = 3;
}
```

| 字段             | 含义                                                    |
| ---------------- | ------------------------------------------------------- |
| `schema_version` | schema 版本。应用侧可用它判断当前程序是否支持该 `.pb`。 |
| `io_mapping`     | 逻辑 I/O 张量与芯片工作帧地址之间的映射。               |
| `config_frames`  | 编译生成的配置帧，按 32-bit word 展平保存。             |

## 4. 配置帧

```proto
message ConfigFrames {
    enum WordOrder {
        HIGH_FIRST = 0;
        LOW_FIRST = 1;
    }

    repeated uint32 words = 1;
    WordOrder word_order = 2;
}
```

`words` 是由 64-bit 配置帧拆分得到的 32-bit word 序列。`word_order` 表示每个 64-bit 配置帧内部两个 word 的排列顺序：

| `word_order` | `words` 排列                                                      |
| ------------ | ----------------------------------------------------------------- |
| `HIGH_FIRST` | `[frame0.high32, frame0.low32, frame1.high32, frame1.low32, ...]` |
| `LOW_FIRST`  | `[frame0.low32, frame0.high32, frame1.low32, frame1.high32, ...]` |

如果应用侧需要从 `config.pb` 还原 64-bit 配置帧，可使用如下逻辑。该示例未经过板端流程验证，仅供实现参考：

```python
from compile_artifacts_pb2 import ConfigFrames


def iter_config_frame_u64(config_frames: ConfigFrames):
    words = list(config_frames.words)
    if len(words) % 2 != 0:
        raise ValueError("config_frames.words must contain an even number of words")

    for first, second in zip(words[0::2], words[1::2]):
        if config_frames.word_order == ConfigFrames.HIGH_FIRST:
            high32, low32 = first, second
        else:
            low32, high32 = first, second
        yield (int(high32) << 32) | int(low32)
```

如果应用侧已经使用 `cfg_frame*.h` 或 `cfg_frames.npy` 下发配置帧，通常不需要再从 `config.pb` 还原配置帧。

## 5. I/O 映射

```proto
message IOMapping {
    repeated ThreadIOMapping threads = 1;
}

message ThreadIOMapping {
    optional uint32 thread_id = 1;
    CoreOffset root_core_offset = 2;
    InputTensorMappings input_mappings = 3;
    OutputTensorMappings output_mappings = 4;
}
```

| 字段               | 含义                                                                             |
| ------------------ | -------------------------------------------------------------------------------- |
| `thread_id`        | 硬件线程编号。一次编译若包含多个网络或多个线程域，可导出多个 `ThreadIOMapping`。 |
| `root_core_offset` | 该线程内全局信号 root core 的相对偏移。                                          |
| `input_mappings`   | 输入逻辑张量到输入工作帧地址的映射。                                             |
| `output_mappings`  | 输出工作帧地址到输出逻辑张量的映射。                                             |

`CoreOffset` 和 `CopyCount` 都包含 `xy/x/y` 三个分量：

```proto
message CoreOffset {
    optional int32 xy = 1;
    optional int32 x = 2;
    optional int32 y = 3;
}

message CopyCount {
    optional int32 xy = 1;
    optional int32 x = 2;
    optional int32 y = 3;
}
```

`CoreOffset` 表示目标 core 的相对偏移；`CopyCount` 表示 AER 多播复制数量。二者都使用 2.5芯片帧格式中的 `XY/X/Y` 三轴概念，但语义不同：`core_offset` 表示目标位置，`copy_count` 表示复制范围。

## 6. 输入映射与输入工作帧编码

```proto
message InputTensorMapping {
    string name = 1;
    Shape shape = 2;
    repeated InputEntry entries = 3;
}

message InputEntry {
    optional uint32 elem_idx = 1;
    CoreOffset core_offset = 2;
    CopyCount copy_count = 3;
    optional uint32 tick_relative = 4;
    optional uint32 addr_axon = 5;
    optional uint32 target_lcn = 6;
    optional uint32 copy_id = 7;
    optional uint32 bit_width = 8;
}
```

| 字段            | 含义                                                        |
| --------------- | ----------------------------------------------------------- |
| `name`          | PAIIR 输入节点名。                                          |
| `shape.size`    | 逻辑输入张量 shape。                                        |
| `elem_idx`      | 输入张量按 C-order 展平后的元素下标。                       |
| `core_offset`   | 输入工作帧目标 core 的相对偏移。                            |
| `copy_count`    | AER 多播复制数量。                                          |
| `tick_relative` | 后端分配出的相对 tick 段。                                  |
| `addr_axon`     | 后端分配出的 axon 地址低段。                                |
| `target_lcn`    | 目标 core 的 LCN 编号，对应 `paicorelib.LCN_EX` 枚举值。    |
| `copy_id`       | tiling/folding 产生的逻辑 copy 编号，不等同于 `CopyCount`。 |
| `bit_width`     | 该逻辑输入元素的位宽。                                      |

应用侧编码输入工作帧时，应按 `shape.size` 准备输入张量，并以 C-order 展平后使用 `elem_idx` 取值。

编码流程：

1. 选择目标 `ThreadIOMapping` 和 `InputTensorMapping`。
2. 将输入张量展平为 `flat_input`。
3. 对每个 `InputEntry`，取 `flat_input[elem_idx]` 作为 payload。
4. 用 `core_offset` 构造工作帧目标偏移。
5. 用 `copy_count` 构造 AER 多播复制数量。
6. 用 `tick_relative`、`addr_axon`、`target_lcn` 得到工作帧中的 timestep 和 axon。
7. 生成 offline work frame type 1。

下面的 Python 代码未经过板端流程验证，仅供实现参考。实际应用应以板端运行时 ABI 和当前 `paicorelib` 版本为准。

```python
from collections import defaultdict
from pathlib import Path

import numpy as np
from paicorelib import AERPacketZXYCopy, CoordZXYOffset, LCN_EX, OfflineFrameGenV2

from compile_artifacts_pb2 import CompileArtifacts

FANIN_BASE = 512


def load_compile_artifacts(pb_path: str | Path) -> CompileArtifacts:
    artifacts = CompileArtifacts()
    artifacts.ParseFromString(Path(pb_path).read_bytes())
    return artifacts


def split_input_address(entry):
    _, ax_width = OfflineFrameGenV2.LCN_TO_TS_AXON_WIDTHS[int(entry.target_lcn)]
    full_addr = int(entry.tick_relative) * FANIN_BASE + int(entry.addr_axon)
    timestep = full_addr >> ax_width
    axon = full_addr & ((1 << ax_width) - 1)
    return timestep, axon


def encode_input_tensor(artifacts: CompileArtifacts, input_name: str, data) -> np.ndarray:
    flat = np.asarray(data).reshape(-1)
    frame_parts = []

    for thread in artifacts.io_mapping.threads:
        for mapping in thread.input_mappings.items:
            if mapping.name != input_name:
                continue

            groups = defaultdict(list)
            for entry in mapping.entries:
                key = (
                    entry.core_offset.xy,
                    entry.core_offset.x,
                    entry.core_offset.y,
                    entry.copy_count.xy,
                    entry.copy_count.x,
                    entry.copy_count.y,
                    entry.target_lcn,
                )
                groups[key].append(entry)

            for key, entries in groups.items():
                core_xy, core_x, core_y, copy_xy, copy_x, copy_y, target_lcn = key
                entries = sorted(entries, key=lambda item: item.elem_idx)

                timesteps = []
                axons = []
                payload = []
                for entry in entries:
                    timestep, axon = split_input_address(entry)
                    timesteps.append(timestep)
                    axons.append(axon)
                    payload.append(flat[int(entry.elem_idx)])

                frames = OfflineFrameGenV2.gen_work_frame1(
                    CoordZXYOffset(int(core_xy), int(core_x), int(core_y)),
                    AERPacketZXYCopy(int(copy_xy), int(copy_x), int(copy_y)),
                    np.asarray(timesteps, dtype=np.uint64),
                    np.asarray(axons, dtype=np.uint64),
                    LCN_EX(int(target_lcn)),
                    np.asarray(payload, dtype=np.uint8),
                )
                frame_parts.append(frames)

    if not frame_parts:
        raise KeyError(f"input mapping not found: {input_name}")

    return np.concatenate(frame_parts).astype(np.uint64, copy=False)
```

注意事项：

- `OfflineFrameGenV2.gen_work_frame1(...)` 会跳过 payload 为 `0` 的元素；全零输入会得到空帧数组。
- 如果模型输入是 `int8`，通常应按 two's complement 视为 `uint8` 发送，例如 `x.astype(np.int8).view(np.uint8)`。
- 当前 work frame type 1 payload 为 8 bit；`bit_width` 描述逻辑元素位宽，具体码字解释仍由模型输入 ABI 决定。

## 7. 输出映射与输出工作帧解码

```proto
message OutputTensorMapping {
    string name = 1;
    Shape shape = 2;
    repeated OutputEntry entries = 3;
}

message OutputEntry {
    optional uint32 elem_idx = 1;
    optional uint32 copy_id = 2;
    optional uint32 bit_width = 3;
    optional uint32 axon_bit_idx = 4;
}
```

| 字段           | 含义                                               |
| -------------- | -------------------------------------------------- |
| `name`         | PAIIR 输出节点名。                                 |
| `shape.size`   | 逻辑输出张量 shape。                               |
| `elem_idx`     | 输出张量按 C-order 展平后的元素下标。              |
| `copy_id`      | tiling/folding 产生的逻辑 copy 编号。              |
| `bit_width`    | 输出元素位宽。                                     |
| `axon_bit_idx` | 后端为该输出元素分配的平坦 output axon bit index。 |

当前 `OutputEntry` 只能表示普通输出数据帧的地址标注与解码，即应用侧可通过 `axon_bit_idx` 把芯片返回的工作帧 payload 放回逻辑输出张量。它不能表示膜电平帧的地址信息，也不能描述膜电平帧的 4 帧基地址。因此，基于当前 schema，应用侧无法仅依赖 `config.pb` 完成膜电平帧定位或解码。

这点已按当前代码核对：`compile_artifacts.proto` 的 `OutputEntry` 只有 `elem_idx/copy_id/bit_width/axon_bit_idx`；`Mapper.export_proto(...)` 只从 `OutputAxonAllocator.axon_infos` 写入普通输出数据帧地址，没有写入膜电平帧字段或 `oneof` 输出位置类型。

普通输出数据帧的解码流程：

1. 从 `OutputTensorMapping.entries` 建立 `axon_bit_idx -> elem_idx` 表。
2. 板端运行时先把芯片返回帧解析为 `(axon_bit_idx, payload_byte)`。
3. 根据映射表把 payload 写回输出张量的展平位置 `elem_idx`。
4. 根据模型 ABI 解释 signedness、语义域和多 byte 组合方式。

下面的 Python 代码未经过板端流程验证，仅供实现参考。实际应用应以板端返回帧格式和运行时 ABI 为准。

```python
import numpy as np

from compile_artifacts_pb2 import CompileArtifacts


def build_output_tables(artifacts: CompileArtifacts):
    tables = {}
    for thread in artifacts.io_mapping.threads:
        for mapping in thread.output_mappings.items:
            shape = tuple(mapping.shape.size)
            by_axon = {}
            for entry in mapping.entries:
                by_axon[int(entry.axon_bit_idx)] = (
                    int(entry.elem_idx),
                    int(entry.copy_id),
                    int(entry.bit_width),
                )
            tables[mapping.name] = (shape, by_axon)
    return tables


def scatter_u8_output(output_tables, output_name: str, decoded_items):
    shape, by_axon = output_tables[output_name]
    output = np.zeros(int(np.prod(shape)), dtype=np.uint8)

    for axon_bit_idx, payload in decoded_items:
        if axon_bit_idx not in by_axon:
            continue
        elem_idx, copy_id, bit_width = by_axon[axon_bit_idx]
        output[elem_idx] = payload & 0xFF

    return output.reshape(shape)
```

如果板端返回的是 64-bit offline work frame type 1 原始帧，可先解析出 `(axon_bit_idx, payload)`，再执行 scatter。当前 backendv2 输出组使用 `LCN_128X` 的 output axon 空间；若未来 schema 增加输出 LCN 字段，应以 schema 字段为准。

下面的解析代码未经过板端流程验证，仅供实现参考：

```python
from paicorelib import LCN_EX, OfflineFrameGenV2


def decode_offline_work_frame1_u64(frame: int, target_lcn: int = LCN_EX.LCN_128X.value):
    ts_width, ax_width = OfflineFrameGenV2.LCN_TO_TS_AXON_WIDTHS[int(target_lcn)]

    payload = frame & 0xFF
    axon = (frame >> 8) & ((1 << ax_width) - 1)

    ts_low_width = ts_width - 1
    ts_low = (frame >> (8 + ax_width)) & ((1 << ts_low_width) - 1)
    ts_high = (frame >> 60) & 0x1
    timestep = (ts_high << ts_low_width) | ts_low
    axon_bit_idx = (timestep << ax_width) | axon

    return axon_bit_idx, payload
```

对于 `bit_width <= 8` 的输出，通常一个 `OutputEntry` 对应一个 payload byte。对于 `bit_width == 32` 的输出，当前后端会为同一个逻辑输出元素保留 `axon_bit_idx + 8 * i` 这四个 byte lane；应用侧需要收集四个 byte 后再按模型 ABI 组合为 32-bit 值。下面的 little-endian 组合逻辑未经过板端流程验证，仅供实现参考：

```python
def collect_u32_le(decoded_by_axon: dict[int, int], base_axon_bit_idx: int) -> int:
    value = 0
    for i in range(4):
        value |= (decoded_by_axon.get(base_axon_bit_idx + 8 * i, 0) & 0xFF) << (8 * i)
    return value
```

## 8. JSON 字段名

`config.json` 使用 protobuf JSON 命名规则，会把 snake_case 字段转成 lowerCamelCase：

| proto 字段         | JSON 字段        |
| ------------------ | ---------------- |
| `schema_version`   | `schemaVersion`  |
| `io_mapping`       | `ioMapping`      |
| `config_frames`    | `configFrames`   |
| `root_core_offset` | `rootCoreOffset` |
| `elem_idx`         | `elemIdx`        |
| `addr_axon`        | `addrAxon`       |
| `axon_bit_idx`     | `axonBitIdx`     |

应用程序读取 `config.pb` 时使用 proto 字段名；人工查看 `config.json` 时使用 JSON 字段名。
