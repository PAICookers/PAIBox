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

| 字段             | 含义                                                                                   |
| ---------------- | -------------------------------------------------------------------------------------- |
| `schema_version` | schema 版本。当前 backendv2 proto 仍为 `1`；应用侧可用它判断当前程序是否支持该 `.pb`。 |
| `io_mapping`     | 逻辑 I/O 张量与芯片工作帧地址之间的映射。                                              |
| `config_frames`  | 编译生成的配置帧，按 32-bit word 展平保存。                                            |

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
    RuntimeParams runtime = 3;
    InputTensorMappings input_mappings = 4;
    OutputTensorMappings output_mappings = 5;
    repeated CoreTick core_ticks = 6;
}
```

| 字段               | 含义                                                                             |
| ------------------ | -------------------------------------------------------------------------------- |
| `thread_id`        | 硬件线程编号。一次编译若包含多个网络或多个线程域，可导出多个 `ThreadIOMapping`。 |
| `root_core_offset` | 该线程内全局信号 root core 的相对偏移。                                          |
| `input_mappings`   | 输入逻辑张量到输入工作帧地址的映射。                                             |
| `output_mappings`  | 输出工作帧地址到输出逻辑张量的映射。                                             |
| `core_ticks`       | 该线程内真实计算核的 tick 明细；不包含全局信号空核。                             |
| `runtime`          | 应用侧运行时序摘要，用于同步步数控制和编解码参数校验。                           |

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

message TickParams {
    optional uint32 tick_start = 1;
    optional uint32 tick_duration = 2;
    optional uint32 tick_initial = 3;
}

message RuntimeParams {
    enum DecodeMode {
        STREAM = 0;
        STEP = 1;
    }

    optional uint32 timesteps = 1;
    optional uint32 tick_depth = 2;
    optional uint32 sync_steps = 3;
    optional DecodeMode decode_mode = 4;
}

/* DATA payload code type. VOLTAGE outputs leave dtype unset and are int32. */
message DataType {
    enum Code {
        NOT_SET = 0;
        UINT1 = 1;
        INT1 = 2;
        UINT2 = 3;
        INT2 = 4;
        UINT4 = 5;
        INT4 = 6;
        UINT8 = 7;
        INT8 = 8;
    }
}

message CoreTick {
    CoreOffset core_offset = 1;
    TickParams tick = 2;
    repeated string nodes = 3;
}
```

`CoreOffset` 表示目标 core 的相对偏移；`CopyCount` 表示 AER 多播复制数量。二者都使用 2.5芯片帧格式中的 `XY/X/Y` 三轴概念，但语义不同：`core_offset` 表示目标位置，`copy_count` 表示复制范围。

`TickParams` 对应 2.5 计算核的 `tick_start/tick_duration/tick_initial` 内部硬件参数；它不同于前端公开编译参数 `timesteps`。`tick_duration=0` 表示持续工作；`tick_initial=0` 表示不自动复位。`CoreTick.tick` 是该物理计算核的 tick 参数，`CoreTick.nodes` 是部署到同一个物理计算核上的 PAIIR 节点名列表。

`RuntimeParams.timesteps` 是应用推理序列长度。后端未显式接收 `Mapper.compile(..., timesteps=...)` 时，会优先从自动复位图的 `tick_initial` 推导；不能在 `auto_reset=True` 场景依赖 `tick_duration>0` 推导运行长度。`tick_depth` 是该 thread 输出 producer 的最大 `tick_start`；`sync_steps = tick_depth + timesteps - 1` 是推荐外部同步步数。`sync_steps` 只描述主机视角的同步控制长度，不参与输出层 `target_lcn` 选择；输出层 `target_lcn` 由实际输出 axon 地址容量反推，优先保留更多本地 timestep 位。输出帧中的 timestep 是输出层本地运行时步。`decode_mode=STREAM` 表示最终 `target_lcn` 的 timestep 位宽可区分运行时步；`STEP` 表示需要应用分步推理、分步解码，或只能进行 warning 级 best-effort 序列解码。

`DataType.Code` 描述普通 DATA payload 的码字类型。`UINT*` / `INT*` 中的数字表示逻辑位宽；`INT*` 按 two's complement 解释。`NOT_SET` 只作为默认值，应用侧不应把它当成有效 DATA 类型。`VOLTAGE` 输出不设置 `dtype`，固定按 `int32` 膜电平解释。

## 6. 输入映射与输入工作帧编码

```proto
message InputTensorMapping {
    string name = 1;
    Shape shape = 2;
    optional uint32 bit_width = 3;
    TickParams tick = 4;
    repeated InputEntry entries = 5;
}

message InputEntry {
    optional uint32 elem_idx = 1;
    CoreOffset core_offset = 2;
    CopyCount copy_count = 3;
    optional uint32 tick_relative = 4;
    optional uint32 addr_axon = 5;
    optional uint32 target_lcn = 6;
    optional uint32 copy_id = 7;
    optional DataType.Code dtype = 9;
}
```

| 字段            | 含义                                                                 |
| --------------- | -------------------------------------------------------------------- |
| `name`          | PAIIR 输入节点名。                                                   |
| `shape.size`    | 逻辑输入张量 shape。                                                 |
| `bit_width`     | 该输入张量所有 entries 共享的 payload 位宽。                         |
| `tick`          | 该输入张量对应的首个实际消费计算核 tick 参数。                       |
| `elem_idx`      | 输入张量按 C-order 展平后的元素下标。                                |
| `core_offset`   | 输入工作帧目标 core 的相对偏移。                                     |
| `copy_count`    | AER 多播复制数量。                                                   |
| `tick_relative` | 后端分配出的相对 tick 段。                                           |
| `addr_axon`     | 后端分配出的 axon 地址低段。                                         |
| `target_lcn`    | 目标 core 的 LCN 编号，对应 `paicorelib.LCN_EX` 枚举值。             |
| `copy_id`       | tiling/folding 产生的逻辑 copy 编号，不等同于 `CopyCount`。          |
| `dtype`         | 首个实际消费计算核解释该输入元素时使用的数据类型，包含位宽和符号性。 |

应用侧编码输入工作帧时，应按 `shape.size` 准备输入张量，并以 C-order 展平后使用 `elem_idx` 取值。

注意：`InputTensorMapping.tick` 是计算核工作窗口；`InputEntry.tick_relative` 是输入工作帧地址的一部分。二者语义不同，生成工作帧时仍使用 `tick_relative/addr_axon/target_lcn` 计算 timestep 和 axon。

`InputTensorMapping.bit_width` 是输入 payload 位宽的读取入口，字段顺序刻意放在 `tick` 和 `entries` 前，便于 JSON 中先看到张量级标量。应用应使用 mapping 级 `bit_width` 和 entry 级 `dtype` 决定输入值域与 signedness。

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
- 根据 `dtype` 把输入值转换为对应码字；例如 `INT8` 通常按 two's complement 视为 `uint8` 发送，可用 `x.astype(np.int8).view(np.uint8)`。
- 当前 work frame type 1 payload 为 8 bit；`bit_width` 描述逻辑元素位宽，`dtype` 描述该 payload 的 signedness 和有效位宽。

## 7. 输出映射与输出工作帧解码

```proto
message OutputTensorMappings {
    optional uint32 target_lcn = 1;
    repeated OutputTensorMapping items = 2;
}

message OutputTensorMapping {
    enum OutputKind {
        DATA = 0;
        VOLTAGE = 1;
    }

    string name = 1;
    Shape shape = 2;
    optional OutputKind kind = 3;
    optional uint32 bit_width = 4;
    TickParams tick = 5;
    repeated OutputEntry entries = 6;
}

message OutputEntry {
    optional uint32 elem_idx = 1;
    optional uint32 copy_id = 2;
    optional uint32 axon_bit_idx = 4;
    optional DataType.Code dtype = 5;
}
```

| 字段                         | 含义                                                                      |
| ---------------------------- | ------------------------------------------------------------------------- |
| `output_mappings.target_lcn` | 输出工作帧地址解析使用的目标 LCN 编号，对应 `paicorelib.LCN_EX` 枚举值。  |
| `name`                       | PAIIR 输出源/生产者节点名，不是虚拟 `OutputNode` 名。                     |
| `shape.size`                 | 逻辑输出张量 shape。                                                      |
| `kind`                       | 输出节点语义。`DATA` 表示普通激活值/脉冲数据，`VOLTAGE` 表示膜电平。      |
| `bit_width`                  | 该输出张量所有 entries 共享的 payload 位宽；`VOLTAGE` 固定为 32 bit。     |
| `tick`                       | 该输出张量的最终实际生产者计算核 tick 参数。                              |
| `elem_idx`                   | 输出张量按 C-order 展平后的元素下标。                                     |
| `copy_id`                    | tiling/folding 产生的逻辑 copy 编号。                                     |
| `axon_bit_idx`               | 平坦 output axon bit index。`DATA` 为数据地址；`VOLTAGE` 为膜电平基地址。 |
| `dtype`                      | `DATA` 输出的数据类型，包含位宽和符号性；`VOLTAGE` 不设置该字段。         |

CPU 接收端仍应先根据返回工作帧的帧头区分 I/II 型。`kind` 的作用是让应用侧在运行前从 `config.pb` 预生成静态解码表，并保留调试语义。不要用 `bit_width` 反推出输出语义；应以 `OutputTensorMapping.kind` 为准。`OutputTensorMapping.bit_width` 是位宽读取入口，字段顺序刻意放在 `tick` 和 `entries` 前，便于 JSON 中先看到张量级标量；`DATA` 输出用 entry 级 `dtype` 解释 signedness；`VOLTAGE` 输出固定按 `int32` 膜电平解释。

`output_mappings.target_lcn` 选择策略与上游 backendv2 保持一致：后端先分配实际输出 axon 地址，再根据最大 axon bit 反推可容纳这些地址的最小 LCN。应用传入的 `timesteps` 不直接扩大 `target_lcn`；若所选 LCN 的 timestep 位宽不足以一次性区分全部运行时步，导出的 `RuntimeParams.decode_mode` 会变为 `STEP`。

应用侧可直接使用 `paibox.backendv2.proto.runtime_codec.FrameCodec` 读取 `config.pb` 或 `config.json`。实时路径推荐读取 `.pb`，并在初始化时预生成输入编码表和输出 scatter 表。`FrameCodec.from_file(..., thread_id=0)` 默认选择硬件线程 0；多线程或非 0 线程产物需显式传入 `thread_id`。单输入模型可直接传入 array-like；多输入模型必须传入 `{input_name: array_like}`。编码输入可以是 NumPy array，也可以是 torch tensor；torch tensor 会通过 `detach().cpu().numpy()` 转为 NumPy。`codec.encode(inputs)` 默认要求输入形状为 `[T, *input_shape]`；`codec.encode(inputs, t=t)` 编码单个运行时步。`codec.decode(frames)` 的 `frames` 固定要求为 `np.ndarray` 且 `dtype=np.uint64`，在 `STREAM` 模式下返回 `[T, *output_shape]`，未输出位置置 0；多输出模型返回 `{output_name: ndarray}`。`STEP` 模式推荐 `codec.decode(frames, t=t)` 分步解码。

`OutputNode` 是图输出的虚拟边界节点，不是实际计算核。`OutputTensorMapping.name` 和 `shape` 使用最终输出源/生产者节点，`tick` 从该输出源追溯到实际生产计算核后导出。

`kind` 是 `OutputTensorMapping` 级字段；当前后端要求同一个输出源/生产者节点内所有 `entries` 共享同一种输出语义。

普通 `DATA` 输出对应工作帧 I 型。解码流程：

1. 从 `mapping.kind == DATA` 的 `OutputTensorMapping.entries` 建立 `axon_bit_idx -> elem_idx/dtype` 表。
2. 板端运行时先把芯片返回帧解析为 `(axon_bit_idx, payload_byte)`。
3. 根据映射表把 payload 写回输出张量的展平位置 `elem_idx`。
4. 根据 `dtype` 把 payload 解释为 signed/unsigned 1/2/4/8-bit 码字。

膜电平 `VOLTAGE` 输出对应工作帧 II 型。后端分配时为每个神经元膜电平保留 `base + {0, 8, 16, 24}` 四个内部 byte lane，但 proto 只记录基地址 `axon_bit_idx`。运行时返回的 4 个 type-II 帧使用同一个基地址作为 axon，payload 按低字节到高字节顺序出现。`FrameCodec` 会按 `(output_name, runtime_t, elem_idx)` 收集同一基地址上的 4 个 payload byte，并用 little-endian signed `int32` 还原膜电平；若缺 byte，则 warning 后补 0。由于返回帧本身不携带 byte-lane 编号，应用侧也需要保持芯片/运行时输出顺序。

如果应用侧需要主动构造工作帧 II 型，可把每个膜电平元素拆成一个 `int32`，再调用 `OfflineFrameGenV2.gen_work_frame2(...)`。该接口会为每个膜电平值自动展开 4 个 byte lane，并生成 4 帧 64-bit work frame type 2。下面的示例未经过板端流程验证，仅供实现参考：

```python
import numpy as np
from paicorelib import AERPacketZXYCopy, CoordZXYOffset, OfflineFrameGenV2


def encode_voltage_frames(
    coord_offset: CoordZXYOffset,
    target_lcn: int,
    timesteps: np.ndarray,
    base_axons: np.ndarray,
    voltage: np.ndarray,
):
    return OfflineFrameGenV2.gen_work_frame2(
        coord_offset,
        AERPacketZXYCopy(0, 0, 0),
        timesteps,
        base_axons,
        target_lcn,
        voltage.astype(np.int32, copy=False),
    )
```

下面的 Python 代码未经过板端流程验证，仅供实现参考。实际应用应以板端返回帧格式和运行时 ABI 为准。

```python
import numpy as np

from compile_artifacts_pb2 import CompileArtifacts


def build_output_tables(artifacts: CompileArtifacts):
    tables = {}
    for thread in artifacts.io_mapping.threads:
        target_lcn = int(thread.output_mappings.target_lcn)
        for mapping in thread.output_mappings.items:
            shape = tuple(mapping.shape.size)
            bit_width = int(mapping.bit_width)
            data_by_axon = {}
            voltage_by_base = {}
            for entry in mapping.entries:
                item = (
                    int(entry.elem_idx),
                    int(entry.copy_id),
                    bit_width,
                    int(entry.dtype) if entry.HasField("dtype") else 0,
                )
                if mapping.kind == mapping.DATA:
                    data_by_axon[int(entry.axon_bit_idx)] = item
                elif mapping.kind == mapping.VOLTAGE:
                    voltage_by_base[int(entry.axon_bit_idx)] = item
            tables[mapping.name] = (shape, target_lcn, data_by_axon, voltage_by_base)
    return tables


def scatter_u8_output(output_tables, output_name: str, decoded_items):
    shape, _, by_axon, _ = output_tables[output_name]
    output = np.zeros(int(np.prod(shape)), dtype=np.uint8)

    for axon_bit_idx, payload in decoded_items:
        if axon_bit_idx not in by_axon:
            continue
        elem_idx, copy_id, bit_width, dtype = by_axon[axon_bit_idx]
        output[elem_idx] = payload & 0xFF

    return output.reshape(shape)
```

如果板端返回的是 64-bit offline work frame type 1 原始帧，可先解析出 `(axon_bit_idx, payload)`，再执行 scatter。输出 work frame 的 timestep / axon 宽度应使用 `thread.output_mappings.target_lcn` 解析。

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

对于 `mapping.kind == VOLTAGE` 的输出，一个 `OutputEntry` 的 `axon_bit_idx` 是膜电平基地址。应用侧需要按帧顺序收集同一基地址上的 4 个 payload byte，再按 little-endian signed `int32` 组合：

```python
def collect_i32_le(payloads: list[int]) -> int:
    if len(payloads) < 4:
        payloads = [*payloads, *([0] * (4 - len(payloads)))]
    return int.from_bytes(bytes(payloads[:4]), byteorder="little", signed=True)
```

例如，若 `OfflineFrameGenV2.gen_work_frame2(...)` 对某个膜电平 `0x11223344` 生成的 4 帧 payload 依次为 `0x44, 0x33, 0x22, 0x11`，则应用侧应按 little-endian 次序把它们还原回 `0x11223344`。若 payload 依次为 `0xFE, 0xFF, 0xFF, 0xFF`，则还原为 `-2`。

## 8. JSON 字段名

`config.json` 使用 protobuf JSON 命名规则，会把 snake_case 字段转成 lowerCamelCase：

| proto 字段         | JSON 字段        |
| ------------------ | ---------------- |
| `schema_version`   | `schemaVersion`  |
| `io_mapping`       | `ioMapping`      |
| `config_frames`    | `configFrames`   |
| `root_core_offset` | `rootCoreOffset` |
| `runtime`          | `runtime`        |
| `tick_depth`       | `tickDepth`      |
| `sync_steps`       | `syncSteps`      |
| `decode_mode`      | `decodeMode`     |
| `bit_width`        | `bitWidth`       |
| `elem_idx`         | `elemIdx`        |
| `addr_axon`        | `addrAxon`       |
| `axon_bit_idx`     | `axonBitIdx`     |
| `target_lcn`       | `targetLcn`      |

应用程序读取 `config.pb` 时使用 proto 字段名；人工查看 `config.json` 时使用 JSON 字段名。`InputTensorMapping.bit_width` 和 `OutputTensorMapping.bit_width` 是 mapping 级标量，字段顺序放在 `tick` 与 `entries` 前，便于人工查看 JSON；应用侧仍应按字段名读取，不应依赖文本顺序。
