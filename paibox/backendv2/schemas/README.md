# backendv2 schemas

本目录保存 backendv2 编译产物 metadata 的手写 schema。生成代码不放在这里，而是由
`scripts/gen_backendv2_schemas.py` 写入 `paibox/backendv2/generated/`。

## Layout

```text
schemas/
  compile_artifacts.proto
  compile_artifacts.fbs
  README.md

generated/
  proto/compile_artifacts_pb2.py
  proto/compile_artifacts_pb2.pyi
  fbs/*.py
```

刷新生成代码：

```bash
python scripts/gen_backendv2_schemas.py --target all
```

可用 `--target proto` 或 `--target fbs` 只刷新其中一种格式。

## Exported Files

`Mapper.compile(..., target_platform="x86")` 导出：

```text
proto/config.pb
proto/config.json
proto/compile_artifacts.proto
proto/compile_artifacts_pb2.py
proto/compile_artifacts_pb2.pyi
```

`Mapper.compile(..., target_platform="riscv")` 导出：

```text
runtime/compile_artifacts.bin
runtime/compile_artifacts.fbs
```

`target_platform="all"` 或 `debug=True` 时两组 metadata 都会导出。

## Protobuf Path

`compile_artifacts.proto` 面向 x86 / Python 上位机和调试工具。顶层 message 是
`CompileArtifacts`，包含 `schema_version`、`target_board`、`io_mapping` 和全局唯一的
`config_frames`。每个 `ThreadIOMapping` 还可声明 `occupied_chip_count`；缺省值 `0`
表示旧产物未声明该信息。

本次字段扩展保持 `schema_version=1` 和旧字段语义不变。旧 reader 会忽略新增字段；
配置帧仍由根消息的 `config_frames` 统一保存和发送，不拆分到线程或芯片。

`proto/config.pb` 是机器接口；`proto/config.json` 只供人工检查。Python 调试程序可直接
复制导出的 `compile_artifacts_pb2.py/.pyi` 后读取 `config.pb`。

## FlatBuffers Path

`compile_artifacts.fbs` 面向 RISC-V runtime。root table 是 `CompileArtifacts`，
`file_identifier` 是 `PBCA`。导出的 `runtime/compile_artifacts.bin` 是同一份
compile artifacts metadata 的 FlatBuffers 编码。

FlatBuffers runtime 不要求文件系统，也不要求在 MCU 上构建 buffer；只读路径直接解析
`runtime/compile_artifacts.bin` 并按现有全局配置帧流发送。

## Data Flow

后端只构造一次 schema-neutral `CompileArtifactsData`，然后分流为 protobuf 或
FlatBuffers：

```text
Mapper state
  -> CompileArtifactsData
    -> protobuf emitter -> proto/config.pb
    -> FlatBuffers emitter -> runtime/compile_artifacts.bin

Multiple subnets
  -> existing compile/artifact flow, with one global configuration-frame stream
```
