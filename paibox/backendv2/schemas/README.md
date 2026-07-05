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
`CompileArtifacts`，包含 `schema_version`、`io_mapping` 和 `config_frames`。

`proto/config.pb` 是机器接口；`proto/config.json` 只供人工检查。Python 调试程序可直接
复制导出的 `compile_artifacts_pb2.py/.pyi` 后读取 `config.pb`。

## FlatBuffers Path

`compile_artifacts.fbs` 面向 RISC-V runtime。root table 是 `CompileArtifacts`，
`file_identifier` 是 `PBCA`。导出的 `runtime/compile_artifacts.bin` 是同一份
compile artifacts metadata 的 FlatBuffers 编码。

板端只读路径：

1. 从 flash 或其他只读存储拿到连续 `uint8_t*` 和长度。
2. 调用 generated verifier，例如 `VerifyCompileArtifactsBuffer(...)`。
3. 调用 generated accessor，例如 `GetCompileArtifacts(...)`。

FlatBuffers runtime 不要求文件系统，也不要求在 MCU 上构建 buffer；v0 只要求 MCU 端
zero-copy 读取和校验。

## Data Flow

后端只构造一次 schema-neutral `CompileArtifactsData`，然后分流为 protobuf 或
FlatBuffers：

```text
Mapper state
  -> CompileArtifactsData
    -> protobuf emitter -> proto/config.pb
    -> FlatBuffers emitter -> runtime/compile_artifacts.bin
```
