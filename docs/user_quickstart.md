# PAIIR + backendv2 用户快速上手

本文档面向“已经拿到量化后模型，想在当前 PAIBox `dev` 分支上生成 2.5 芯片部署产物”的用户。

目标不是介绍训练或 QAT，而是把当前仓库里真正可用的工具链路径讲清楚：

```text
部署态 PyTorch 模型
  -> compile_to_paiir(...)
  -> PAIIRGraph
  -> Mapper().compile(...)
  -> cfg_frame*.txt / .npy / .h + proto/
  -> （如需要）work_frame1.h
```

这份 quickstart 以 `paibox.paiir` 和 `paibox.backendv2` 的公共接口为主线，应用样例只作为参考证据。

## 1. 这条工具链负责什么

当前 `dev` 分支里的 `PAIIR -> backendv2` 路径主要负责：

- 把部署态 PyTorch 模型编译成 `PAIIRGraph`
- 把 `PAIIRGraph` 下沉到 `backendv2`
- 导出芯片配置帧 `cfg_frame*.txt/.npy/.h`
- 导出 `proto/config.pb` 与便于人工查看的 `proto/config.json`
- 为应用侧生成输入工作帧和解析输出工作帧提供 `proto/config.pb` 中的 I/O 映射

它**不负责**：

- 训练、校准、QAT、本身的量化流程
- 任意 PyTorch 量化图的自动编译
- 板端工程里的 UART/NoC 运行时协议
- “前端编译成功”到“后端一定布线成功”的自动保证

更准确地说，当前工具链擅长的是：

- 你已经把模型整理成“部署态 PyTorch 模型”
- 你希望把它变成 `PAIIRGraph`、平台相关帧文件以及 `proto/` 产物

## 2. 部署前必须先理解的边界

### 2.1 `compile_to_paiir(...)` 接收的是“部署态模型”，不是任意量化图

当前前端降低器的主入口是：

```python
from paibox.paiir import compile_to_paiir
```

它能识别的是当前 lowering 代码支持的模块/算子表面，而不是任意 `torch.ao` 量化执行图。

因此，“量化后模型可以部署”在当前仓库里的实际含义是：

- 量化流程已经完成
- 你已经把需要部署的整数语义整理到一个**可 trace、可 lowering** 的 PyTorch 模型表面上
- 这个模型的 `forward(...)` 使用的是当前 lowering 能理解的模块和数据流

### 2.2 推荐把量化结果整理成这种部署表面

对于 ANN 量化模型，当前最稳妥的通用方式是：

- 保留标准 `nn.Conv1d` / `nn.Conv2d` / `nn.Linear` / `nn.MaxPool*` / `nn.AvgPool*` / `nn.AdaptiveMaxPool*` / `nn.AdaptiveAvgPool*`
- 或保留当前支持的 `spikingjelly.activation_based.layer.X` wrapper，注意这和原生 `torch.nn.X` 支持面是两类入口
- 或保留静态参数可解析的 `torch.nn.functional.conv1d/conv2d`
- 把部署用的整数权重、偏置、scale、LUT 等信息固化到模块参数或 buffer 中
- 把 requant / 激活整理成当前前端能识别的激活表面
  - 直接使用内建 `nn.ReLU` / `nn.Sigmoid` / `nn.Tanh` / `nn.Softsign`
  - 或使用 `paibox.paiir` 里的 LUT 激活
  - 或使用 `register_neuron(...)` 注册自定义 neuron / 激活模块
- 对自定义计算模块，使用 `register_module(...)` 显式转换到已有 canonical lowering 模块
- 移除训练态、运行时量化包裹器、部署中不需要的辅助分支

这也是为什么：

- 芯片硬件能力和 `docs/Support-Ops.md` 里列出的能力范围，通常**大于**
- 当前 `paibox.paiir.lowering.converter` 直接支持的 lowering 表面

### 2.3 当前 lowering 的常用公共入口

从代码接口看，当前高频公共入口是：

```python
from paibox.paiir import (
    ANNNodeV25,
    CompileConfig,
    IFNodeV25,
    LIFNodeV25,
    LutCustom,
    LutReLU,
    compile_to_paiir,
    register_module,
    register_neuron,
    torch_to_paiir,
)
from paibox.backendv2 import Mapper
```

其中：

- `compile_to_paiir(...)` 是推荐的一站式编译入口
- `torch_to_paiir(...)` 只做 1:1 lowering，不做完整融合和部署校验
- `register_module(...)` 把自定义计算模块转换成 lowering 已支持的 canonical 模块
- `register_neuron(...)` 把自定义 neuron / 激活模块转换成芯片 neuron 或 LUT
- `Mapper.compile(...)` 负责布线、放置和帧文件导出

## 3. 环境准备

### 3.1 推荐安装方式

```bash
git clone https://github.com/PAICookers/PAIBox.git
cd PAIBox

uv venv .venv
source .venv/bin/activate
uv sync --group dev
```

当前 `pyproject.toml` 的 `dev` 依赖组已经覆盖这条路径所需的核心依赖，包括：

- `torch`
- `spikingjelly`
- `paicorelib`
- `ortools`
- `numba`

### 3.2 快速检查

```bash
uv run python -c "import torch, spikingjelly, paicorelib, numba, paibox; print('ok')"
```

## 4. 当前前端通常能直接识别什么

根据 `paibox.paiir.lowering.converter`，当前常用 lowering 表面主要包括：

- 计算模块
  - `nn.Conv1d`
  - `nn.Conv2d`
  - `nn.Linear`
  - `nn.MaxPool1d`
  - `nn.MaxPool2d`
  - `nn.AvgPool1d`
  - `nn.AvgPool2d`
  - `nn.AdaptiveMaxPool1d`
  - `nn.AdaptiveMaxPool2d`
  - `nn.AdaptiveAvgPool1d`
  - `nn.AdaptiveAvgPool2d`
- 函数式卷积
  - `torch.nn.functional.conv1d`
  - `torch.nn.functional.conv2d`
  - 要求 weight / bias 等参数能从 buffer、常量或静态 tensor 表达式中解析出来
- SpikingJelly `activation_based.layer` wrapper
  - `layer.Conv1d`
  - `layer.Conv2d`
  - `layer.Linear`
  - `layer.MaxPool1d`
  - `layer.MaxPool2d`
  - `layer.AvgPool1d`
  - `layer.AvgPool2d`
  - `layer.AdaptiveAvgPool1d`
  - `layer.AdaptiveAvgPool2d`
  - `layer.Flatten`
  - 这些 wrapper 按当前 PAIIR 支持的普通模块路径 lowering；支持面与当前安装的 SpikingJelly `activation_based.layer` 模块实际提供的 wrapper 保持一致
- 常见激活
  - `nn.ReLU`
  - `nn.Sigmoid`
  - `nn.Tanh`
  - `nn.Softsign`
- 脉冲神经元
  - `spikingjelly.activation_based.neuron.IFNode`
  - `spikingjelly.activation_based.neuron.LIFNode`
- PAIIR 内建模块
  - `ANNNodeV25`
  - `IFNodeV25`
  - `LIFNodeV25`
  - `LutReLU`
  - `LutSigmoid`
  - `LutTanh`
  - `LutSoftsign`
  - `LutCustom`
- 形状和布局相关变换
  - `flatten`
  - 常见 reshape / layout 变换

此外：

- `nn.Dropout`、`nn.Identity` 会在 lowering 早期被擦除
- `layer.Dropout`、`layer.Dropout2d` 也会在 lowering 早期被擦除
- `nn.BatchNorm1d`、`nn.BatchNorm2d` 当前按 bypass 处理，不应把它们当成部署后仍需要的独立硬件语义
- 普通 `AvgPool1d/2d` 与 `MaxPool1d/2d` 当前不支持 `ceil_mode=True`；这类设置会直接报 `UnsupportedOpError`，不会因为 `strict=False` 被旁路

如果模型里出现 unsupported op：

- `strict=True` 时会抛 `UnsupportedOpError`
- `strict=False` 时会 warning 并旁路该节点

对于最终要落板的模型，建议最后回到 `strict=True`。

## 5. 编译前需要满足什么

### 5.1 模型约束

进入 `compile_to_paiir(...)` 前，建议至少满足：

- 已调用 `model.eval()`
- `forward(...)` 中不再依赖训练态行为
- 权重、偏置和激活语义已经是部署要用的版本
- 不再依赖运行时 `QuantStub/DeQuantStub` 这类纯软件包裹来表达芯片行为

### 5.2 `sample_inputs` 是部署契约的一部分

当前 `torch_to_paiir(...)` / `compile_to_paiir(...)` 会用 `sample_inputs` 做 shape propagation 和 dims propagation。

因此：

- 输入 shape 必须与真实部署输入一致
- 多输入模型按 `forward(self, xa, xb, ...)` 的顺序传入
- 每个 `sample_input` 的 `batch_size` 必须为 `1`

如果 batch size 不是 1，前端会直接报错。

### 5.3 当前对量化模型的通用建议

如果你手里的是 `torch.ao` 量化后的 checkpoint，不要默认认为可以直接：

```python
graph = compile_to_paiir(raw_quantized_model, sample_input)
```

更稳妥的通用做法是先做一层“部署态重建”：

1. 从量化模型中提取部署参数
   例如 `int8` 权重、`int32` 偏置、输入/输出 scale、requant 规则。
2. 用当前 lowering 能识别的模块重建一个部署态 `nn.Module`
3. 对不能直接表达的激活或 requant，使用 LUT 激活或 `register_neuron(...)`
4. 再把这个部署态模型送入 `compile_to_paiir(...)`

## 6. 最小可运行示例

下面给出一个通用的一站式示例：

- 编译到 `PAIIRGraph`
- 保存 `graph.summary()`
- 调用 `backendv2`
- 导出平台相关帧文件与 `proto/` 目录

```python
from contextlib import redirect_stdout
from pathlib import Path

import torch
import torch.nn as nn
from paicorelib import DataSign, DataWidth

from paibox.backendv2 import Mapper
from paibox.paiir import compile_to_paiir


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 16 * 16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


model = Model().eval()
sample_input = torch.zeros(1, 3, 32, 32)

build_dir = Path("build/my_model").resolve()
output_dir = build_dir / "output"
summary_path = build_dir / "paiir_summary.log"
backend_log_path = build_dir / "backendv2.log"
build_dir.mkdir(parents=True, exist_ok=True)

graph = compile_to_paiir(
    model,
    sample_input,
    input_formats={
        "InputNode_0": (DataSign.SIGNED, DataWidth.WIDTH_8BIT),
    },
    strict=True,
)

with summary_path.open("w", encoding="utf-8") as fp, redirect_stdout(fp):
    graph.summary()

mapper = Mapper()
with backend_log_path.open("w", encoding="utf-8") as fp, redirect_stdout(fp):
    mapper.compile(
        graph,
        output_path=str(output_dir),
        literal_format="hex",
        target_platform="x86",
        word_order="high_first",
        export_merged_frames=True,
        debug=True,
        export_proto_python=True,
    )

print("summary:", summary_path)
print("backend_log:", backend_log_path)
print("routing_groups:", len(mapper.routing_groups))
print("frame_dir:", output_dir)
```

推荐显式传 `output_path`，不要依赖当前工作目录。

## 7. `compile_to_paiir(...)` 常用参数

当前公共签名位于 `paibox.paiir.pipeline.compile.compile_to_paiir`。

高频参数如下：

| 参数                              | 作用                                                    |
| --------------------------------- | ------------------------------------------------------- |
| `*sample_inputs`                  | 示例输入，参与 shape/dims 推断，`batch_size` 必须为 `1` |
| `timesteps`                       | 一次样本/一次推理的时间步数，默认 `1`，必须为正整数     |
| `auto_reset`                      | 是否按 `timesteps` 自动复位，默认 `True`                |
| `input_formats`                   | 按 `InputNode` 名称指定输入数据格式                     |
| `compile_config`                  | 统一承载默认配置                                        |
| `concrete_args`                   | 固定 FX tracing 时的非 Tensor 参数                      |
| `strict`                          | 遇到 unsupported op 时是否直接报错                      |
| `enable_avgpool_calibration`      | 共享核 `AvgPool + LIF` 阈值细化开关                     |
| `enable_split_avgpool_lif`        | 条件式 `AvgPool + LIF` 分核部署开关                     |
| `enable_delayed_avgpool_division` | AvgPool 延迟除法改写开关，默认开启                      |
| `output_approx`                   | 输出层近似策略，默认 `"default"`                        |

优先级为：

```text
显式关键字参数 > CompileConfig > 内置默认值
```

默认 `timesteps=1`、`auto_reset=True`，不区分 ANN/SNN 模式。公开参数会映射到底层计算核 tick 字段：`auto_reset=True` 时导出 `tick_duration=0`、`tick_initial=timesteps`；`auto_reset=False` 时导出 `tick_duration=timesteps`、`tick_initial=0`。公开 API 不使用 `tick_duration=0` 表示推理长度，`tick_duration` 仅作为底层硬件字段出现在导出元数据中。

### 7.1 多输入模型

```python
graph = compile_to_paiir(
    model,
    sample_input_a,
    sample_input_b,
    strict=True,
)
```

### 7.2 固定 tracing 时的非 Tensor 参数

```python
graph = compile_to_paiir(
    model,
    sample_input,
    concrete_args={
        "deploy_mode": True,
        "return_aux": False,
    },
)
```

### 7.3 指定输入数据格式

```python
from paicorelib import DataSign, DataWidth

graph = compile_to_paiir(
    model,
    sample_input,
    input_formats={
        "InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT),
    },
)
```

如果你不确定 `InputNode` 的名字，先保存一次 `graph.summary()` 查看。

### 7.4 只用于排查兼容性的 `strict=False`

```python
graph = compile_to_paiir(model, sample_input, strict=False)
```

这只适合：

- 快速摸清有哪些 unsupported op
- 看结构、看 warning、做前端兼容性排查

不要把 `strict=False` 的返回图直接视为可部署图。

### 7.5 输出层 AvgPool 近似策略

默认 `output_approx="default"` 不改变标准输出策略。对于输出层 `AvgPool1d/2d`，如果默认路径需要把平均池化降级为多数脉冲输出，编译期会发出 `OutputApproxWarning`，提醒该输出已不再是原始浮点/整数平均值。

如果应用侧希望保留“计数”语义，可以显式启用：

```python
graph = compile_to_paiir(
    model,
    sample_input,
    output_approx="sum_approx_if_avgpool",
)
```

该策略只作用于直接流向图输出边界的 `AvgPool1d/2d`，包括由 SpikingJelly `VotingLayer` lowering 得到的 `AvgPool1d`。满足条件时，前端会把输出层平均池化导出为：

```text
AvgPool -> SumPool + identity LUT
```

这意味着芯片侧输出的是未除以 divisor 的 sum/count，而不是原始平均值。该 sum/count 的 `u1/u2/u4/u8 DATA` 位宽由 PAIIR 保守推断：只有 logical LUT 能被对应位宽的硬件 SAR LUT 精确表达，才会选择窄位宽；否则回退到更宽位宽。backendv2 只打包 IR 已生成的 `hw_lut_data`。应用侧 CPU 必须按任务语义继续处理：

- 若需要恢复平均值，除以 warning 中给出的 logical divisor
- 若分类任务只关心多 tick 累计后的 `argmax`，可直接累计 count 后再做 `argmax`

该策略当前要求输出层池化窗口完整、无 padding、输入为 VALUE 域且可推导精确 code range，并且 sum/count 范围能放入 8-bit VALUE 数据路径。策略生效时会发出 `OutputApproxWarning`，其中包含原节点、导出形式、导出范围、数据格式和 CPU 侧责任。当前 CPU 后处理责任只通过 warning 和文档表达，还没有作为结构化 CPU task 写入导出产物。

## 8. 自定义激活 / 自定义 neuron / 自定义模块

当模型里有当前 lowering 默认不认识的激活或 neuron 时，可先注册转换器：

```python
import torch
import torch.nn as nn

from paibox.paiir import LutCustom, register_neuron


class MyQuantAct(nn.Module):
    def __init__(self, thresholds: torch.Tensor, values: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("thresholds", thresholds.to(torch.int32))
        self.register_buffer("values", values.to(torch.int8))
        self.output_signed = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("runtime behavior omitted here")


register_neuron(
    MyQuantAct,
    lambda mod: LutCustom(mod.thresholds, mod.values, output_signed=mod.output_signed),
)
```

对于整数域的自定义分段激活，推荐用区间语义表达 logical LUT：

```python
# round(clamp(x, 0, 4))
lut = LutCustom.from_intervals(
    starts=torch.tensor([0, 1, 2, 3, 4]),
    values=torch.tensor([0, 1, 2, 3, 4]),
)
```

`starts[i]` 表示第 `i` 段的起始输入值，`values[i]` 表示该段输出值。PAIIR 会自动 padding 到 256 项 logical LUT，并在导出时尝试生成等价的硬件 SAR LUT。例如上面的 5 段输出可安全推断为 unsigned 4-bit；但如果 256 项 logical LUT 交替输出 `{0, 1}`，即使值域只需 1 bit，也不会被错误压到 u1/u2/u4，因为硬件 SAR 查找无法用那么少的比较次数精确表达所有区间。

说明：

- `register_neuron(...)` 必须先于 `compile_to_paiir(...)`
- converter 可返回 `CoreNeuronV25`，也可直接返回 `LutActivation`
  - 直接返回 LUT 时，PAIIR 会按 ANN neuron 兼容层处理
- 同一模块类型重复注册会抛 `ValueError`
- 通常不需要额外手工设置 `_is_leaf_module`
  - 当前 tracer 会把已注册类型视作自定义 leaf module

这条接口对于量化部署尤其重要，因为很多 requant / 自定义激活本质上都可以落到 LUT 表达。

对于自定义计算模块，优先把它转换成已有 canonical 模块：

```python
import torch.nn as nn

from paibox.paiir import register_module


class MyConvWrapper(nn.Module):
    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        self.conv = conv

    def forward(self, x):
        return self.conv(x)


register_module(MyConvWrapper, lambda mod: mod.conv)
```

说明：

- `register_module(...)` 的 converter 必须返回当前 PAIIR lowering 已支持的 `nn.Module`
- 典型返回值是 `nn.Conv1d/2d`、`nn.Linear`、pool、内建激活或 PAIIR neuron / LUT 模块
- 它不会根据字段名猜测量化参数；需要在 converter 里显式构造 canonical 模块

## 9. `Mapper.compile(...)` 做了什么

当前后端主入口是：

```python
from paibox.backendv2 import Mapper

mapper = Mapper()
mapper.compile(
    graph,
    output_path="./output",
    literal_format="bin",
)
```

它会完成：

- 构建 routing groups
- 分配神经元和 SRAM
- 求解路由
- 生成三类帧并导出到磁盘
- 生成 `proto/config.pb`
- 在 `debug=True` 时生成 `proto/config.json`

### 9.1 参数说明

- `output_path`
  - 导出目录
  - 如果不传，则优先取环境变量 `PAIBOX_OUTPUT_PATH`
  - 若环境变量也没有，则默认写到当前目录下的 `./output`
- `literal_format`
  - 只影响 `.h` 文件中的 32 位字面量格式
  - 可选 `"bin"` 或 `"hex"`
  - 不影响 `.txt`、`.npy`、`.pb` 或 `.json`
- `target_platform`
  - `"x86"`、`"riscv"` 或 `"all"`
  - `"x86"` 导出 `cfg_frame*.npy`
  - `"riscv"` 导出 `cfg_frame*.h`
  - `"all"` 同时导出两套平台相关产物
- `word_order`
  - 控制 `proto/config.pb` 与 `proto/config.json` 中 `config_frames.words` 的 32 位拆分顺序
  - 可选 `"high_first"` 或 `"low_first"`
- `export_merged_frames`
  - 是否同时导出不同类型帧合并后的文件
- `debug`
  - 是否保留供人工查看的 `.txt` 与 `proto/config.json`
  - 当 `debug=True` 时，会同时导出 x86 与 riscv 两套平台相关产物
- `export_proto_python`
  - 仅在 `target_platform="x86"` 时生效
  - 控制是否把 `compile_artifacts_pb2.py` / `compile_artifacts_pb2.pyi` 复制到导出目录的 `proto/`

### 9.2 默认导出文件

后端当前会导出三类产物：

- 人类可读 debug 文本帧文件（`debug=True` 时）
  - `cfg_frame1.txt`
  - `cfg_frame2.txt`
  - `cfg_frame3.txt`
  - `cfg_frames.txt`（按物理核顺序合并，每个核内依次写入 type1/type2/type3）
- 平台相关帧文件
  - `target_platform="x86"`：
    - `cfg_frame1.npy`
    - `cfg_frame2.npy`
    - `cfg_frame3.npy`
    - `cfg_frames.npy`（按物理核顺序合并，取决于 `export_merged_frames`）
  - `target_platform="riscv"`：
    - `cfg_frame1.h`
    - `cfg_frame2.h`
    - `cfg_frame3.h`
    - `cfg_frames.h`（按物理核顺序合并，取决于 `export_merged_frames`）
- protobuf 产物（固定放在 `proto/` 子目录）
  - `proto/config.pb`
  - `proto/config.json`（`debug=True` 时）
  - `proto/compile_artifacts.proto`
  - `proto/compile_artifacts_pb2.py`（仅 x86 且 `export_proto_python=True`）
  - `proto/compile_artifacts_pb2.pyi`（仅 x86 且 `export_proto_python=True`）

### 9.3 三类帧的大致分工

- `cfg_frame1`
  - 核配置帧
- `cfg_frame2`
  - LUT 配置帧
- `cfg_frame3`
  - 神经元参数、权重等 SRAM 相关配置帧

### 9.4 `.txt`、`.npy`、`.h` 和 `proto/` 的区别

`cfg_frame*.txt`：

- 仅在 `debug=True` 时导出
- 面向人类阅读的 debug 文本
- 按核坐标分段，保留类型分组信息

`cfg_frame*.npy`：

- 仅在 `target_platform="x86"` 时导出
- 适合 Python / NumPy 侧直接加载和进一步处理

`cfg_frame*.h`：

- 仅在 `target_platform="riscv"` 时导出
- 每个 64 位帧拆成两个 32 位元素
- 可直接接入 C / C++ 板端工程

`proto/config.pb` / `proto/config.json`：

- 包含 I/O 映射与展平后的配置帧数据
- `config.pb` 适合程序消费
- `config.json` 适合开发人员人工查看
- `config_frames.words` 与 `cfg_frames.npy/.h` 使用相同的物理核优先合并顺序
- `config_frames.word_order` 明确描述了每个 64 位配置帧拆成 32 位 words 时的顺序

### 9.5 用 visualizer 检查编译产物

如果安装了可选可视化依赖，可以直接用 `proto/config.pb` 检查后端最终帧：

```bash
pip install "paibox[visualizer]"
paiviz validate --artifact output_path/proto/config.pb
paiviz --artifact output_path/proto/config.pb
```

`validate` 只做解析和一致性检查，适合放在调试脚本或 CI smoke 中。默认命令会启动本地 Web UI，
用于查看 9 x 9 chip map、core config、LUT、神经元、权重、I/O Map 和 raw frame 调试入口。

更多视图说明见 [编译产物可视化](Visualizer.md)。

### 9.6 `config.pb` 里的 tick 元数据

`proto/config.pb` 会随 I/O 映射导出计算核时序信息，供推理应用侧决定何时送入输入、等待输出、或做复位控制。

- `InputTensorMapping.tick` 是该输入 tensor 首个实际消费计算核的 `tick_start/tick_duration/tick_initial`。
- `InputEntry.tick_relative` 是输入工作帧地址分段，不是计算核启动时间；生成输入工作帧仍使用 `tick_relative/addr_axon/target_lcn`。
- `OutputTensorMapping.name` 使用最终输出源/生产者节点名，便于应用侧定位网络中哪层是输出层；它不是虚拟 `OutputNode` 名。
- `OutputTensorMapping.kind` 描述该输出节点的语义，`DATA` 表示普通激活值/脉冲数据，`VOLTAGE` 表示膜电平。
- `OutputTensorMapping.tick` 是该输出 tensor 最终实际生产者计算核的时序。
- `ThreadIOMapping.core_ticks` 按物理计算核列出 `core_offset/tick/nodes`，不包含全局信号空核。

`TickParams` 是内部硬件字段语义，不是公开 `timesteps` 参数语义。`TickParams.tick_duration=0` 表示持续工作，`tick_duration>0` 表示工作 N 个时间步；`tick_initial=0` 表示不自动复位。若同一个输入或输出 tensor 推导出多个不同 tick，导出阶段会报错，应用侧不应假定可以静默合并。

### 9.7 `config.pb` 里的 I/O 数据类型元数据

`InputEntry.dtype` 和 `OutputEntry.dtype` 描述普通 DATA payload 的 signedness 与 1/2/4/8-bit 逻辑位宽，取值为 `UINT1/INT1/.../UINT8/INT8`。`bit_width` 保留为兼容字段；新应用应优先使用 `dtype` 做输入编码和 DATA 输出解码，并把 `bit_width` 当作冗余校验。

`OutputTensorMapping.kind == VOLTAGE` 时，对应 entries 的 `dtype` 不设置，读取默认值时可视为 `NOT_SET`。这类输出固定按 `int32` 膜电平解释，entry `bit_width=32`，且 `axon_bit_idx` 是 4 个 byte lane 的基地址。

## 10. 输入工作帧

当前 `backendv2` 的 `cfg_frame*.txt` 是人类可读 debug 文本，不是专门的输入布局描述文件。

输入工作帧生成逻辑应读取 `proto/config.pb` 中的 `InputTensorMapping` 和 `InputEntry`，并结合实际输入张量填充 payload。

## 11. 如何准备输入工作帧

只有在你的板端流程真的需要“输入工作帧”时，才需要这一步。当前文档只约定应用侧应消费的元数据；具体 `work_frame1.h` 生成工具可按板端工程格式自行实现。

### 11.1 使用 `proto/config.pb` 作为输入布局来源

输入工作帧的地址信息来自 `proto/config.pb` 中的 `InputTensorMapping.entries`：

- `elem_idx` 指向输入 tensor 按 C-order 展平后的元素。
- `core_offset`、`copy_count`、`tick_relative`、`addr_axon`、`target_lcn` 用于构造目标地址。
- `dtype` 描述输入元素码字类型；例如 `INT8` 通常以 two's complement 原始码字写入 payload。
- `bit_width` 描述该输入元素位宽，并应与 `dtype` 一致。
- `InputTensorMapping.tick` 描述消费该输入的计算核工作窗口，不参与单个输入工作帧地址计算。

### 11.2 多输入模型

如果 `proto/config.pb` 里包含多个输入 tensor：

- 按 `InputTensorMapping.name` 选择对应输入。
- 每个输入 tensor 都按自己的 `shape.size` 和 `entries` 编码。

应用侧不应假定只有一个输入，也不应把 `InputEntry.tick_relative` 当成计算核启动时间。

## 12. 推荐的一般化部署流程

对于“量化后网络模型”的通用部署，推荐按下面的顺序做：

1. 完成量化，并拿到部署所需的整数参数
   例如输入 scale、每层权重、偏置、输出 scale、requant/LUT 规则。
2. 重建一个“部署态 PyTorch 模型”
   保持当前 lowering 可识别的模块表面，不直接依赖原始量化执行图。
3. 用 `compile_to_paiir(..., strict=True)` 编译
   同时保存 `graph.summary()`。
4. 用 `Mapper.compile(...)` 导出平台相关帧文件与 `proto/` 目录
   同时保存 `backendv2.log` 与 `proto/config.pb`。
5. 用 `paiviz validate --artifact proto/config.pb` 检查最终配置帧；
   如需人工排查，再用 `paiviz --artifact proto/config.pb` 打开可视化页面。
6. 如果板端需要输入工作帧，读取 `proto/config.pb` 中的输入映射生成 `work_frame1.h`
7. 向板端交付至少这几类文件
   - 平台相关帧文件（`cfg_frame*.h` 或 `cfg_frame*.npy`）
   - `proto/config.pb`
   - 如需人工检查，再附带 `proto/config.json`
   - `paiir_summary.log`
   - `backendv2.log`

## 13. 参考脚本组织方式

这份文档不以单个应用为中心。应用侧部署脚本通常可以拆成两步：

- 编译脚本：负责整理部署态模型、调用 `compile_to_paiir(...)` 和 `Mapper.compile(...)`。
- 工作帧脚本：负责读取 `proto/config.pb`，把应用输入编码为板端需要的输入工作帧。

这类脚本通常遵循一种通用模式：

- 从量化 checkpoint 中提取 `int_repr()` 权重
- 把 bias 转成芯片友好的 `int32`
- 把 requant 保留为精确 LUT
- 重建成 lowering 可识别的部署态 `nn.Module`
- 保存 `paiir_summary.log` 和 `backendv2.log`
- 导出平台相关帧文件和 `proto/` 目录
- 如板端需要，再根据 `proto/config.pb` 生成输入工作帧头文件

## 14. 常见排障建议

### 14.1 `UnsupportedOpError`

优先顺序通常是：

1. 改模型表面，回到 lowering 已支持的模块
2. 用 `register_module(...)` 把自定义计算模块转换成 supported canonical module
3. 用 `register_neuron(...)` 注册自定义 neuron / 激活模块
4. 只在排查阶段使用 `strict=False`

### 14.2 前端编译成功，但后端没有产物

这通常意味着：

- 前端 `compile_to_paiir(...)` 成功了
- 但 `Mapper.compile(...)` 在路由、容量或导出阶段失败了

此时要先看：

- `backendv2.log`
- `routing_groups` 数量
- `output_path` 下是否真的写出了平台相关帧文件
- `output_path/proto/config.pb`
- `output_path/proto/config.json`（若 `debug=True`）

### 14.3 `work_frame1.h` 生成失败

优先检查：

- `proto/config.pb` 是否包含预期的 `InputTensorMapping` 和 `InputEntry`
- 输入数组长度和布局是否与 `InputNode` 契约一致
- 多输入模型是否按 `InputTensorMapping.name` 选择了正确输入

### 14.4 不要混淆“硬件支持”和“当前 lowering 直接支持”

`docs/Support-Ops.md` 更接近硬件能力说明。

真正决定你这次能否直接 `compile_to_paiir(...)` 成功的，是当前 `paibox.paiir.lowering.converter` 和后续编译/融合链路支持到什么程度。

## 15. 一句话总结

如果你希望“量化后模型能够通过这份手册完成编译部署”，当前最稳妥的通用方法不是把原始量化执行图直接丢给前端，而是：

- 先把量化结果整理成 lowering 能理解的部署态 PyTorch 模型
- 再走 `compile_to_paiir(...) -> Mapper.compile(...) -> 平台相关帧文件 + proto/`
- 如有需要，再根据 `proto/config.pb` 的输入映射生成 `work_frame1.h`
