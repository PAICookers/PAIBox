# PAIIR 计算图后端开发指南

本文档面向后端开发人员，介绍如何从编译完成的 PAIIR 计算图中获取各项信息，以开发芯片部署工作流。

## 目录

- [概述](#概述)
- [编译流程与 API](#编译流程与-api)
- [核心数据结构](#核心数据结构)
- [计算图遍历与查询](#计算图遍历与查询)
- [算子节点信息提取](#算子节点信息提取)
- [完整示例](#完整示例)
- [附录：关键类型参考](#附录关键类型参考)

## 概述

PAIIR（PAIBox Intermediate Representation）是 PAIBox 工具链的中间表示层，将 PyTorch 模型转换为芯片可部署的计算图。

设计原则：

- **版本无关**：IR 层不区分芯片版本（v2.0 / v2.5），后端负责版本特异的参数编码
- **精确映射**：每个 `OfflineCoreOp` 对应芯片上的一个离线计算核
- **信息完备**：编译后的计算图包含权重、神经元参数、LUT 数据、数据格式、时序等部署所需的全部信息

## 包结构速览

当前 `paiir` 已按职责拆分为以下子包：

```text
paibox.paiir/
├── ir/         # IR 实体：节点、图、神经元、LUT、参数
├── lowering/   # PyTorch / FX -> PAIIR 构图
├── pipeline/   # 编译流水、graph pass、数据格式、AvgPool 部署策略
└── nn/         # 可复用运行时模块（如 SumPool1d / SumPool2d）
```

对后端开发者来说，通常有两类入口：

- **公共入口**：从 `paibox.paiir` 顶层导入 `compile_to_paiir`、`PAIIRGraph`、`OfflineCoreOp` 等稳定 API
- **内部扩展入口**：从 `paibox.paiir.pipeline.passes`、`paibox.paiir.ir.*` 等模块导入更细粒度的类型与 pass

补充说明：

- `paibox.paiir.pipeline.passes` 是当前编译 pass 的公共入口
- `paibox.paiir.pipeline.pass_manager` 目前仍是实验性基础设施，不驱动默认 `compile_to_paiir()` 路径

## 编译流程与 API

### 编译流程概览

```
PyTorch 模型
    │
    ▼  ① torch_to_paiir()           — FX 追踪，表达层 1:1 节点映射
    │
    ▼  ② specialize_general_adds()  — 将可部署的 GeneralAddOp 收紧为 PotentialAddOp
    │
    ▼  ③ fuse_to_offline_cores()    — 融合原子节点为离线核单元
    │
    ▼  ④ validate_graph()           — 结构验证 + 自动清理断联节点
    │
    ▼  ⑤ propagate_signal_domain()  — 标注 VALUE / POTENTIAL 语义域
    │
    ▼  ⑥ propagate_data_format()    — 两阶段数据格式推理（输出/权重 → 输入传播）
    │
    ▼  ⑦ assign_tick_params()       — 基于 DAG 深度分配时序参数
    │
    ▼  ⑧ calibrate_avgpool_thresholds() — 可选 AvgPool 阈值细化
    │
    ▼  ⑨ validate_compiled_graph()  — 编译完成后的结构/元信息校验
    │
    ▼  ⑩ validate_deployable_graph() — backend-ready 子集与契约校验
    │
    ▼
PAIIRGraph (backend-ready，可交付后端)
```

### 一站式编译接口

`compile_to_paiir()` 封装了上述全部流程：

```python
from paibox.paiir import compile_to_paiir

# 基本编译
graph = compile_to_paiir(model, torch.randn(1, 3, 32, 32))

# 带时序配置
graph = compile_to_paiir(
    model,
    torch.randn(1, 3, 32, 32),
    tick_duration=100,       # 每核工作时长（0 = 常开）
    auto_reset=True,         # 工作周期结束后自动复位神经元状态
)

# 使用 CompileConfig + 关键字覆盖
from paibox.paiir import CompileConfig

cfg = CompileConfig(tick_duration=100, auto_reset=True)
graph = compile_to_paiir(model, x, compile_config=cfg, strict=False)
```

`compile_to_paiir` 完整参数：

| 参数                         | 类型                              | 说明                                                |
| ---------------------------- | --------------------------------- | --------------------------------------------------- |
| `model`                      | `nn.Module`                       | PyTorch 模型                                        |
| `*sample_inputs`             | `Tensor`                          | 示例输入（batch_size 必须为 1），用于推断形状和维度 |
| `tick_duration`              | `int \| None`                     | 全局工作时长，默认 0（常开）                        |
| `auto_reset`                 | `bool \| None`                    | 工作周期结束后自动复位，默认 True                   |
| `tick_overrides`             | `dict[str, TickOverride] \| None` | 按节点名指定时序覆盖                                |
| `input_formats`              | `dict[str, DataFormat] \| None`   | 按 InputNode 名指定输入数据格式                     |
| `compile_config`             | `CompileConfig \| None`           | 配置对象（关键字参数优先级更高）                    |
| `concrete_args`              | `dict[str, Any] \| None`          | 传递给 `fx.Tracer.trace` 的具体参数                 |
| `strict`                     | `bool`                            | True = 遇到不支持的算子时报错；False = 警告并跳过   |
| `enable_avgpool_calibration` | `bool \| None`                    | 是否启用共享核 AvgPool+LIF 阈值细化，默认关闭       |
| `enable_split_avgpool_lif`   | `bool \| None`                    | 是否允许条件式 AvgPool+LIF 分核部署，默认关闭       |

### 分步编译

需要更细粒度控制时，可单独调用各阶段：

```python
from paibox.paiir import torch_to_paiir
from paibox.paiir.pipeline.passes import (
    specialize_general_adds,
    fuse_to_offline_cores,
    validate_graph,
    propagate_signal_domain,
    propagate_data_format,
    assign_tick_params,
    calibrate_avgpool_thresholds,
    validate_compiled_graph,
    validate_deployable_graph,
)

# ① FX 追踪 + 表达层 1:1 节点映射
graph = torch_to_paiir(model, sample_input)

# torch_to_paiir() 的输出仍可能包含 GeneralAddOp，
# 适合前端检查 / 仿真，但还不是 backend-ready 图。

# ② 将可部署的 GeneralAddOp 收紧为 PotentialAddOp
graph = specialize_general_adds(graph)

# ③ 算子融合（返回新图）
graph = fuse_to_offline_cores(graph)

# ④ 结构验证（原地修改，断联节点自动移除并发出警告）
validate_graph(graph)

# ⑤ 信号域传播（原地填充 output_domain）
propagate_signal_domain(graph)

# ⑥ 数据格式推理（原地填充 core_params 中的数据格式字段）
propagate_data_format(graph)

# ⑦ 时序参数分配（原地填充 tick_start / tick_duration / tick_initial）
assign_tick_params(graph, tick_duration=100, auto_reset=True)

# ⑧ 可选：共享核 AvgPool+LIF 阈值细化
calibrate_avgpool_thresholds(graph)

# ⑨ 元信息校验（形状、数据格式、tick 参数与连通性）
validate_compiled_graph(graph)

# ⑩ backend-ready 子集校验（禁止残留 GeneralAddOp 等表达层节点）
validate_deployable_graph(graph)
```

> **注意 1**：各阶段的 pass 函数（`fuse_to_offline_cores` 等）不在 `paibox.paiir` 的顶层导出中，需要从 `paibox.paiir.pipeline.passes` 导入。
>
> **注意 2**：`validate_graph()` 与 `validate_compiled_graph()` 的职责不同：
>
> - `validate_graph()` 用于融合后的中途结构清理与基础校验
> - `validate_compiled_graph()` 用于所有编译期注解填充完成后的最终验收

## 核心数据结构

### PAIIRGraph

计算图容器，管理节点和有向边。

```python
from paibox.paiir.ir.graph import PAIIRGraph

graph.name: str                        # 图名称
graph.nodes: dict[str, PAIIRNode]      # 节点字典，key = 节点名
graph.edges: list[Edge]                # 有向边列表
```

### Edge

```python
from paibox.paiir.ir.graph import Edge

@dataclass(frozen=True)
class Edge:
    src: str       # 源节点名
    dst: str       # 目标节点名
    src_port: int  # 源节点输出端口号（默认 0，单输出节点恒为 0）
    dst_port: int  # 目标节点输入端口号（默认 0，多输入节点用于保持顺序）
```

补充说明：

- 对大多数单输出节点，`src_port` 恒为 `0`
- `dst_port` 仍然表示目标节点的输入槽位
- 当前 `SplitOp` 是图内唯一的多输出特例，split 的分支选择通过 `Edge.src_port` 表达
  - 例如 `SplitOp --(src_port=0)--> conv_left`
  - `SplitOp --(src_port=1)--> conv_right`

### 节点类型层次

```
PAIIRNode (基类，自动分配唯一 name)
├── InputNode         — 图输入占位符，携带边界 TensorLayout
├── OutputNode        — 图输出节点，携带边界 TensorLayout
└── OpNode (算子基类，携带 TensorLayout 元信息)
    ├── OfflineCoreOp (离线核，映射到芯片核心)
    │   ├── SequentialOp      — comp -> act（最常见）
    │   ├── AccumulateOp      — comps -> add/sub -> act（多路径融合）
    │   ├── PotentialAddOp             — 纯加法/减法（输出膜电位，无激活）
    │   ├── StandaloneCompOp  — 纯计算（融合前的中间状态）
    │   └── StandaloneActOp   — 纯激活（融合前的中间状态）
    ├── ConcatOp        — 路由拼接（非核操作，不占用核资源）
    ├── OnlineCoreOp    — 在线学习核（占位符）
    └── CPUOp           — CPU 回退（占位符，仅 v2.5）
```

融合后的正常图中主要出现 `SequentialOp`、`AccumulateOp`、`PotentialAddOp`、`ConcatOp` 四种算子。`StandaloneCompOp` 和 `StandaloneActOp` 通常在融合后不再独立存在，但在以下场景中仍可能保留：

前端表达层还可能出现 `GeneralAddOp`，用于忠实表示 PyTorch 的通用 `add/sub` 语义；但它不属于 backend-ready 子集。只要图是通过 `compile_to_paiir()` 生成的，`validate_deployable_graph()` 会确保这类表达层节点已经被收紧或拒绝。

- `strict=False` 下的部分旁路图结构
- AvgPool 条件式分核部署中，`StandaloneActOp` 作为第二核保留
- 某些尚未进一步融合的中间编译状态

当前前端 lowering 还可能保留 `SplitOp`，但要注意：

- `SplitOp` 是 frontend-only IR
- 它可以出现在 `torch_to_paiir()` 或 compile 中途图里
- `validate_deployable_graph()` 之后的 backend-ready 图不允许残留 `SplitOp`
- 因此后端如果只消费 `compile_to_paiir()` 的最终结果，默认不需要实现 `SplitOp` 的真实部署逻辑

## 计算图遍历与查询

### 图级查询

```python
# 获取所有输入/输出节点
input_nodes: list[InputNode] = graph.input_nodes()
output_nodes: list[OutputNode] = graph.output_nodes()

# 拓扑排序（返回节点名列表，保证依赖顺序）
ordered_names: list[str] = graph.topo_sort()

# 打印图摘要；verbose=True 时包含 layout 与端口细节
graph.summary()
graph.summary(verbose=True)
```

### 节点级查询

```python
# 前驱节点（按 dst_port 排序，保持输入顺序）
preds: list[str] = graph.predecessors("node_name")

# 后继节点
succs: list[str] = graph.successors("node_name")

# 输入边（按端口排序）
in_edges: list[Edge] = graph.incoming_edges("node_name")

# 输出边
out_edges: list[Edge] = graph.outgoing_edges("node_name")
```

如果你需要读取 split 相关的分支连接关系，请优先使用 `incoming_edges()` / `outgoing_edges()` 而不是只看 `predecessors()`：

- `predecessors()` 只保留节点名，不包含 `src_port`
- `incoming_edges()` / `outgoing_edges()` 保留完整的边端口信息
- 对 `SplitOp` 而言，`src_port` 就是“该边携带第几个 split 分支”

### 遍历模式

```python
from paibox.paiir.ir.ir_base import InputNode, OutputNode
from paibox.paiir.ir.op_node import OfflineCoreOp

# 按拓扑序遍历，筛选 OfflineCoreOp
for name in graph.topo_sort():
    node = graph.nodes[name]

    if isinstance(node, InputNode):
        print(f"Input: {name}, shape={node.shape}, dims={node.dims}")
    elif isinstance(node, OutputNode):
        print(f"Output: {name}, shape={node.shape}, dims={node.dims}")
    elif isinstance(node, OfflineCoreOp):
        # 后端关心的节点
        preds = graph.predecessors(name)
        succs = graph.successors(name)
        print(f"Core: {name} ({type(node).__name__})")
        print(f"  inputs <- {preds}")
        print(f"  outputs -> {succs}")
```

## 算子节点信息提取

### OpNode 通用属性

所有 `OpNode` 子类共享：

```python
node.name: str                             # 唯一名称
node.input_layouts: tuple[TensorLayout, ...]   # 各输入端口的 layout
node.output_layouts: tuple[TensorLayout, ...]  # 各输出端口的 layout
node.num_inputs: int                           # 输入端口数量
node.num_outputs: int                          # 输出端口数量
node.output_domain                         # SignalDomain.VALUE / POTENTIAL
```

其中：

```python
@dataclass(frozen=True, slots=True)
class TensorLayout:
    shape: torch.Size
    dims: tuple[int, ...]
```

轴顺序 `(0, 1, 2, 3)` 表示标准 NCHW；如果模型中有 `transpose` / `permute` 操作，轴顺序会相应改变，后端可据此决定数据排布。

当前接口约定：

- 单输出节点通常满足 `len(node.output_layouts) == 1`
- 多输出节点（当前典型是 `SplitOp`）通过 `output_layouts[src_port]` 区分具体输出分支
- 后端和 pass 应优先消费 `TensorLayout`，而不是旧的 `input_shapes / output_shape / input_dims / output_dims`

`output_domain` 是图级语义注解：

- `SignalDomain.VALUE`：值域输出
- `SignalDomain.POTENTIAL`：膜电位域输出

当前 `output_domain` 按节点定义，是单值语义，不按输出端口拆分。对当前 routing-only 节点：

- `ReshapeOp` 输出域继承其唯一输入
- `SplitOp` 虽然是多输出，但所有 split 分支继承同一个输入域
- `ConcatOp` 要求所有输入域一致，然后输出该共同域

后端应把它视为节点输出语义，而不是直接等同于芯片寄存器里的 `OutputType`。对有 `neuron_params` 的离线核节点，两者通常一致；对 `InputNode`、`OutputNode`、`ConcatOp`、`SplitOp`、`ReshapeOp`、`GeneralAddOp` 这类非神经元或 routing 节点，则只能使用 `SignalDomain`。

### OfflineCoreOp：离线核参数

每个 `OfflineCoreOp` 映射到芯片上的一个离线核。后端需要提取三类信息：

#### 1. 核心配置（core_params）

```python
op.core_params: OfflineCoreParams

cp = op.core_params

# 工作模式
cp.snn_mode: SNNMode              # SNN / ANN
cp.pooling_mode: PoolingMode      # AVERAGE / MAX
cp.add_potential: AddPotentialMode # 加电位模式
cp.zero_output: ZeroOutputMode    # 零输出模式

# 数据格式（由 propagate_data_format 填充）
cp.input_sign: DataSign           # 输入符号（UNSIGNED / SIGNED）
cp.input_width: DataWidth         # 输入位宽
cp.output_sign: DataSign          # 输出符号
cp.output_width: DataWidth        # 输出位宽
cp.weight_sign: DataSign          # 权重符号
cp.weight_width: DataWidth        # 权重位宽

# 时序参数（由 assign_tick_params 填充）
cp.tick_start: int | None         # 启动时刻（第几个 sync_all）
cp.tick_duration: int             # 工作时长（0 = 常开）
cp.tick_initial: int              # 自动复位周期（0 = 不复位）

# 部署前验证时序参数是否在寄存器范围内
cp.validate_tick_params()         # 越界或未分配时抛出 ValueError
```

#### 2. 原始参数权重（weights）

```python
raw_weights: list[Tensor] | None = op.weights
```

- `SequentialOp`：若 `comp` 自带显式参数（如 Conv / Linear），返回 `[weight_tensor]`（int8）；池化返回 `None`
- `AccumulateOp`：返回 `[w0, w1, ...]`（int8），与 `comps` 一一对应；若任一 comp 无显式参数则返回 `None`
- `PotentialAddOp`：返回 `None`
- `StandaloneActOp`：返回 `None`

> `op.weights` 现在只表示 IR 图侧“算子自身携带的显式参数张量”，不再承担统一部署矩阵接口的职责。
>
> 后端如果需要真正的 `[out, in]` 路径矩阵，应在 lowering / routing 阶段按算子语义单独物化：
>
> - Conv：由 kernel 展开为 dense matrix
> - Pool：由 `kernel_size / stride / padding / dilation` 合成窗口连接矩阵
> - `StandaloneActOp` / `PotentialAddOp`：在 `comp is None` 时按路径语义合成 signed identity matrix
>
> 因此，不要把 `op.weights` 直接理解为“最终部署权重矩阵”。
>
> 另外，`get_weight_value_range()` 仍可为无显式参数的直通类节点返回隐式传输系数范围（当前为 `[0, 1]`），用于 `weight_format` 推断；这与 `weights is None` 并不矛盾。

#### 3. 神经元参数（neuron_params）

```python
params: NeuronParams = op.neuron_params
```

关键字段：

| 字段                  | 类型                       | 说明                                             |
| --------------------- | -------------------------- | ------------------------------------------------ |
| `reset_mode`          | `RM`                       | `MODE_NORMAL`（硬复位）/ `MODE_LINEAR`（软复位） |
| `reset_v`             | `float`                    | 复位电压                                         |
| `thres_pos`           | `float`                    | 正阈值                                           |
| `thres_neg`           | `float`                    | 负阈值                                           |
| `thres_pos_mode`      | `ThresholdPosMode`         | `FIRE`（触发）/ `CEILING`（截断）                |
| `thres_neg_mode`      | `ThresholdNegMode`         | `FIRE`（触发）/ `FLOOR`（截断）                  |
| `leak_tau`            | `int`                      | 移位指数（正 = 左移放大，负 = 右移衰减）         |
| `leak_v`              | `float`                    | 加性漏电压（含融合后的 bias）                    |
| `init_v`              | `float`                    | 初始膜电位                                       |
| `output_type`         | `OutputType`               | 输出类型                                         |
| `lateral_inhi`        | `LateralInhibitionMode`    | 侧抑制                                           |
| `leak_multi_sequence` | `LeakMultiComparisonOrder` | 乘性漏执行顺序                                   |
| `leak_multi_input`    | `LeakMultiInputMode`       | 输入是否参与乘性漏                               |
| `leak_multi_mode`     | `LeakMultiMode`            | 乘性漏模式                                       |
| `leak_add_mode`       | `LeakAddMode`              | 加性漏方向                                       |

> **bias 融合**：`SequentialOp` 和 `AccumulateOp` 的 `neuron_params` 已将 Conv/Linear 的 bias 融合到 `leak_v` 中，后端无需额外处理。

#### 4. LUT 数据（ANN 模式）

```python
lut: LutData | None = op.lut_data
```

`LutData` 结构：

```python
@dataclass
class LutData:
    thresholds: Tensor  # shape (256,)，分桶边界
    values: Tensor      # shape (256,)，输出值
    is_float: bool      # True = float32 阈值 + bfloat16 值
```

SNN 模式下 `lut_data` 为 `None`。

### 各算子类型的特有信息

#### SequentialOp

```python
from paibox.paiir.ir.op_node import SequentialOp

seq: SequentialOp
seq.comp: nn.Module        # 计算模块（Conv2d / Linear / MaxPool2d / AvgPool2d 等）
seq.act: CoreNeuronV25     # 激活模块
seq.weights                # list[Tensor] | None，原始参数张量；池化通常为 None
seq.neuron_params          # NeuronParams（含 bias 融合、AvgPool 补偿）
seq.lut_data               # LutData | None（含 AvgPool LUT 补偿）
```

#### AccumulateOp

```python
from paibox.paiir.ir.op_node import AccumulateOp

acc: AccumulateOp
acc.comps: nn.ModuleList   # 计算模块列表
acc.signs: tuple[int, ...] # 各路径符号，(1, 1) = 加，(1, -1) = 减
acc.act: CoreNeuronV25     # 激活模块
acc.weights                # list[Tensor] | None，原始参数张量，与 comps 一一对应
acc.neuron_params          # NeuronParams（多路径 bias 按 signs 融合）
acc.lut_data               # LutData | None
```

后端可依赖的最小契约：

- `graph.predecessors(acc.name)`、`acc.comps`、`acc.signs` 必须一一对应
- `len(graph.predecessors(acc.name)) == len(acc.comps) == len(acc.signs)`
- 若 `acc.weights is not None`，则 `len(acc.weights) == len(acc.comps)`
- 若 `input_layouts` 已填充，则它们的长度也应与 `acc.comps` 一致
- 当前符号语义只允许 `+/-1`

> **后端注意**：当前 backendv2 会并行消费 predecessor / comp / weight / sign 列表；如果这些列表长度不一致，Python `zip(...)` 会静默截断。因此上面的结构一致性应视为 backend-ready 图的硬约束，而不是“最好满足”的建议。

#### PotentialAddOp

```python
from paibox.paiir.ir.add_ops import PotentialAddOp

add: PotentialAddOp
add.signs: tuple[int, ...]  # 输入符号
add.weights                  # None（无原始参数）
# neuron_params 为默认直通配置（output_type=POTENTIAL）
```

后端可依赖的最小契约：

- 所有前驱路径都必须是逐元素同形状的 `POTENTIAL` 域输入
- `len(graph.predecessors(add.name)) == len(add.signs)`
- 当前符号语义只允许 `+/-1`

> backendv2 若要为 `PotentialAddOp` 构造部署矩阵，应根据 `signs` 和路径对齐关系自行合成 signed identity matrix，而不是从 `add.weights` 读取。

#### ConcatOp

```python
from paibox.paiir.ir.op_node import ConcatOp

cat: ConcatOp
cat.dim: int  # 拼接维度（通常为 1，即 channel 维）

# 输入顺序由 dst_port 保证
input_names = graph.predecessors(cat.name)  # 已按 dst_port 排序
```

`ConcatOp` 不映射到任何芯片核，后端利用输入端口顺序和各前驱的输出形状来确定轴突地址范围。

## 完整示例

### 示例 1：提取部署所需的全部核信息

```python
from paibox.paiir import compile_to_paiir
from paibox.paiir.ir.ir_base import InputNode, OutputNode
from paibox.paiir.ir.op_node import AccumulateOp, ConcatOp, OfflineCoreOp, SequentialOp


def extract_cores(graph):
    """提取每个 OfflineCoreOp 的部署信息"""
    cores = []

    for name in graph.topo_sort():
        node = graph.nodes[name]
        if not isinstance(node, OfflineCoreOp):
            continue

        cp = node.core_params
        core = {
            "name": name,
            "type": type(node).__name__,
            "input_layouts": node.input_layouts,
            "output_layouts": node.output_layouts,
            # 工作模式
            "snn_mode": cp.snn_mode,
            "pooling_mode": cp.pooling_mode,
            # 数据格式
            "input_format": (cp.input_sign, cp.input_width),
            "output_format": (cp.output_sign, cp.output_width),
            "weight_format": (cp.weight_sign, cp.weight_width),
            # 时序
            "tick_start": cp.tick_start,
            "tick_duration": cp.tick_duration,
            "tick_initial": cp.tick_initial,
            # 连接
            "predecessors": graph.predecessors(name),
            "successors": graph.successors(name),
            # 数据
            "raw_weights": node.weights,
            "weight_value_range": node.get_weight_value_range(),
            "neuron_params": node.neuron_params,
            "lut_data": node.lut_data,
        }

        # 算子特有信息
        if isinstance(node, SequentialOp):
            core["comp_type"] = type(node.comp).__name__
        elif isinstance(node, AccumulateOp):
            core["comp_types"] = [type(c).__name__ for c in node.comps]
            core["signs"] = node.signs

        cores.append(core)

    return cores
```

### 示例 2：提取图的拓扑连接关系

```python
def extract_topology(graph):
    """提取图的拓扑连接，用于后端路由"""
    topology = {
        "inputs": [],
        "outputs": [],
        "cores": [],
        "routing": [],  # ConcatOp
        "edges": [],
    }

    for name in graph.topo_sort():
        node = graph.nodes[name]
        if isinstance(node, InputNode):
            topology["inputs"].append({"name": name, "shape": node.shape})
        elif isinstance(node, OutputNode):
            topology["outputs"].append({"name": name, "shape": node.shape})
        elif isinstance(node, ConcatOp):
            topology["routing"].append({
                "name": name,
                "dim": node.dim,
                "input_order": graph.predecessors(name),
            })
        elif isinstance(node, OfflineCoreOp):
            topology["cores"].append(name)

    for edge in graph.edges:
        topology["edges"].append({
            "src": edge.src,
            "dst": edge.dst,
            "dst_port": edge.dst_port,
        })

    return topology
```

### 示例 3：调试打印

```python
def debug_graph(graph):
    """打印图的完整信息用于调试"""
    graph.summary()
    print()

    for name in graph.topo_sort():
        node = graph.nodes[name]
        if not isinstance(node, OfflineCoreOp):
            continue

        cp = node.core_params
        print(f"[{name}] {type(node).__name__}")
        print(f"  Mode:   {cp.snn_mode.name}, Pooling: {cp.pooling_mode.name}")
        print(f"  Input:  {cp.input_sign.name} {cp.input_width.name}")
        print(f"  Output: {cp.output_sign.name} {cp.output_width.name}")
        print(f"  Weight: {cp.weight_sign.name} {cp.weight_width.name}")
        print(f"  Timing: start={cp.tick_start}, duration={cp.tick_duration}, initial={cp.tick_initial}")

        raw_weights = node.weights
        if raw_weights is not None:
            for i, w in enumerate(raw_weights):
                print(
                    f"  RawWeight[{i}]: shape={tuple(w.shape)}, "
                    f"range=[{w.min().item()}, {w.max().item()}]"
                )
        else:
            print(f"  RawWeight: None, implicit_range={node.get_weight_value_range()}")

        if node.lut_data:
            lut = node.lut_data
            print(f"  LUT: {lut.thresholds.shape}, is_float={lut.is_float}")
        print()
```

若要调试 backend 最终使用的 dense path matrix，请查看 backend routing / lowering 阶段的展开逻辑；该矩阵不直接存放在 `OfflineCoreOp.weights` 中。

## 附录：关键类型参考

### paicorelib 枚举

所有以下类型从 `paicorelib` 导入：

```python
from paicorelib import (
    SNNMode,                    # SNN, ANN
    PoolingMode,                # AVERAGE, MAX
    DataSign,                   # UNSIGNED, SIGNED
    DataWidth,                  # WIDTH_1BIT, WIDTH_2BIT, WIDTH_4BIT, WIDTH_8BIT
    OutputType,                 # VALUE, SPIKE, POTENTIAL, ...
    ThresholdPosMode,           # FIRE, CEILING
    ThresholdNegMode,           # FIRE, FLOOR
    RM,                         # MODE_NORMAL（硬复位）, MODE_LINEAR（软复位）
    AddPotentialMode,
    ZeroOutputMode,
    LateralInhibitionMode,
    LeakMultiComparisonOrder,   # BEFORE_COMPARE, AFTER_COMPARE
    LeakMultiInputMode,
    LeakMultiMode,
    LeakAddMode,                # FORWARD, BACKWARD
)
```

### OfflineCoreParams 定义

```python
@dataclass
class OfflineCoreParams:
    snn_mode: SNNMode = SNNMode.SNN
    pooling_mode: PoolingMode = PoolingMode.AVERAGE
    add_potential: AddPotentialMode = AddPotentialMode.NORMAL
    zero_output: ZeroOutputMode = ZeroOutputMode.DISABLE

    input_sign: DataSign = DataSign.SIGNED
    input_width: DataWidth = DataWidth.WIDTH_8BIT
    output_sign: DataSign = DataSign.SIGNED
    output_width: DataWidth = DataWidth.WIDTH_8BIT
    weight_sign: DataSign = DataSign.SIGNED
    weight_width: DataWidth = DataWidth.WIDTH_8BIT

    tick_start: int | None = None    # None = 待自动分配
    tick_duration: int = 0           # 0 = 常开
    tick_initial: int = 0            # 0 = 不自动复位
```

时序参数的寄存器限制：

- `tick_start`: 16-bit 无符号，[0, 65535]
- `tick_duration`: 32-bit 无符号，[0, 4294967295]
- `tick_initial`: 16-bit 无符号，[0, 65535]

### NeuronParams 定义

```python
@dataclass
class NeuronParams:
    reset_mode: RM = RM.MODE_NORMAL
    reset_v: float = 0.0
    thres_neg_mode: ThresholdNegMode = ThresholdNegMode.FLOOR
    thres_pos_mode: ThresholdPosMode = ThresholdPosMode.FIRE
    thres_neg: float = -131072
    thres_pos: float = 0.0
    lateral_inhi: LateralInhibitionMode = LateralInhibitionMode.DISABLE
    leak_multi_sequence: LeakMultiComparisonOrder = LeakMultiComparisonOrder.AFTER_COMPARE
    leak_multi_input: LeakMultiInputMode = LeakMultiInputMode.DISABLE
    leak_multi_mode: LeakMultiMode = LeakMultiMode.DISABLE
    leak_add_mode: LeakAddMode = LeakAddMode.FORWARD
    leak_tau: int = 0
    leak_v: float = 0.0
    init_v: float = 0.0
    output_type: OutputType = OutputType.VALUE
```

### LutData 定义

```python
@dataclass
class LutData:
    thresholds: Tensor   # shape (256,)
    values: Tensor       # shape (256,)
    is_float: bool = False
```

### 数据格式推理规则

输出格式（由激活模块决定）：

| 模式 | 条件                      | 输出格式                          |
| ---- | ------------------------- | --------------------------------- |
| SNN  | `thres_neg_mode == FIRE`  | SIGNED, WIDTH_2BIT（{-1, 0, +1}） |
| SNN  | `thres_neg_mode == FLOOR` | UNSIGNED, WIDTH_1BIT（{0, 1}）    |
| ANN  | `output_sign == 1`        | SIGNED, WIDTH_8BIT（[-128, 127]） |
| ANN  | `output_sign == 0`        | UNSIGNED, WIDTH_8BIT（[0, 255]）  |

权重格式：从量化权重的实际值范围推断最窄的 `(DataSign, DataWidth)` 组合。

输入格式：从前驱节点的输出格式传播；多输入取最宽表示（sign 取 SIGNED 若任一为 SIGNED，width 取最大值）。

## 当前功能边界说明

面向后端使用时，建议将当前 `paiir` 的输出理解为“编译完成、可供部署侧消费的 IR 图”，但同时注意以下边界：

- `strict=True` 才表示遇到不支持算子会立即失败
- `strict=False` 下图中可能存在被旁路的 unsupported 节点，此时返回图适合做结构分析或部分验证，但不应自动等价理解为“全图已严格支持”
- `graph.summary()`、`graph.predecessors()`、`graph.successors()`、`graph.get_edge_output_layout(...)`、`core_params`、`neuron_params`、`lut_data` 等接口，是后端读取部署信息的主要入口
- 若需要扩展编译流程，请优先在 `paibox.paiir.pipeline.passes` 中新增或调整 pass；`pass_manager` 目前不驱动默认编译路径
- 当前分支已经将 `OpNode` 的 shape/dims 正式接口切换为 `input_layouts/output_layouts`；`backendv2` 尚未适配这次接口变化，需要单独跟进
- `Edge.src_port` 与 `Edge.dst_port` 仍然保留：
  - `src_port` 表示源节点输出索引，`SplitOp` 依赖它选择分支
  - `dst_port` 表示目标节点输入槽位，`ConcatOp` / `AccumulateOp` / `PotentialAddOp` 等依赖它保持输入顺序
