# PAIIR Compile Pass 设计说明

本文档记录 `TensorLayout` 重构后的 PAIIR 编译路径设计要点，面向继续维护 `paibox/paiir/**` 的开发者。

## 1. 核心接口变化

重构前，`OpNode` 将 shape 与 dims 分散记录为四个字段：

- `input_shapes`
- `output_shape`
- `input_dims`
- `output_dims`

重构后，统一改为：

```python
@dataclass(frozen=True, slots=True)
class TensorLayout:
    shape: torch.Size
    dims: tuple[int, ...]
```

```python
class OpNode(...):
    input_layouts: tuple[TensorLayout, ...]
    output_layouts: tuple[TensorLayout, ...]
```

配套接口：

- `node.num_inputs`
- `node.num_outputs`
- `graph.get_edge_output_layout(edge)`

设计意图：

- 把 shape 和 dims 绑定成同一个不可变对象，避免平行数组漂移
- 让多输出节点成为统一模型的一部分，`SplitOp` 不再需要伪造单个 `output_shape`
- 保留 `Edge.src_port/dst_port`，分别表示源输出索引与目标输入槽位

## 2. 编译路径中的 layout 流动

### FX Lowering

在 lowering 阶段，FX 图的 `tensor_meta.shape` 和 `DimsProp` 结果会直接写入：

- `input_layouts`
- `output_layouts`

单输出节点通常生成一个 `output_layouts[0]`。

`SplitOp` 是例外：

- `input_layouts` 长度为 1
- `output_layouts` 长度为 split 分支数
- 每条边通过 `src_port` 选择具体输出分支

### Graph 层

图级消费应优先通过边来获取具体输出 layout：

```python
layout = graph.get_edge_output_layout(edge)
shape = layout.shape
dims = layout.dims
```

这样泛型消费者无需自己判断“这个节点是单输出还是多输出”。

### Pass 层

所有 pass 中关于 shape/dims 的逻辑都应基于 `TensorLayout`：

- 节点拷贝 / 融合时复制 `input_layouts` / `output_layouts`
- concat 校验读取所有输入 layout 的 `shape`
- reshape 校验同时比较输入 shape 和输入 dims
- split 校验读取 `output_layouts[edge.src_port]`

## 3. `src_port` / `dst_port` 为什么保留

本次重构不删除端口字段，原因如下：

- `src_port`
  - 表示源节点输出索引
  - 对 `SplitOp` 以及未来可能出现的多输出节点是必需信息
- `dst_port`
  - 表示目标节点输入槽位
  - 对 `ConcatOp`、`AccumulateOp`、`PotentialAddOp` 等多输入节点是必需信息

即使 `OpNode` 已经使用 `output_layouts` / `input_layouts`，边仍然必须告诉图“消费的是哪一个输出”和“写入的是哪一个输入口”。

## 4. 对 backendv2 的影响

本次分支不改 `paibox/backendv2/**`。

因此需要明确：

- 当前分支只保证 `paibox/paiir/**` 与 `tests/paiir/**` 侧打通
- backendv2 仍然依赖旧的 `raw_node.output_shape` 等接口
- backendv2 适配需要在后续独立分支完成

建议 backendv2 后续迁移方向：

- 优先改为按边读取 `graph.get_edge_output_layout(edge)`
- 对单输出节点再降级使用 `output_layouts[0]`
- 不要在 backendv2 内部重新引入 shape/dims 平行数组

## 5. 当前重构的验收边界

本次重构完成后，应满足：

- `tests/paiir/ir/**` 通过
- `tests/paiir/pipeline/**` 通过
- `OpNode` 不再把旧四字段作为正式接口
- `SplitOp` 的多输出 layout 可通过 `output_layouts + src_port` 正确消费

不在本次验收内：

- `paibox/backendv2/**` 适配
- `tests/backendv2/**` 通过
