# PAIIR TensorLayout 架构要点

本文档不是正式 PPT，而是面向后续整理幻灯片时可直接取材的文字版提纲。

## Slide 1: 为什么要做这次重构

重构前：

- `OpNode` 用四个字段分散表达 metadata
  - `input_shapes`
  - `output_shape`
  - `input_dims`
  - `output_dims`
- `SplitOp` 是多输出节点，但 `OpNode` 的输出接口天然偏单输出
- shape 与 dims 作为平行数组，维护成本高

重构后：

- 用 `TensorLayout(shape, dims)` 把 shape 与 dims 绑定
- `OpNode` 用：
  - `input_layouts`
  - `output_layouts`
- 单输出只是 `len(output_layouts) == 1` 的特例

## Slide 2: 新的核心类型

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

补充查询接口：

- `node.num_inputs`
- `node.num_outputs`
- `graph.get_edge_output_layout(edge)`

## Slide 3: 为什么 `TensorLayout` 需要 frozen

原因：

- shape 与 dims 必须成对变化
- 编译期 metadata 更适合整体替换，不适合对象内部原地修改
- 不可变对象更利于 pass 间复制与复用

结论：

- `TensorLayout` 使用 `frozen=True`
- 若需要更新 layout，就构造新的 `TensorLayout`

## Slide 4: `src_port` / `dst_port` 仍然保留

这次重构不删除端口字段：

- `src_port`
  - 源节点输出索引
  - `SplitOp` 依赖它选择具体分支
- `dst_port`
  - 目标节点输入槽位
  - `ConcatOp` / `AccumulateOp` / `PotentialAddOp` 依赖它保持输入顺序

即使节点已经有 `output_layouts` / `input_layouts`，边仍然必须明确“连的是哪一个输出”和“进的是哪一个输入口”。

## Slide 5: Split 的新表达方式

`SplitOp` 节点本身保留：

- `sections`
- `dim`

输出 metadata 则变为：

- `output_layouts[0]`
- `output_layouts[1]`
- ...

边连接：

- `SplitOp --(src_port=0)--> branch_a`
- `SplitOp --(src_port=1)--> branch_b`

这样就不再需要为 `SplitOp` 伪造一个统一的 `output_shape`。

## Slide 6: 泛型消费者怎么写

推荐统一走边级 helper：

```python
layout = graph.get_edge_output_layout(edge)
shape = layout.shape
dims = layout.dims
```

优势：

- 泛型代码不需要关心节点是单输出还是多输出
- `SplitOp` 不再需要散落在各处的 shape 特判

## Slide 7: 对 compile pass 的影响

主要改动点：

- lowering 直接填充 `input_layouts/output_layouts`
- fusion pass 拷贝 layout tuple，而不是分别拷贝四个字段
- concat / reshape / split contract 校验全部改成读取 `TensorLayout`
- layout canonicalization / elision 直接比较 `TensorLayout.shape` 与 `TensorLayout.dims`

## Slide 8: 当前分支的边界

本分支已经完成：

- `paibox/paiir/**` 接口改造
- `tests/paiir/**` 迁移

本分支尚未完成：

- `paibox/backendv2/**` 适配

因此需要对后续协作明确：

- backendv2 需要单独迁移到 `TensorLayout` 接口
- 在那之前，本分支不承诺 backendv2 侧可直接运行
