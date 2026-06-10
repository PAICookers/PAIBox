# ResNet CIFAR-10 `deploy_backendv2.py` 排查记录

本文记录 `examples/quantize/cifar10/resnet/deploy_backendv2.py` 的一次运行排查。

排查目标有两个：

1. 解释 backendv2 阶段的断言报错原因。
2. 确认量化 residual `ConvAddReLU2d` 是否按预期展开为“三个 core”结构。

结论先行：

- 当前报错不是量化转换失败，也不是 residual 三 core 展开失败。
- PAIIR 图中 residual 已经按预期展开。
- 报错发生在 backendv2 routing group 构建阶段，原因是分组逻辑暂时不能处理同一个前驱同时连接 `RemapNode` 和 `CoreOpNode` 的 mixed successor set。

## 1. 运行阶段概览

运行命令：

```powershell
.venv\Scripts\python.exe examples\quantize\cifar10\resnet\deploy_backendv2.py
```

运行流程实际已经完成了以下阶段：

1. 加载 FP32 checkpoint。
2. FX fuse。
3. FX prepare 并插入 observer。
4. calibration。
5. `convert_fx_to_manual(...)` 转成 manual quantized modules。
6. 导出量化 summary 和 int 权重。
7. `compile_to_paiir(...)` 成功生成 PAIIRGraph。
8. 进入 backendv2 `Mapper.compile(...)`。

报错发生在第 8 步，不在前面的 FX quantization 或 PAIIR compile 阶段。

## 2. 具体报错

运行输出最后的异常是：

```text
AssertionError: Expected exactly one non-empty node set for group building, but got:
{
    'remap_node_set': {RemapNode(SequentialOp_0_Padded)},
    'routing_node_set': {CoreOpNode(SequentialOp_5)}
}
```

调用链是：

```text
deploy_backendv2.py
  -> mapper.compile(...)
    -> Mapper.generate_routing_groups(...)
      -> build_groups(...)
        -> AssertionError
```

对应代码位置：

```text
paibox/backendv2/mapper.py
paibox/backendv2/rg_build.py
```

## 3. 报错原因

backendv2 的 `build_groups(...)` 当前按如下思路构造 group：

1. 对每个节点，把它的 successors 作为一个 set。
2. 再把节点自身也作为一个 set。
3. 如果两个 set 有交集，就合并。
4. 合并后要求每个 group 里只能包含一种节点类型：
   - `RemapNode`
   - `CoreOpNode`
   - `InNode`
   - `OutNode`

关键逻辑可以概括为：

```python
for node in nodes:
    if len(node.successors) > 0:
        node_sets.append(set(node.successors))
    node_sets.append({node})

# 后面合并相交 set

non_empty_sets = {
    name: s
    for name, s in group_node_sets.items()
    if len(s) > 0
}
assert len(non_empty_sets) == 1
```

这套逻辑隐含了一个假设：

> 一个 group 里不会同时出现 remap 节点和 core 节点。

但当前 ResNet residual 图中，这个假设被 padding/remap 插入打破了。

## 4. 为什么会出现 mixed successor set

在 PAIIR 图中，`SequentialOp_0` 的输出同时走两条路：

```text
SequentialOp_0
  -> SequentialOp_1
  -> SequentialOp_5
```

其中：

- `SequentialOp_1` 是主分支下一层卷积路径。
- `SequentialOp_5` 是 residual shortcut 的 `M`/`n` 缩放路径。

进入 backendv2 后，因为后续卷积需要 padding，backend 插入了 remap/padding 节点：

```text
SequentialOp_0
  -> SequentialOp_0_Padded
  -> SequentialOp_5
```

此时两个 successor 的类型不同：

```text
SequentialOp_0_Padded : RemapNode
SequentialOp_5        : CoreOpNode
```

`build_groups(...)` 把这两个 successor 放进同一个 set：

```text
{RemapNode(SequentialOp_0_Padded), CoreOpNode(SequentialOp_5)}
```

后续类型检查发现这个 set 同时包含 `remap_node_set` 和 `routing_node_set`，因此触发断言。

运行输出中还有第二个同类 mixed set：

```text
{RemapNode(AccumulateOp_0_Padded), CoreOpNode(SequentialOp_3)}
```

所以这不是单个节点的偶发问题，而是 backendv2 group builder 对“一个 producer 分叉到 remap 和 core”的通用支持不足。

## 5. 这不是 residual 三 core 展开错误

在报错前，`compile_to_paiir(...)` 已经成功输出 PAIIRGraph。

图中第一处 residual 展开结果是：

```text
SequentialOp_5 (Conv2d -> PotentialPassthroughNodeV25) (16, 32, 32) s32
StandaloneCompOp_1 (Conv2d) (16, 32, 32) s32
AccumulateOp_0 (NoneType + NoneType -> ANNNodeV25) (16, 32, 32) s8
```

其数据流是：

```text
SequentialOp_1
  -> StandaloneCompOp_1
  -> AccumulateOp_0[dst_port=0]

SequentialOp_0
  -> SequentialOp_5
  -> AccumulateOp_0[dst_port=1]

AccumulateOp_0
  -> next nodes
```

这正是预期的三 core residual add 结构。

## 6. 三 core 结构逐项对照

预期结构：

1. 第一个 core：做卷积，输出膜电平。
2. 第二个 core：做 `weight = M`，乘法泄露 `n`，输出膜电平。
3. 第三个 core：输入前两个 core 的膜电平，做膜电平加法，然后配置 LUT ReLU 输出。

当前 PAIIR 图对应如下。

### 6.1 Core 1: residual 主分支卷积

PAIIR 节点：

```text
StandaloneCompOp_1 (Conv2d) (16, 32, 32) s32
```

含义：

- 使用 `StandaloneCompOp`。
- 只做卷积，不带 activation。
- 输出是 `s32`，即膜电平/电位输出。
- 输出接到 `AccumulateOp_0` 的输入 0。

运行输出：

```text
StandaloneCompOp_1 (Conv2d) (16, 32, 32) s32
  -> src_port=0 -> AccumulateOp_0[dst_port=0]
```

### 6.2 Core 2: shortcut 分支 `M`/`n` 缩放

PAIIR 节点：

```text
SequentialOp_5 (Conv2d -> PotentialPassthroughNodeV25) (16, 32, 32) s32
```

含义：

- `Conv2d` 是 depthwise `1x1 Conv2d`。
- 权重填充为 `M`。
- 后接 `PotentialPassthroughNodeV25(n)`。
- `n` 通过 leak 配置表达。
- 输出是 `s32`，即膜电平/电位输出。
- 输出接到 `AccumulateOp_0` 的输入 1。

运行输出：

```text
SequentialOp_5 (Conv2d -> PotentialPassthroughNodeV25) (16, 32, 32) s32
  -> src_port=0 -> AccumulateOp_0[dst_port=1]
```

代码中这个节点来自：

```python
shortcut_conv = _build_shortcut_scale_conv(...)
shortcut_core = SequentialOp(
    shortcut_conv,
    PotentialPassthroughNodeV25(node.shortcut_n),
)
```

其中 `_build_shortcut_scale_conv(...)` 会创建 depthwise `1x1 Conv2d`，并将权重填为 `shortcut_m`。

### 6.3 Core 3: 膜电平加法 + LUT ReLU

PAIIR 节点：

```text
AccumulateOp_0 (NoneType + NoneType -> ANNNodeV25) (16, 32, 32) s8
```

含义：

- `comps=[None, None]`。
- 两个输入都已经是膜电平，不再做额外 compute。
- 使用 direct membrane add。
- 后接 `ANNNodeV25`，其中 activation 是量化后的 LUT ReLU。
- 输出是 `s8`。

运行输出：

```text
AccumulateOp_0 (NoneType + NoneType -> ANNNodeV25) (16, 32, 32) s8
  <- StandaloneCompOp_1[src_port=0] -> dst_port=0
  <- SequentialOp_5[src_port=0] -> dst_port=1
```

测试中也覆盖了该语义：

```python
assert acc_nodes[0].core_params.add_potential == AddPotentialMode.DIRECT_ADD
assert acc_nodes[0].comps == [None, None]
```

## 7. 第二处 residual 的额外 projection

图中第二处 residual 也按三 core 展开：

```text
StandaloneCompOp_2
SequentialOp_6
AccumulateOp_1
```

但这一处 residual 位于 `layer2`，shortcut 不是纯 identity，而是有 projection/downsample：

```text
SequentialOp_3 (Conv2d -> ANNNodeV25) (32, 16, 16) s8
  -> SequentialOp_6
```

因此第二处 residual 的完整路径是：

```text
projection shortcut conv
  -> M/n shortcut potential core
  -> membrane add + LUT ReLU core
```

也就是说：

- residual add 这部分仍然是三 core。
- 但因为模型本身有 projection shortcut，所以整个 shortcut 路径会额外多一个正常 projection conv core。

这符合 ResNet downsample block 的结构，不是 materialize 错误。

## 8. Materialize 代码如何生成三 core

量化 residual 的展开逻辑位于：

```text
paibox/paiir/pipeline/quantized_materialize.py
```

核心流程如下。

### 8.1 主分支卷积 core

```python
conv_core = StandaloneCompOp(copy.deepcopy(node.conv))
```

注释中明确说明：

```text
Core A: conv branch. No activation here, so it naturally emits POTENTIAL.
```

### 8.2 shortcut 缩放 core

```python
shortcut_conv = _build_shortcut_scale_conv(
    node.input_layouts[1],
    node.output_layouts[0],
    node.shortcut_m,
)
shortcut_core = SequentialOp(
    shortcut_conv,
    PotentialPassthroughNodeV25(node.shortcut_n),
)
```

这里：

- `shortcut_m` 落到 depthwise `1x1 Conv2d` 的 weight。
- `shortcut_n` 落到 `PotentialPassthroughNodeV25`。
- 输出为 potential。

### 8.3 add + act

```python
add = PotentialAddOp((1, 1))
add._allow_direct_add_activation_fusion = True

act = StandaloneActOp(node.act.clone())
```

后续 PAIIR pass 会把这两个节点融合成：

```text
AccumulateOp (NoneType + NoneType -> ANNNodeV25)
```

这就是最终看到的第三个 core。

## 9. 当前应该怎么处理

短期判断：

- 不建议先改 `quantized_materialize.py`。
- 不建议先改 `ManualConvAddReLU2d`。
- 不建议怀疑 `shortcut_m` / `shortcut_n` 的三 core 语义。

真正需要修的是 backendv2 group builder 或 padding/remap 插入后的分组策略。

推荐修复方向：

1. 修改 `paibox/backendv2/rg_build.py`。
2. `build_groups(...)` 不应把一个节点的所有 successors 无条件放进同一个 group。
3. 对 mixed successor set，至少要按节点类型拆分：
   - `RemapNode` 单独形成 remap group。
   - `CoreOpNode` 单独形成 routing group。
4. 或者在 connected component 构建时只合并同类节点，避免后续再用断言兜底。

需要注意，运行输出里至少有两个 mixed set：

```text
{RemapNode(SequentialOp_0_Padded), CoreOpNode(SequentialOp_5)}
{RemapNode(AccumulateOp_0_Padded), CoreOpNode(SequentialOp_3)}
```

所以修复不能只针对 `SequentialOp_0` 特判，而应该处理通用的 producer fanout 到 remap/core 的情况。

## 10. 临时绕过方式

如果只想确认 PAIIR 图和三 core 结构：

- 可以运行到 `compile_to_paiir(...)` 后停止。
- 或在脚本里临时不执行 `mapper.compile(...)`。

但如果目标是生成 backendv2 frame/output，目前不能仅靠命令行参数绕过这个问题。

`--no-debug` 只影响 debug 输出，不会改变 routing group 构建逻辑，因此不能解决这个断言。

## 11. 最终判断

本次问题应分成两层看：

1. 量化 residual lowering 层：
   - 已按预期生成三 core。
   - `Conv2d -> potential`、`M/n -> potential`、`membrane add + LUT ReLU` 都在图中出现。

2. backendv2 routing group 层：
   - 当前不能处理 padding/remap 后形成的 mixed successor group。
   - 这是 backendv2 分组算法的泛化能力问题。

因此后续修复重点应放在 backendv2 `build_groups(...)`，而不是修改 residual 三 core 设计。
