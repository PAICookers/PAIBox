# 量化模型 PAIIR 部署路径说明

本文档整理本次针对 `paibox/quantize_tools` 与 `paibox/paiir` 的对接改动，目标是说明：

1. 这次到底改了哪些地方
2. 现在量化模型应该怎样部署
3. `compile_to_paiir()` 内部会发生什么
4. `ConvAddReLU2d` 这类原来 backend 不直接支持的算子，现在是怎样落到现有后端能力上的
5. 后续如果继续扩展 `ConvReLU2d`、`Conv2d`、`LinearReLU`、`Linear` 甚至更多 fused 算子，应该沿着什么路线做

本文档重点面向当前仓库的开发者，而不是一般用户，因此会尽量把设计取舍、内部 IR 变化和后端约束都写清楚。

---

## 1. 背景与目标

在 `quantize_tools` 里，经过 FX 图改写与校准后，模型会被替换成一批 `Manual*` 量化模块，例如：

- `ManualConvReLU2d`
- `ManualConv2d`
- `ManualLinearReLU`
- `ManualLinear`
- `ManualConvAddReLU2d`

这些模块已经携带了部署所需的量化信息，例如：

- 输入 scale / zero-point
- 权重 scale / zero-point
- 输出 scale / zero-point
- 量化后的 `weight_q`
- 原始 bias
- activation 是否使用对称量化
- residual shortcut 分支的量化比例

原先的一个临时思路，是在 `quantize_tools/deploy.py` 中把这些模块改写成“更接近部署态”的 PyTorch 模块，再尝试喂给后端。这条路线的问题是：

1. 路径不统一  
   普通模型走 `compile_to_paiir()`，量化模型走一条额外的 deploy 临时路径。

2. fused 语义容易散落在多处  
   像 `ConvAddReLU2d` 这种残差算子，scale ratio、shortcut 处理、输出类型、ReLU 融合都容易分散在前端和后端的多处特殊逻辑里。

3. backend 不应该被迫认识越来越多的“新 fused 算子”  
   后端真正稳定的边界应该是“已有的基础 PAIIR 核心算子集合”，而不是不断把项目定制算子直接塞进 backendv2。

因此，本次改动的核心目标是：

> 把 Manual 量化算子统一接入 PAIIR，在 PAIIR 内部先表达“量化部署语义”，然后在 compile pipeline 里 materialize 成 backend 已支持的基础 IR，最后仍然走标准 backendv2 流程。

也就是说，新的统一路径是：

```text
FX/Prepared Model
    -> convert_fx_to_manual(...)
    -> ManualQuant* / ManualConvAddReLU2d
    -> register_manual_quantized_paiir()
    -> compile_to_paiir(...)
    -> Quantized* adapter IR
    -> materialize_quantized_ops(...)
    -> 普通 PAIIR OfflineCoreOp
    -> backendv2 Mapper.compile(...)
```

---

## 2. 本次改动总览

### 2.1 新增文件

#### `paibox/paiir/ir/quantized_ops.py`

新增量化部署适配 IR，定义了四个关键对象：

- `IdentityScale`
- `PotentialPassthroughNodeV25`
- `QuantizedSequentialOp`
- `QuantizedConvAddReLU2dOp`

它们不是直接给 backendv2 消费的最终 IR，而是位于“Manual 量化模块”和“backend-ready PAIIR”之间的一层适配层。

#### `paibox/paiir/pipeline/quantized_materialize.py`

新增一个 pre-fusion rewrite pass，把 `Quantized*` IR 转成现有 PAIIR 基础节点。

#### `paibox/quantize_tools/paiir.py`

新增 `quantize_tools` 到 PAIIR 的注册与映射入口，提供：

- `register_manual_quantized_paiir()`

这个函数把各类 `Manual*` 模块注册为可直接 lower 到 PAIIR IR 的模块。

#### `tests/paiir/pipeline/test_quantized_materialize.py`

新增测试，覆盖：

- `QuantizedSequentialOp -> SequentialOp`
- `QuantizedConvAddReLU2dOp -> backend-ready fragment`
- `ManualConvAddReLU2d -> compile_to_paiir()`
- `ManualConvAddReLU2d -> backendv2 smoke`

---

### 2.2 修改文件

#### `paibox/paiir/lowering/converter.py`

新增：

- `register_ir_module(...)`

作用是允许用户把某个 `nn.Module` 直接 lower 成一个 `OpNode`，而不是先转换成标准 PyTorch 模块。

这和已有的 `register_module(...)` 不同：

- `register_module(...)`：把自定义模块转成 canonical `nn.Conv2d` / `nn.Linear` / 标准模块
- `register_ir_module(...)`：直接返回 PAIIR IR 节点

对量化部署来说，后者更合适，因为 `ManualConvAddReLU2d` 本身就带有明确的部署语义，不应该先被“假装成普通 PyTorch 层”。

#### `paibox/paiir/pipeline/compile.py`

在 pre-fusion pass 里新增：

- `materialize_quantized_ops`

执行顺序是：

1. `materialize_quantized_ops`
2. `fold_zero_pad_into_convs`
3. `canonicalize_layout_chains`
4. `commute_pre_activation_transforms`

也就是说，量化适配 IR 会先被展开，再参与后续普通 PAIIR rewrite 和 fusion。

#### `paibox/paiir/pipeline/passes.py`

主要改动：

1. 新增 `_materialize_direct_add_activation(...)`
2. 允许特定 `PotentialAddOp` 与 activation 融合成 direct-add `AccumulateOp`
3. 允许 `AccumulateOp` 作为合法的 32-bit membrane input consumer，只要它配置了 `AddPotentialMode.DIRECT_ADD`

这是 `ConvAddReLU2d` 能够最终落成“三核结构”的关键一环。

#### `paibox/paiir/ir/op_node.py`

主要改动：

1. `AccumulateOp` 允许 `comps: Sequence[nn.Module | None]`
2. `None` 表示这一条输入路径不再做额外计算，直接接收膜电平输入
3. 当所有 `comps` 都是 `None` 时，自动把 `core_params.add_potential` 设成 `AddPotentialMode.DIRECT_ADD`
4. 保留了非空 `comp` 的 PyTorch 子模块注册，避免权重从模块树里消失

这使得“两个膜电平输出核 -> 一个膜电平直接相加核 -> ReLU”可以在现有 `AccumulateOp` 语义内成立。

#### `paibox/backendv2/get_weight.py`

新增：

- `identity_scale_weight_matrix(...)`

并在 `expanded_path_weight_matrix(...)` 中识别 `IdentityScale`。

作用是：把 shortcut 分支中的 `M` 落成一个“identity matrix * M”的权重矩阵，而不是新造一种后端权重协议。

#### `paibox/backendv2/routing.py`

修正 `OfflineNeuFoldedAttrsV2Part2` 构造时的默认值，补了：

- `fold_vjt_3=0`
- `fold_vjt_2=0`
- `fold_vjt_1=0`
- `fold_vjt_0=0`

这是 backendv2 路由编译过程中暴露出来的一个兼容性修正。

#### 其他导出与注册文件

这些文件做了配套导出：

- `paibox/paiir/__init__.py`
- `paibox/paiir/lowering/__init__.py`
- `paibox/paiir/ir/__init__.py`
- `paibox/quantize_tools/__init__.py`

---

## 3. 现在的推荐部署流程

### 3.1 高层流程

现在推荐的量化部署流程如下：

```python
from paibox.quantize_tools import (
    convert_fx_to_manual,
    register_manual_quantized_paiir,
)
from paibox.paiir import compile_to_paiir
from paibox.backendv2.mapper import Mapper

# 1. 准备并校准 FX 模型
prepared_model = ...

# 2. 把标准量化图改成 Manual 量化模块图
manual_model = convert_fx_to_manual(prepared_model, activation_symmetric=True)

# 3. 注册 Manual 模块到 PAIIR lowering
register_manual_quantized_paiir()

# 4. 编译到 PAIIR
graph = compile_to_paiir(manual_model, *sample_inputs)

# 5. 交给 backendv2
mapper = Mapper()
mapper.compile(graph, output_path=..., target_platform="x86", debug=False)
```

### 3.2 不再推荐的思路

不再建议把“量化部署”长期建立在 `deploy.py` 的临时重写路径上，原因是：

1. 路径割裂
2. fused 语义难以统一管理
3. backend 侧特判会越来越多

`deploy.py` 现在更适合作为：

- 对旧流程的兼容保留
- 做概念验证
- 做和新路径的对照

而不是未来扩展新量化算子的主要入口。

---

## 4. `quantize_tools` 到 PAIIR 的接入方式

### 4.1 `register_manual_quantized_paiir()`

入口在：

- `paibox/quantize_tools/paiir.py`

它做了两件事：

1. 把 `Manual*` 模块标记成 FX leaf module
2. 用 `register_ir_module(...)` 把这些模块注册到 PAIIR lowering

当前注册关系如下：

| Manual 模块           | lower 到的 IR              |
| --------------------- | -------------------------- |
| `ManualConvReLU2d`    | `QuantizedSequentialOp`    |
| `ManualConv2d`        | `QuantizedSequentialOp`    |
| `ManualLinearReLU`    | `QuantizedSequentialOp`    |
| `ManualLinear`        | `QuantizedSequentialOp`    |
| `ManualConvAddReLU2d` | `QuantizedConvAddReLU2dOp` |

### 4.2 为什么要标记成 leaf module

如果不把 `Manual*` 模块视为 leaf module，FX tracing 会直接展开它们的 Python `forward()`。这样会有两个问题：

1. 量化参数语义丢失  
   比如 `weight_q`、`x_scale`、`activation_symmetric` 等信息，不再对应一个明确的模块边界。

2. fused 算子语义丢失  
   `ManualConvAddReLU2d` 会在 tracing 后变成一串普通的算子表达式，无法保留“这是一个需要特殊部署策略的 residual fused operator”。

因此这里的设计是：

> 先保住 Manual 模块的语义边界，再在 PAIIR 里进行受控展开。

---

## 5. `Manual*` 模块如何映射到 PAIIR

### 5.1 简单单路径算子

这类算子包括：

- `ManualConvReLU2d`
- `ManualConv2d`
- `ManualLinearReLU`
- `ManualLinear`

它们的共同特点是：

- 单输入
- 单条计算路径
- 一个计算模块 + 一个部署激活

因此统一 lower 到：

- `QuantizedSequentialOp(comp, act)`

其中：

- `comp` 是构造出的标准 `nn.Conv2d` 或 `nn.Linear`
- `act` 是 `ANNNodeV25(LutReLU / LutReLUSymmetric / LutLinear)`

随后 `materialize_quantized_ops()` 会把它们直接变成：

- `SequentialOp(comp, act)`

也就是说，这类算子的量化接入只是“前面多了一层量化参数绑定”，后面的 PAIIR / backend 路径与普通单核部署保持一致。

### 5.2 参数绑定 `_bind_quantized_params(...)`

在 `paibox/quantize_tools/paiir.py` 中，`_bind_quantized_params(...)` 会把 Manual 模块里的量化参数拷贝到 canonical PyTorch 模块上。

关键点：

1. `weight_q` 会注册成 buffer：`weight_int8`
2. `weight_q` 也会拷贝到 `module.weight`
3. bias 会先按 `accum_scale = s_in * s_w` 量化成 `int32`
4. 量化 bias 会注册成 buffer：`bias_int32`
5. 若模块本身有 bias parameter，也会同步拷贝进去

这样做的原因是：

- 现有 PAIIR / backendv2 主要从标准模块的 `weight` / `bias` 入口取参数
- 同时保留 `weight_int8` / `bias_int32` buffer，便于后续调试与显式语义表达

换句话说，这里不是重新发明一套“量化权重容器协议”，而是尽量贴住当前后端已经稳定支持的参数入口。

---

## 6. `ConvAddReLU2d` 的核心设计

### 6.1 问题本质

`ConvAddReLU2d` 与前面几类单路径算子不同，它本质上是：

```text
conv(y) + shortcut(x) -> ReLU
```

而且其中 shortcut 分支还带有量化比例关系：

```text
x_scale / (conv_input_scale * weight_scale) ~= M * 2^n
```

也就是说，部署时不仅要做 add，还要把 shortcut 分支的缩放关系落到硬件寄存器和权重里。

用户提出的目标是：

1. `M` 配成一个核的权重
2. `n` 配进乘法泄露 / shift 对应寄存器
3. shortcut 分支输出膜电平
4. conv 分支输出膜电平
5. 第三个核做膜电平累加
6. 再配置 ReLU

本次实现正是按这个思路落到现有 PAIIR / backendv2 语义上的。

---

### 6.2 为什么不直接教 backendv2 认识 `ConvAddReLU2d`

因为这样会带来持续扩张的问题：

- backend 需要认识越来越多的 fused 算子
- 每种算子都要有独立导出协议
- backend 的稳定边界会不断后移

更好的做法是：

> 前端先把高层量化部署语义压缩成一种很薄的 adapter IR，再在 compile pipeline 里展开成 backend 已知的基础节点组合。

这就是 `QuantizedConvAddReLU2dOp` 的定位。

---

### 6.3 `ManualConvAddReLU2d -> QuantizedConvAddReLU2dOp`

`_map_manual_conv_add_relu(...)` 会做三件关键事情：

1. 计算目标累加尺度  
   `target_scale = module.conv.s_in * module.conv.s_w`

2. 计算 shortcut 比例  
   `exact_ratio = module.x_scale / target_scale`

3. 调用 `approximate_scale_ratio(exact_ratio)`，得到：
   - `shortcut_m`
   - `shortcut_n`

最终形成：

- `conv`
- `act`
- `shortcut_m`
- `shortcut_n`

这四项构成 `QuantizedConvAddReLU2dOp`。

这里的关键取舍是：

- `M` 不直接写死成后端字段，而是先保留为 IR 语义
- `n` 不直接在前端做位移运算，而是保留成神经元配置语义

这样后续 materialize 时可以更明确地分配到权重和寄存器位置。

---

## 7. `QuantizedConvAddReLU2dOp` 如何展开

### 7.1 展开时机

展开发生在：

- `compile_to_paiir()` 的 pre-fusion rewrite phase
- pass 名称：`materialize_quantized_ops`

也就是说，`QuantizedConvAddReLU2dOp` 不会一路保留到 backendv2，它会在 PAIIR 编译中前期就被展开掉。

### 7.2 展开结果

它会被展开成四个中间节点，随后再由普通 pass 融成三核结构：

1. `StandaloneCompOp(conv)`
2. `SequentialOp(IdentityScale(M), PotentialPassthroughNodeV25(n))`
3. `PotentialAddOp((1, 1))`
4. `StandaloneActOp(ReLU/LUT)`

从物理含义看，对应的是：

```text
Core A: conv(y)                    -> POTENTIAL
Core B: shortcut(x) * M * 2^n      -> POTENTIAL
Core C1: potential add
Core C2: activation (ReLU/LUT)
```

接着普通 fusion pass 会把 `Core C1 + Core C2` 进一步融合成一个 direct-add `AccumulateOp`，于是最终部署形态是：

```text
Core A: conv branch -> POTENTIAL
Core B: shortcut branch -> POTENTIAL
Core C: DIRECT_ADD membrane accumulation + ReLU/LUT
```

这正是“三核结构”。

---

## 8. 三核结构的具体语义

### 8.1 Core A：卷积分支

展开代码：

- `StandaloneCompOp(copy.deepcopy(node.conv))`

特点：

1. 没有激活
2. 输出的是膜电平语义
3. 后续作为 `PotentialAddOp` 的一个输入

这里不需要额外教 backend “这是 residual 的卷积分支”，因为它就是一个普通的单核卷积计算节点。

### 8.2 Core B：shortcut 分支

展开代码：

```python
SequentialOp(
    IdentityScale(node.shortcut_m),
    PotentialPassthroughNodeV25(node.shortcut_n),
)
```

它由两部分组成：

#### `IdentityScale`

语义：

- 表示一个 identity 映射
- 权重增益为整数 `M`

backendv2 在取权重时会识别它，并生成：

```text
identity matrix * M
```

这样 `M` 就被放进了权重矩阵，而不需要新造一种特殊权重类型。

#### `PotentialPassthroughNodeV25`

语义：

- 这不是普通激活
- 它是一个“把膜电平输出明确保留下来”的神经元配置节点

它做的事情是：

1. `leak_multi_input = ENABLE`
2. `leak_tau_shift = n`
3. `output_type = POTENTIAL`

从部署目标看，它正对应了用户提出的：

- `n` 配进泄露 / 移位相关寄存器
- shortcut 核输出膜电平

---

### 8.3 Core C：膜电平相加 + ReLU

`materialize_quantized_ops()` 里先构造：

- `PotentialAddOp((1, 1))`
- `StandaloneActOp(node.act.clone())`

然后给这个 `PotentialAddOp` 打上一个内部标记：

- `_allow_direct_add_activation_fusion = True`

这个标记的意思是：

> 只有量化 residual 展开的这类膜电平加法，允许在后续 pass 中变成 direct-add `AccumulateOp`。

这样可以避免把普通 PyTorch add 也误当成同一种后端语义。

接着在 `passes.py` 中：

- `_materialize_direct_add_activation(...)`

会把：

```text
PotentialAddOp -> StandaloneActOp
```

融合成：

```text
AccumulateOp([None, None], act, signs=(1, 1))
```

这里 `comps == [None, None]` 的含义是：

- 两条输入都已经是膜电平
- 不再额外做 conv / linear / pooling 等计算
- 直接做膜电平加法

当 `AccumulateOp` 发现所有 `comp is None` 时，会自动设：

- `core_params.add_potential = AddPotentialMode.DIRECT_ADD`

于是第三个核的硬件语义就是：

```text
直接接收两个膜电平输入
按符号累加
再接 ReLU/LUT
```

---

## 9. 关键寄存器/字段是怎样落地的

这一部分是理解“为什么这条路线对后端友好”的关键。

### 9.1 `M` 落在哪

`M` 来自：

- `shortcut_m`

它最终通过：

- `IdentityScale`
- `identity_scale_weight_matrix(...)`

落到：

- shortcut 核的权重矩阵

其表现形式不是新权重协议，而是：

```text
I * M
```

即单位矩阵乘以整数增益。

### 9.2 `n` 落在哪

`n` 来自：

- `shortcut_n`

它最终通过：

- `PotentialPassthroughNodeV25(leak_tau_shift=n)`

落到：

- `NeuronParams.leak_tau`

同时还会打开：

- `NeuronParams.leak_multi_input`

也就是说，`n` 不是体现在权重里，而是体现在神经元配置里。

### 9.3 “膜电平输出”落在哪

不论是 conv 分支还是 shortcut 分支，都要求输出膜电平。

对应的落点是：

- `NeuronParams.output_type = OutputType.POTENTIAL`

其中：

- conv 分支因为是 `StandaloneCompOp`，天然是 compute-only / potential output
- shortcut 分支通过 `PotentialPassthroughNodeV25.to_neuron_params()` 强制输出 `POTENTIAL`

### 9.4 “膜电平累加模式”落在哪

第三个核要求直接把两个膜电平输入相加。

对应落点是：

- `OfflineCoreParams.add_potential = AddPotentialMode.DIRECT_ADD`

这是在 `AccumulateOp([None, None], act, ...)` 中自动推导出来的。

### 9.5 ReLU 落在哪

ReLU 不再作为一个“特殊 fused residual 算子”的私有协议存在，而是沿用已有 PAIIR 激活方式：

- `ANNNodeV25(LutReLU)`
- `ANNNodeV25(LutReLUSymmetric)`

最终第三个核就是：

```text
DIRECT_ADD + LUT ReLU
```

这样 ReLU 仍然走原本 backendv2 已支持的 LUT 导出路径。

---

## 10. 为什么这是“统一路径”

这次实现之后，量化模型并没有绕过 `compile_to_paiir()`。

相反，它更深地进入了 PAIIR：

1. 量化模型先成为 `Manual*` 模块
2. `Manual*` 模块注册到 PAIIR lowering
3. lower 成 `Quantized*` adapter IR
4. adapter IR 在 compile pipeline 中被标准化
5. 最终仍然得到普通 backend-ready PAIIR graph
6. backendv2 只需要识别少量基础扩展点，如 `IdentityScale`

因此统一性体现在两层：

### 10.1 用户入口统一

最终部署入口仍然是：

- `compile_to_paiir()`
- `Mapper.compile()`

### 10.2 后端边界统一

backendv2 看到的仍然是：

- `SequentialOp`
- `StandaloneCompOp`
- `PotentialAddOp`
- `StandaloneActOp`
- `AccumulateOp`

加上一个非常小的权重展开补充：

- `IdentityScale`

backend 没有被迫理解 `ManualConvAddReLU2d` 或 `QuantizedConvAddReLU2dOp` 这类项目定制前端算子。

---

## 11. `compile_to_paiir()` 内部现在会发生什么

量化模型进入 `compile_to_paiir()` 以后，主要过程如下。

### 11.1 `torch_to_paiir(...)`

首先做 FX tracing 和 1:1 lowering。

如果模型里已经有 `Manual*` 模块，并且调用过：

- `register_manual_quantized_paiir()`

那么这些模块不会按普通 Python `forward()` 展开，而是直接 lower 成：

- `QuantizedSequentialOp`
- `QuantizedConvAddReLU2dOp`

### 11.2 pre-fusion rewrite

接下来首先运行：

- `materialize_quantized_ops`

这一步会把量化 adapter IR 展开成普通基础节点。

之后继续常规 pre-fusion pass：

- `fold_zero_pad_into_convs`
- `canonicalize_layout_chains`
- `commute_pre_activation_transforms`

### 11.3 add specialization 与 offline core fusion

后续仍然是 PAIIR 既有流程：

- `flatten_general_add_chains`
- `specialize_general_adds`
- `fuse_to_offline_cores`

对于 `ConvAddReLU2d` 展开的 fragment，关键是：

- `PotentialAddOp + StandaloneActOp`

在这里被进一步融合成 direct-add `AccumulateOp`。

### 11.4 分析、格式传播、tick 配置、校验

再往后仍然和普通模型一致：

- 图分析
- signal domain 传播
- data format 传播
- tick 参数赋值
- AvgPool 相关 rewrite / calibration
- 最终 deployable graph 校验

所以可以把新的量化路径理解为：

> 只是把“前端进入 PAIIR 之前”的量化语义接好了，后半段编译器和后端都尽量复用现有能力。

---

## 12. 这次对后端提出的真实要求是什么

本次实现刻意控制了对 backendv2 的改动范围，避免后端承担过多新语义。

后端现在新增或默认依赖的能力只有这些：

1. 能从 `IdentityScale` 生成 `identity * M` 的权重矩阵
2. 能接受 `OutputType.POTENTIAL`
3. 能接受 `NeuronParams.leak_tau`
4. 能接受 `NeuronParams.leak_multi_input`
5. 能接受 `AddPotentialMode.DIRECT_ADD`
6. 能继续按原有方式导出 LUT/ReLU

换句话说，后端并没有被要求支持：

- 新的 residual fused operator
- 新的专用残差寄存器协议
- 新的专用量化 shortcut 协议

这就是为什么这条路线更稳。

---

## 13. 当前实现支持了哪些量化算子

### 13.1 已支持并走统一 PAIIR 路径

- `ManualConvReLU2d`
- `ManualConv2d`
- `ManualLinearReLU`
- `ManualLinear`
- `ManualConvAddReLU2d`

### 13.2 支持方式

可分成两类：

#### 第一类：直接单核

- `ManualConvReLU2d`
- `ManualConv2d`
- `ManualLinearReLU`
- `ManualLinear`

路径：

```text
Manual* -> QuantizedSequentialOp -> SequentialOp
```

#### 第二类：多核展开

- `ManualConvAddReLU2d`

路径：

```text
ManualConvAddReLU2d
    -> QuantizedConvAddReLU2dOp
    -> StandaloneCompOp + SequentialOp(IdentityScale, PotentialPassthrough) + PotentialAddOp + StandaloneActOp
    -> AccumulateOp(DIRECT_ADD) 融合
```

---

## 14. 后续扩展新算子时应该怎么做

这是后续继续扩展时最重要的工程规则。

### 14.1 总原则

优先扩展 PAIIR adapter IR，不优先扩 backendv2。

也就是说：

1. 先判断这个新算子的部署语义能否分解为现有基础核组合
2. 如果可以，就在 PAIIR 前面加一层很薄的 adapter IR
3. 在 `materialize_quantized_ops()` 里展开
4. 让后续普通 pass 和 backend 继续处理

### 14.2 什么时候只用 `QuantizedSequentialOp`

如果新算子满足：

- 单输入
- 单条计算路径
- 最终仍然是一个“算子 + 激活”的单核形态

那就优先走：

- `QuantizedSequentialOp`

例如：

- `ConvReLU2d`
- `Conv2d`
- `LinearReLU`
- `Linear`

### 14.3 什么时候需要新增一种量化 adapter IR

如果新算子满足以下任一条件，就应该考虑新增新的 `Quantized*Op`：

1. 有多输入路径
2. 有 residual / shortcut / branch merge
3. 某些量化参数需要拆开分别落到“权重”和“神经元寄存器”
4. 不能被一个普通 `SequentialOp` 表达

`ConvAddReLU2d` 就是这种情况。

### 14.4 什么时候才应该碰 backendv2

只有在以下情况才应该扩 backend：

1. 现有基础核组合无法表达目标硬件行为
2. PAIIR 里已经很难再分解
3. 新增的后端能力具有长期稳定性，而不是服务某一个临时算子

如果只是某个高层 fused 算子需要特殊展开，一般应该停在 PAIIR 层解决。

---

## 15. 对 `deploy.py` 的定位建议

建议把 `paibox/quantize_tools/deploy.py` 定位成：

- 历史实现
- 参考实现
- 对照验证实现

而把新能力持续加在：

- `paibox/quantize_tools/paiir.py`
- `paibox/paiir/ir/quantized_ops.py`
- `paibox/paiir/pipeline/quantized_materialize.py`

原因很简单：

- `deploy.py` 更像一次性改写脚本
- PAIIR 路线才是统一编译路径

---

## 16. 测试与验证情况

本次改动已验证以下内容。

### 16.1 PAIIR 相关测试

通过：

- `tests/paiir/ir/test_op_node.py`
- `tests/paiir/pipeline/test_data_format.py`
- `tests/paiir/pipeline/test_passes.py`
- `tests/paiir/pipeline/test_quantized_materialize.py`

总计：

- `208 passed`

### 16.2 backendv2 子集验证

通过的可运行子集：

- `tests/backendv2/test_get_weight.py`
- `tests/backendv2/test_mapper.py`
- `tests/backendv2/test_output_axon_allocator.py`
- `tests/backendv2/test_reorder_node.py`
- `tests/backendv2/test_sum_pool_dilation.py`

总计：

- `49 passed`

### 16.3 backend smoke

新增测试中已经实际调用：

- `Mapper().compile(...)`

说明 `ManualConvAddReLU2d -> compile_to_paiir() -> backendv2` 这条路径已经能真实跑通。

### 16.4 当前仓库里与本次无关的旧测试问题

在尝试运行更大范围 backendv2 测试时，发现有两个现有测试问题：

1. `tests/backendv2/test_routing_v2_raw_weights.py` 仍从旧位置导入 `InputElem`
2. `tests/backendv2/test_routing_v2_allocation.py` 使用了旧 mock 名称 `OfflineCorePlamentV2`

这两个问题不是本次量化 PAIIR 对接引入的。

---

## 17. 本次改动涉及的文件清单

### 新增

- `docs/quantized_paiir_deployment.md`
- `paibox/paiir/ir/quantized_ops.py`
- `paibox/paiir/pipeline/quantized_materialize.py`
- `paibox/quantize_tools/paiir.py`
- `tests/paiir/pipeline/test_quantized_materialize.py`

### 修改

- `paibox/backendv2/get_weight.py`
- `paibox/backendv2/routing.py`
- `paibox/paiir/__init__.py`
- `paibox/paiir/ir/__init__.py`
- `paibox/paiir/ir/op_node.py`
- `paibox/paiir/lowering/__init__.py`
- `paibox/paiir/lowering/converter.py`
- `paibox/paiir/pipeline/compile.py`
- `paibox/paiir/pipeline/passes.py`
- `paibox/quantize_tools/__init__.py`

---

## 18. 一句话总结

这次改动的本质不是“给 backend 增加一个量化残差特判”，而是：

> 在 `Manual*` 量化模块和 backend-ready PAIIR 之间加了一层很薄的量化 adapter IR，把 `ConvAddReLU2d` 这类复杂 fused 算子的部署语义，规范地拆解到现有的权重矩阵、膜电平输出、leak_tau、DIRECT_ADD 和 LUT 激活这些稳定后端能力上。

因此后续扩展的正确方向是：

> 尽量扩 PAIIR adapter IR 和 materialize pass，尽量少直接扩 backendv2。
