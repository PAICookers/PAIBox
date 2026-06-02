# ConvAddReLU2d 量化残差到 PAIIR/backendv2 的数据流说明

本文档记录本次围绕 `ManualConvAddReLU2d`、PAIIR 量化适配 IR、三核展开和 backendv2 导出路径的改动思路。它代表当前 paibox 对量化 residual `ConvAddReLU2d` 的接入方式：前端保留量化语义，PAIIR 内部将其 materialize 成已有基础 IR，backendv2 继续消费标准离线核节点，而不是直接认识新的 fused 算子。

## 1. 背景

在 FX 量化和校准之后，模型里可能出现 PyTorch intrinsic fused module：

```text
torch.ao.nn.intrinsic.ConvAddReLU2d
```

它表示一个残差结构：

```text
conv(y) + shortcut(x) -> ReLU
```

其中 `conv(y)` 和 `shortcut(x)` 通常来自不同量化域。也就是说：

```text
conv 分支输入 y 有自己的 scale / zero-point
shortcut 分支输入 x 有自己的 scale / zero-point
conv weight 有自己的 scale / zero-point
最终输出还有自己的 scale / zero-point
```

本次改动的目标不是让 backendv2 直接支持一个新的 `ConvAddReLU2d` 硬件算子，而是把它拆成 backendv2 已经支持的基础能力：

```text
1. 一个卷积分支核，输出膜电平 POTENTIAL
2. 一个 shortcut 缩放核，输出膜电平 POTENTIAL
3. 一个 DIRECT_ADD + ReLU/LUT 核，接收两个膜电平并输出最终量化值
```

最终部署形态是三个离线核：

```text
y ----> [Core A: Conv -> POTENTIAL] ----------------\
                                                    +----> [Core C: DIRECT_ADD + ReLU/LUT] ----> out
x ----> [Core B: Depthwise 1x1 Conv2d(M) + 2^n -> POTENTIAL] --/
```

## 2. 总体数据流

完整链路如下：

```text
FX prepared / calibrated model
    |
    | convert_fx_to_manual(...)
    v
ManualConvAddReLU2d
    |
    | register_manual_quantized_paiir()
    | torch_to_paiir(...)
    v
QuantizedConvAddReLU2dOp
    |
    | materialize_quantized_ops(...)
    v
StandaloneCompOp(conv)
SequentialOp(depthwise 1x1 Conv2d, PotentialPassthroughNodeV25)
PotentialAddOp
StandaloneActOp
    |
    | fuse_to_offline_cores(...)
    v
StandaloneCompOp(conv)
SequentialOp(depthwise 1x1 Conv2d, PotentialPassthroughNodeV25)
AccumulateOp(DIRECT_ADD + act)
    |
    | backendv2 Mapper.compile(...)
    v
core config / neuron attrs / LUT / weights / routing frames
```

关键原则：

- `QuantizedConvAddReLU2dOp` 只是 PAIIR 内部的适配节点，不应直接进入 backendv2。
- `materialize_quantized_ops` 之后，图里应只剩 backend-ready 的普通 PAIIR 节点。
- backendv2 继续通过 `OfflineCoreOp.core_params`、`neuron_params`、`lut_data`、`weights` 获取配置和权重。

## 3. FX 量化模型到 ManualConvAddReLU2d

转换入口在：

```text
simples_quantize/quantize_tools/converter.py
```

`FxGraphConverter._handle_convaddrelu2d` 会识别：

```text
torch.ao.nn.intrinsic.ConvAddReLU2d
```

然后提取以下量化参数：

```text
s_in, z_in      : conv 分支输入 y 的 scale / zero-point
s_w, z_w        : conv weight 的 scale / zero-point
s_out, z_out    : 整个 ConvAddReLU2d 输出的 scale / zero-point
x_scale, x_zp   : shortcut 分支 x 的 scale / zero-point
conv_out_scale  : s_in * s_w
```

随后它会把原始 fused module 替换成：

```text
ManualConvAddReLU2d
```

`ManualConvAddReLU2d` 保存了两条分支的量化信息：

```text
conv 分支：
  conv.s_in
  conv.z_in
  conv.s_w
  conv.z_w
  conv.weight_q
  conv.bias_val

shortcut 分支：
  x_scale
  x_zp

输出：
  out_scale
  out_zp
  lut
```

它的 Python forward 语义可以理解成：

```text
target_scale = conv_input_scale * weight_scale
exact_ratio = shortcut_input_scale / target_scale
exact_ratio ~= M * 2^n

conv_acc = int_conv(y_int, weight_int8)
shortcut_acc = round((x_int - x_zp) * M * 2^n)
sum_acc = conv_acc + shortcut_acc
out = ReLU_LUT(sum_acc)
```

这里 `M, n` 来自：

```text
simples_quantize/quantize_tools/utils.py::approximate_scale_ratio
```

## 4. ManualConvAddReLU2d 到 QuantizedConvAddReLU2dOp

注册入口在：

```text
simples_quantize/quantize_tools/paiir.py
```

`register_manual_quantized_paiir()` 会做两件事：

```text
1. 把 ManualConvAddReLU2d 标成 FX leaf module
2. 通过 register_ir_module 注册到 PAIIR lowering
```

因此 `torch_to_paiir(...)` 分析模型时，不会展开 `ManualConvAddReLU2d.forward`，而是直接调用：

```text
_map_manual_conv_add_relu
```

该 mapper 计算：

```text
target_scale = module.conv.s_in * module.conv.s_w
exact_ratio = module.x_scale / target_scale
shortcut_m, shortcut_n = approximate_scale_ratio(exact_ratio)
```

并生成：

```text
QuantizedConvAddReLU2dOp(
    conv=canonical_int8_conv,
    act=calibrated_relu_lut,
    shortcut_m=shortcut_m,
    shortcut_n=shortcut_n,
)
```

此时 PAIIR 图里仍是一个量化适配节点。它表达的是 residual 量化语义，而不是最终硬件核分布。

## 5. 权重和 bias 如何进入 PAIIR

`_bind_quantized_params(...)` 会把 manual module 中的量化权重复制到 canonical PyTorch module：

```text
ManualConvAddReLU2d.conv.weight_q
    -> nn.Conv2d.weight
    -> buffer weight_int8
```

也就是说，后续 PAIIR 和 backendv2 不需要理解 `ManualConvAddReLU2d.weight_q` 这个私有字段。它们通过标准的：

```text
nn.Conv2d.weight
```

获取 int8 权重。

如果原始 conv 有 bias，则会做：

```text
bias_q = round(bias_fp32 / accum_scale).to(int32)
```

并写入：

```text
buffer bias_int32
nn.Conv2d.bias
```

注意：当前已经在 `StandaloneCompOp(conv)` 的 `neuron_params` 中把 `conv.bias` 写入 `leak_v`。如果 residual conv 带 bias，导出时应检查 backend neuron attrs 的 `leak_v` 是否包含该 bias。详见本文“注意点”。

## 6. QuantizedConvAddReLU2dOp 的 materialize 展开

展开 pass 在：

```text
paibox/paiir/pipeline/quantized_materialize.py
```

`_materialize_quantized_conv_add_relu(...)` 会把一个：

```text
QuantizedConvAddReLU2dOp
```

展开成四个普通 PAIIR 节点：

```text
conv_core
shortcut_core
add
act
```

对应代码语义：

```text
conv_core = StandaloneCompOp(copy.deepcopy(node.conv))

shortcut_core = SequentialOp(
    depthwise_1x1_conv(weight=node.shortcut_m),
    PotentialPassthroughNodeV25(node.shortcut_n),
)

add = PotentialAddOp((1, 1))

act = StandaloneActOp(node.act.clone())
```

展开后的图形结构：

```text
y -> conv_core ----------------\
                                add -> act -> out
x -> shortcut_core ------------/
```

其中：

- `conv_core` 负责卷积分支。
- `shortcut_core` 负责 shortcut 的 `M * 2^n` 缩放。
- `add` 负责两个膜电平相加。
- `act` 负责 ReLU/LUT 重量化。

## 7. 为什么最终是三个核

`materialize_quantized_ops` 展开后暂时有四个节点，但后续：

```text
fuse_to_offline_cores(...)
```

会把：

```text
PotentialAddOp -> StandaloneActOp
```

融合成：

```text
AccumulateOp
```

因此最终 backend-ready 图里是三个离线核：

```text
Core A:
  StandaloneCompOp(conv)

Core B:
  SequentialOp(depthwise 1x1 Conv2d(M), PotentialPassthroughNodeV25(n))

Core C:
  AccumulateOp([None, None], act, signs=(1, 1))
```

`Core C` 的 `comps=[None, None]` 表示它不再执行 conv/linear，只接收前两个核输出的膜电平，然后做 direct-add 和 activation。

## 8. Core A: 卷积分支核

Core A 是：

```text
StandaloneCompOp(nn.Conv2d)
```

语义：

```text
y -> int8 conv -> membrane potential
```

它没有 activation，因此输出域是：

```text
OutputType.POTENTIAL
```

权重数据流：

```text
ManualConvAddReLU2d.conv.weight_q
    -> canonical nn.Conv2d.weight
    -> StandaloneCompOp.weights
    -> backendv2 CoreOpNode.weights
    -> get_raw_weights(...)
    -> conv2d_weight_matrix(...)
    -> backend Weight package
```

backendv2 展开卷积权重的位置：

```text
paibox/backendv2/get_weight.py::expanded_path_weight_matrix
```

当 `comp` 是 `nn.Conv2d` 时，会根据：

```text
stride
padding
dilation
groups
input_shape
output_shape
```

把卷积核展开成 dense connectivity matrix。

## 9. Core B: shortcut 缩放核

Core B 是：

```text
SequentialOp(
    nn.Conv2d(C, C, kernel_size=1, groups=C, bias=False, weight=M),
    PotentialPassthroughNodeV25(n),
)
```

它表达：

```text
x -> x * M * 2^n -> membrane potential
```

其中：

```text
M 放在 depthwise 1x1 Conv2d.weight 里
n 放在 PotentialPassthroughNodeV25.leak_tau 中
```

这里没有继续让 backendv2 特判 `IdentityScale`。原因是 backend 侧不应该为了 residual fused 算子随便增加新语义。现在 Core B 的整数乘法 `M` 被 materialize 成一个 backendv2 已经支持的普通 `nn.Conv2d`：

```text
in_channels  = C
out_channels = C
kernel_size  = 1
groups       = C
bias         = False
weight shape = (C, 1, 1, 1)
weight[:]    = M
```

因为 `groups=C`，这个卷积是 depthwise 1x1 卷积。每个通道只连接自己，不混通道，不改变 H/W：

```text
out[:, c, h, w] = x[:, c, h, w] * M
```

backendv2 看到它时，只会走已有路径：

```text
nn.Conv2d
    -> Standalone/SequentialOp.weights
    -> expanded_path_weight_matrix(...)
    -> conv2d_weight_matrix(...)
```

`PotentialPassthroughNodeV25(n)` 会生成 neuron 参数：

```text
leak_multi_input = ENABLE
leak_tau = n
output_type = POTENTIAL
```

因此 shortcut 的二进制移位缩放走硬件 leak/shift 字段：

```text
2^n -> leak_tau
```

仿真语义中，`leak_tau` 的效果是：

```text
leak_tau >= 0 : v << leak_tau
leak_tau <  0 : v >> (-leak_tau)
```

因此 shortcut 分支的缩放被拆成：

```text
M   -> weight
2^n -> leak_tau / multiplicative leak shift
```

### 9.1 为什么 depthwise Conv2d 和 PotentialPassthroughNodeV25 是一个 core

Core B 代码里看起来有两个对象：

```text
nn.Conv2d(C, C, kernel_size=1, groups=C, weight=M)
PotentialPassthroughNodeV25(node.shortcut_n)
```

但它们不是两个独立 core，也不是后面才被某个 pass 融合成一个 core。它们从创建时就被包进同一个 `SequentialOp`：

```text
SequentialOp(
    depthwise_1x1_conv(M),
    PotentialPassthroughNodeV25(n),
)
```

`SequentialOp` 本身就是一个 `OfflineCoreOp`，表示芯片离线核内部的两段：

```text
compute stage -> neuron / activation stage
```

因此 Core B 的结构是：

```text
depthwise_1x1_conv(M)
    -> 作为 comp 阶段
    -> 进入 backendv2 的 weight / connectivity 展开

PotentialPassthroughNodeV25(n)
    -> 作为 act / neuron 阶段
    -> 进入 backendv2 的 neuron attrs
```

backendv2 创建 `CoreOpNode` 时，如果原始节点是 `SequentialOp`，会把：

```text
raw_node.comp
```

放进该 core 的 `comps`，所以这个 depthwise 1x1 `Conv2d` 会被当作该 core 的计算部分。

同时，`SequentialOp.neuron_params` 会调用：

```text
raw_node.act.to_neuron_params(...)
```

所以 `PotentialPassthroughNodeV25(n)` 的配置会进入同一个 core 的 neuron 配置：

```text
leak_multi_input = ENABLE
leak_tau = n
output_type = POTENTIAL
```

最终 Core B 的一个硬件核同时拿到：

```text
weight / connectivity : depthwise 1x1 conv, 每个通道权重为 M
neuron attrs          : leak_tau = n, output_type = POTENTIAL
```

也就是：

```text
x -> depthwise_1x1_conv(M) -> leak_tau shift 2^n -> POTENTIAL
```

### 9.2 为什么不把 M 和 2^n 都放进 weight

`M` 是正整数，适合放进权重：

```text
weight = identity * M
```

但 `2^n` 可能是小数缩放。例如：

```text
n = -3
2^n = 1 / 8
```

普通 weight 是整数权重，不适合直接表达 `1/8`。芯片 neuron 的 `leak_tau` 字段可以表达二进制移位：

```text
n >= 0 -> 左移 n 位
n <  0 -> 右移 -n 位
```

因此当前设计把 shortcut scale ratio 拆成：

```text
整数乘法 M      -> depthwise 1x1 Conv2d / weight
二进制移位 2^n -> PotentialPassthroughNodeV25 / leak_tau
```

这对应硬件已有的 weight 和 leak/shift 两类能力。

### 9.3 M 的 weight 大小如何确定

`M` 本身不是一个和 `x` 同 shape 的大张量。它只是一个标量 gain。

现在 materialize 阶段会根据 shortcut 的 NCHW layout 取通道数 `C`，构造一个真实 PyTorch `Conv2d` 参数张量：

```text
weight.shape = (C, 1, 1, 1)
weight[c, 0, 0, 0] = M
```

例如 `x.shape = (1, 16, 32, 32)`，则：

```text
channels = 16
Conv2d(16, 16, kernel_size=1, groups=16, bias=False)
weight.shape = (16, 1, 1, 1)
```

因此 Core B 的 shortcut 映射是：

```text
out[i] = M * x[i]
```

但它不是创建一个和完整 feature map 同 shape 的 `(1, C, H, W)` 权重张量。卷积权重只按通道保存，空间位置由 1x1 卷积在每个 `(h,w)` 上复用。

### 9.4 为什么要检查 shortcut 输入和输出 shape 一样

depthwise 1x1 conv 只能表达逐元素 shortcut 缩放：

```text
x.shape == residual_output.shape
```

因此 `_build_shortcut_scale_conv(...)` 会检查：

```text
tuple(input_layout.shape) == tuple(output_layout.shape)
```

如果 shortcut 输入和 residual 输出 shape 不一致，比如 channel 数变化、spatial size 变化、stride/downsample/projection shortcut，就不能用这个 Core B 表达。那类结构应该把 shortcut 分支自身 materialize 成真实 compute 路径，例如 shortcut conv/pool/reshape，然后再进入 `DIRECT_ADD + activation` 核。

### 9.5 为什么 M 可以到 255

`approximate_scale_ratio(...)` 默认会搜索：

```text
1 <= M <= 255
```

之前曾经考虑过把普通 `Conv2d.weight` 当 signed int8，因此错误地引入了 `M <= 127` 的限制。这个限制不应该存在，因为 shortcut 核的权重格式可以显式配置成 unsigned 8-bit：

```text
shortcut_core.core_params.set_weight_format((DataSign.UNSIGNED, DataWidth.WIDTH_8BIT))
```

同时，PAIIR 的 `OfflineCoreOp.weights` 导出不能再无条件 `to(torch.int8)`，而要根据该 core 的 `weight_sign` 选择 dtype：

```text
weight_sign == UNSIGNED -> weight.to(torch.uint8)
weight_sign == SIGNED   -> weight.to(torch.int8)
```

这样 `M=255` 在 PAIIR 和 backendv2 权重展开中都会保持为 255，而不会先变成 signed int8 的 `-1`。

当前 materialize 阶段只检查：

```text
0 < shortcut_m <= 255
```

这保留了原始 `approximate_scale_ratio` 的搜索空间，也不需要 backend 认识 `IdentityScale`。

### 9.6 backend 和 group_tile 的修改退回

之前尝试过让 backendv2 识别 `IdentityScale`，并给 `group_tile.py` 增加同 shape identity-scale group 的 tiling 逻辑。现在该方向撤回：

```text
paibox/backendv2/get_weight.py 不再识别 IdentityScale
paibox/backendv2/group_tile.py 不再增加 IdentityScale tiling 分支
```

新的 residual 路径只依赖 backendv2 已有的 `nn.Conv2d` 权重展开和已有的 potential/direct-add 逻辑。这样 backend 不需要新增 `ConvAddReLU2d` fused 算子，也不需要新增 `IdentityScale` 这种项目侧辅助算子的语义。

## 10. Core C: 膜电平累计和 ReLU/LUT

Core C 是：

```text
AccumulateOp([None, None], act, signs=(1, 1))
```

语义：

```text
input0 = Core A 输出的 POTENTIAL
input1 = Core B 输出的 POTENTIAL

acc = input0 + input1
out = act(acc)
```

因为 `comps` 全是 `None`，`AccumulateOp` 会设置：

```text
core_params.add_potential = AddPotentialMode.DIRECT_ADD
```

这告诉 backendv2：

```text
该核接收的是 32-bit 膜电平输入，而不是普通 8-bit VALUE 输入
```

backendv2 中 `CoreOpNode.set_io_bit_num` 会根据 `add_potential` 决定输入 bit 数：

```text
AddPotentialMode.NORMAL     -> input_width 决定输入 bit 数
AddPotentialMode.DIRECT_ADD -> 输入按 32 bit 膜电平处理
```

ReLU/LUT 配置来自：

```text
AccumulateOp.lut_data
    -> act.export_lut()
```

neuron attrs 来自：

```text
AccumulateOp.neuron_params
    -> act.to_neuron_params(...)
```

backendv2 会在 `CoreOpNode.attrs_part2(...)` 中把这些字段转成芯片 neuron 属性：

```text
reset_mode
reset_v
threshold_neg_mode
threshold_pos_mode
threshold_neg
threshold_pos
lateral_inhibition
leak_multi_sequence
leak_multi_input
leak_multi_mode
leak_add_mode
leak_tau
leak_v
vjt_initial
```

## 11. backendv2 消费 PAIIR 的入口

backendv2 不直接读取 `QuantizedConvAddReLU2dOp`。它只消费 materialize/fusion 后的普通 PAIIR 节点。

核心入口：

```text
paibox/backendv2/mapper.py::Mapper.compile
```

重要阶段：

```text
PAIIRGraph
    -> build_nodes(...)
    -> build_groups(...)
    -> routing / allocation
    -> get_raw_weights(...)
    -> core config / neuron attrs / weight package export
```

每个离线核在 backendv2 中会变成：

```text
CoreOpNode
```

`CoreOpNode` 会读取：

```text
raw_node.core_params
raw_node.lut_data
raw_node.weights
raw_node.neuron_params
```

这些字段分别进入：

```text
Frontend_Core_Config
LUT 配置帧
权重包
OfflineNeuFullAttrsV2Part2
```

## 12. 配置字段的数据来源

### 12.1 core_params

来源：

```text
OfflineCoreOp.core_params
```

主要字段：

```text
snn_mode
pooling_mode
add_potential
input_sign / input_width
output_sign / output_width
weight_sign / weight_width
tick_start / tick_duration / tick_initial
```

对于本 residual 三核：

```text
Core A:
  add_potential = NORMAL
  output_type = POTENTIAL

Core B:
  add_potential = NORMAL
  output_type = POTENTIAL
  snn_mode 来自 PotentialPassthroughNodeV25

Core C:
  add_potential = DIRECT_ADD
  snn_mode = ANN
  lut_data = ReLU LUT
```

### 12.2 neuron_params

来源：

```text
OfflineCoreOp.neuron_params
```

对于本 residual 三核：

```text
Core A:
  默认 potential 输出

Core B:
  PotentialPassthroughNodeV25.to_neuron_params()
  leak_multi_input = ENABLE
  leak_tau = shortcut_n
  output_type = POTENTIAL

Core C:
  act.to_neuron_params()
  ReLU/LUT 相关阈值和模式
```

### 12.3 lut_data

来源：

```text
act.export_lut()
```

对于 Core C：

```text
LutReLU / LutReLUSymmetric
    -> thresholds
    -> values
    -> backend LUT frame
```

### 12.4 weights

来源：

```text
OfflineCoreOp.weights
```

对于本 residual 三核：

```text
Core A:
  nn.Conv2d.weight
  -> conv2d_weight_matrix

Core B:
  depthwise 1x1 Conv2d.weight
  -> conv2d_weight_matrix

Core C:
  comps=[None, None]
  -> direct-add input path
  -> no conv/linear parameter weight
```

## 13. 当前注意点

### 13.1 shortcut zero-point 尚未显式 materialize

`ManualConvAddReLU2d.forward` 的数学语义是：

```text
shortcut_acc = round((x_int - x_zp) * M * 2^n)
```

但是当前 PAIIR materialize 路径表达的是：

```text
shortcut_acc ~= x_int * M * 2^n
```

也就是说：

```text
- x_zp * M * 2^n
```

这项没有作为独立 bias / leak_v / 常量补偿显式进入 PAIIR 图。

因此：

- 如果 shortcut 使用对称量化，通常 `x_zp = 0`，该问题不明显。
- 如果 shortcut 使用非对称量化，`x_zp != 0`，当前 backend 部署语义可能与 Python manual forward 不完全一致。

后续可选修正方向：

```text
1. 在 shortcut_core 的 neuron_params.leak_v 中加入 zero-point 补偿
2. 或在 Core C 的 DIRECT_ADD activation 核中加入补偿
3. 或限制该路径只允许 activation_symmetric=True / x_zp=0
```

需要注意补偿项是否是 per-tensor、per-channel，及其是否能合法广播到输出 shape。

### 13.2 conv branch bias 已写入 StandaloneCompOp.leak_v（对StandaloneCompOp修改）

`_bind_quantized_params(...)` 会把 bias 写成：

```text
bias_q = round(bias_fp32 / accum_scale)
```

并复制到 canonical `nn.Conv2d.bias`。

residual materialize 后，conv 分支是：

```text
StandaloneCompOp(conv)
```

本次修改前，`StandaloneCompOp.neuron_params` 只返回：

```text
NeuronParams(output_type=OutputType.POTENTIAL)
```

这会导致 `conv.bias` 虽然已经保存在 canonical `nn.Conv2d.bias` 中，但没有进入 backend neuron attrs 的 `leak_v`。

本次修改后，`StandaloneCompOp.neuron_params` 的普通 potential 输出路径会读取：

```text
_get_bias(self.comp)
```

并生成：

```text
NeuronParams(
    leak_v=bias if bias is not None else 0.0,
    output_type=OutputType.POTENTIAL,
)
```

因此 Core A 的数据流现在是：

```text
ManualConvAddReLU2d.conv.bias_val
    -> bias_q = round(bias_fp32 / accum_scale)
    -> canonical nn.Conv2d.bias
    -> StandaloneCompOp.neuron_params.leak_v
    -> backendv2 CoreOpNode.attrs_part2(...).leak_v
    -> OfflineNeuFullAttrsV2Part2.leak_v
```

这让 residual conv branch 与 `SequentialOp` / `AccumulateOp` 的 bias 处理保持一致：compute bias 统一通过 neuron additive leak `leak_v` 进入硬件配置。

该修改保持 MaxPool 特殊导出逻辑不变。MaxPool 没有 `bias`，仍走原有 identity spike/LUT 的特殊 `neuron_params` 分支。

新增验证点：

```text
1. StandaloneCompOp(nn.Conv2d(..., bias=True)).neuron_params.leak_v 等于 conv.bias
2. QuantizedConvAddReLU2dOp materialize 后的 conv_core.neuron_params.leak_v 等于 canonical conv.bias
3. conv_core.neuron_params.output_type 仍为 POTENTIAL
```

### 13.3 M 和 n 是近似，不是精确实数缩放

shortcut 比例来自：

```text
exact_ratio = x_scale / (conv_input_scale * weight_scale)
```

部署时近似为：

```text
M * 2^n
```

其中：

```text
1 <= M <= 255
min_n <= n <= max_n
```

这里 `M <= 255` 依赖 Core B 的显式权重配置：

```text
weight_sign  = UNSIGNED
weight_width = WIDTH_8BIT
```

因此会存在近似误差：

```text
error = abs(exact_ratio - M * 2^n)
```

该误差会影响 residual add 前的 accumulator 对齐。

后续建议：

```text
1. 在 summary/export 中记录 exact_ratio、M、n、approx_ratio、error
2. 在测试中比较 manual forward 与 PAIIR/backend simulation 的误差
3. 对过大 error 给出 warning 或拒绝部署
```

### 13.4 Quantized* 节点不应进入 backendv2

`QuantizedConvAddReLU2dOp` 是 PAIIR 适配层节点，不是 backend-ready 节点。

正确流程必须包含：

```text
materialize_quantized_ops
```

并在最终部署前确认：

```text
validate_deployable_graph
```

不会再看到 `Quantized*` 前端适配节点。

### 13.5 depthwise 1x1 shortcut 只适用于同 shape shortcut

Core B 使用：

```text
nn.Conv2d(C, C, kernel_size=1, groups=C, bias=False, weight=M)
```

表示 shortcut 分支的整数缩放部分。这个设计隐含一个前提：

```text
shortcut 输入 x 和 residual 输出具有完全相同的 NCHW shape，并且逐元素一一对应。
```

depthwise 1x1 conv 的每个通道只连自己：

```text
out[:, c, h, w] = x[:, c, h, w] * M
```

如果：

```text
input_shape != output_shape
```

则 materialize 阶段会提前报错：

```text
QuantizedConvAddReLU2dOp shortcut scale requires input/output shapes to match
```

因此当前 `ManualConvAddReLU2d -> PAIIR` 路径适用于 identity shortcut，不适用于需要投影的 shortcut，例如：

```text
1x1 conv shortcut
stride shortcut
channel 数变化
spatial size 变化
```

这类情况不应使用当前的 depthwise 1x1 identity shortcut 表达，而应把 shortcut 分支自身 materialize 成真实 compute 路径，例如：

```text
shortcut conv/pool/reshape -> POTENTIAL
```

再进入 `DIRECT_ADD + activation` 核。

当前已经在 `materialize_quantized_ops` 阶段提前检查：

```text
shortcut_input_shape == residual_output_shape
```

这样可以在 PAIIR 阶段更早报错，而不是等 backendv2 展开 weight 时才发现 shape 不匹配。

## 14. 推荐排查方式

如果需要确认某个模型是否按预期走三核路径，可以按以下顺序检查：

```text
1. convert_fx_to_manual 后：
   模型里是否存在 ManualConvAddReLU2d

2. torch_to_paiir 后：
   图里是否存在 QuantizedConvAddReLU2dOp

3. materialize_quantized_ops 后：
   QuantizedConvAddReLU2dOp 是否消失
   是否出现 StandaloneCompOp(conv)
   是否出现 SequentialOp(depthwise 1x1 Conv2d, PotentialPassthroughNodeV25)
   是否出现 PotentialAddOp + StandaloneActOp

4. compile_to_paiir 后：
   是否出现 AccumulateOp
   AccumulateOp.core_params.add_potential 是否为 DIRECT_ADD
   AccumulateOp.comps 是否为 [None, None]

5. backendv2 Mapper.compile 后：
   是否生成对应 routing group / core placement
   shortcut 核是否为 groups=C 的 1x1 Conv2d，且每个通道 weight 是否为 M
   shortcut 核 neuron attrs 中 leak_tau 是否为 n
   direct-add 核 add_potential 是否为 DIRECT_ADD
```

## 15. 一句话总结

本次改动把 `ConvAddReLU2d` 量化残差块从项目私有 manual module，接入到标准 PAIIR 编译链路：

```text
ManualConvAddReLU2d
    -> QuantizedConvAddReLU2dOp
    -> conv POTENTIAL 核
    -> shortcut M * 2^n POTENTIAL 核
    -> DIRECT_ADD + ReLU/LUT 核
    -> backendv2 标准配置和权重导出
```

这样 backendv2 不需要新增 `ConvAddReLU2d` fused 算子，也不需要认识 `IdentityScale`。它只需要继续处理已有的 `StandaloneCompOp`、`SequentialOp`、`AccumulateOp` 和普通 `nn.Conv2d`。
