<div align="center">

# PAIBox使用指南

</div>

## 快速上手

PAIBox使用 `pyproject.toml` 管理依赖。若使用Poetry：

```bash
poetry install
```

或者采用开发版环境

```bash
poetry install --with dev
```

若使用conda等，则手动安装如下依赖至Python虚拟环境：

```toml
python = "^3.8"
pydantic = "^2.0"
numpy = "^1.24.0"
paicorelib = "^1.1.1"
orjson = "^3.10.1" # Optional
```

通过pip安装PAIBox：

```bash
pip install paibox
```

添加 `--pre` 或克隆 `dev` 分支以使用开发版

```bash
git clone -b dev https://github.com/PAICookers/PAIBox.git
cd PAIBox
```

可查看版本号以确认安装：

```python
import paibox as pb

print(pb.__version__)
>>> x.y.z
```

## 基本组件

PAIBox提供**神经元**与**突触**作为基本组件，用于搭建神经网络。

结合**输入节点**，可以对输入数据进行脉冲编码，并传入网络中进行仿真推理。

### 神经元

PAIBox提供了多种类型的神经元模型，能够实现各种特殊的功能。

神经元均支持 `delay`，`tick_wait_start`，`tick_wait_end`，`keep_shape`，`unrolling_factor` 参数。

⚠️ 神经元初始膜电位为0。

#### IF

IF神经元实现了经典的“积分发射”模型，其调用方式及参数如下：

```python
import paibox as pb

n1 = pb.IF(shape=10, threshold=127, reset_v=0, keep_shape=False, delay=1, tick_wait_start=1, tick_wait_end=0, name='n1')
```

其中：

- `shape`：代表神经元组的尺寸，其形式可以是整形标量、元组或列表。
- `threshold`：神经元阈值，其形式为整数。
- `reset_v`：神经元的重置膜电位。
- `delay`：设定神经元输出的延迟。默认为1，即本时间步的计算结果，**下一时间步**传递至后继节点。
- `tick_wait_start`：设定神经元启动时间。神经元将在第 `T` 个时间步时启动。0表示不启动。默认为1。
- `tick_wait_end`：设定神经元持续工作时长。神经元将持续工作 `T` 个时间步。0表示**持续工作**。默认为0。
- `unrolling_factor`：该参数与后端流程相关。展开因子表示神经元将被展开，部署至更多的物理核上，以降低延迟并提高吞吐率。
- `keep_shape`：是否在仿真记录数据时保持尺寸信息，默认为 `False`。实际进行运算的尺寸仍视为一维。
- `name`：神经元的名称。可选参数。

#### LIF

LIF神经元实现了“泄露-积分-发射”神经元模型，其调用方式及参数如下：

```python
n1 = pb.LIF(shape=128, threshold=127, reset_v=0, leak_v=-1, keep_shape=False, name='n1')
```

- `leak_v`：LIF神经元的泄露值（有符号）。其他参数含义与IF神经元相同。

#### Tonic Spiking

Tonic Spiking神经元可以实现对持续脉冲刺激的周期性反应。

```python
n1 = pb.TonicSpiking(shape=128, fire_step=3, keep_shape=False, name='n1')
```

- `fire_step`：发放周期，每接收到 `N` 次刺激后发放脉冲。

#### Phasic Spiking

Phasic Spiking神经元可以实现，在接受一定数量脉冲后发放，然后保持静息状态，不再发放。

```python
n1 = pb.PhasicSpiking(shape=128, time_to_fire=3, neg_floor=10, keep_shape=False, name='n1')
```

- `time_to_fire`：发放时间。
- `neg_floor`：地板阈值，静息时的膜电位为其负值。

#### Always1Neuron

Always1神经元在工作期间持续输出1。

#### Spiking Relu

SNN模式下，具有Relu功能的神经元。当输入为1，则输出为1；输入为0，则输出为0。

### 突触

#### 全连接

PAIBox中，突触用于连接不同神经元组，并包含了连接关系以及权重信息。以全连接类型的突触为实例：

```python
s1= pb.FullConn(source=n1, dest=n2, weights=weight1, conn_type=pb.SynConnType.All2All, name='s1')
```

其中：

- `source`：前向神经元组，可以是**神经元或者输入节点**类型。
- `dest`：后向神经元组，只能为**神经元**类型。
- `weights`：突触的权重。
- `conn_type`：连接形式，默认为 `MatConn` 矩阵连接。当设置为 `All2All`、`One2One` 或 `Identity` 时，`weights` 有更简洁的表达。
- `name`：可选，为该对象命名。

突触表达的是两个神经元组之间的连接关系。PAIBox提供了三种主要的连接关系表达：

- `All2All`：全连接
- `One2One`：单对单连接
- `Identity`：恒等映射，同单对单连接，权重为缩放因子标量
- `MatConn`：普通的矩阵连接

通常情况下，`MatConn` 适合所有的连接关系，而 `All2All`、`One2One` 则提供了对于特殊连接更为方便的表达。

##### All2All 全连接

对于全连接，其权重 `weights` 有两种输入类型：

- 标量：默认为1。例如，设置为 `X`，则权重矩阵实际为元素均为 `X` 的 `N1*N2` 矩阵。
- 矩阵：尺寸为 `N1*N2` 的矩阵。

其中，`N1` 为前向神经元组数目，`N2` 为后向神经元组数目。

##### One2One 单对单连接

两组神经元之间依次单对单连接，这要求**前向与后向神经元数目相同**。其权重 `weights` 主要有以下几种输入类型：

- 标量：默认为1。这表示前层的各个神经元输出线性地输入到后层神经元，即 $\lambda\cdot\mathbf{I}$

  ```python
  n1 = pb.IF(shape=5, threshold=1)
  n2 = pb.IF(shape=5, threshold=1)
  s1 = pb.FullConn(source=n1, dest=n2, conn_type=pb.SynConnType.One2One, weights=2, name='s1')

  print(s1.weights)
  >>>
  2
  ```

  其权重以标量的形式储存。

- 数组：尺寸要求为 `(N2,)`，可以自定义每组对应神经元之间的连接权重。如下例所示，设置 `weights` 为 `[1, 2, 3, 4, 5]`，

  ```python
  n1 = pb.IF(shape=5, threshold=1)
  n2 = pb.IF(shape=5, threshold=1)
  s1 = pb.FullConn(source=n1, dest=n2, conn_type=pb.SynConnType.One2One, weights=np.arange(1, 6, dtype=np.int8), name='s1')

  print(s1.weights)
  >>>
  [[1, 0, 0, 0, 0],
   [0, 2, 0, 0, 0],
   [0, 0, 3, 0, 0],
   [0, 0, 0, 4, 0],
   [0, 0, 0, 0, 5]]
  ```

  其权重实际上为 `N*N` 矩阵，其中 `N` 为前向/后向神经元组数目。

##### Identity 恒等映射

具有缩放因子的单对单连接，即 `One2One` 中权重项为标量的特殊情况。

##### MatConn 一般连接

普通的神经元连接类型，仅可以通过矩阵设置其权重 `weights`。

#### 1D卷积

全展开形式1D卷积为全连接突触的一种特殊表达。需**严格指定**输入神经元的尺寸与维度、卷积核权重、卷积核维度顺序与步长。对于输出神经元的具体尺寸不做严格要求。

- `kernel`：卷积核权重。
- `stride`：步长，标量。默认为1。
- `padding`：填充，标量。
- `kernel_order`：指定卷积核维度顺序为 `OIL` 或 `IOL` 排列。默认为 `OIL`。
- 神经元维度顺序仅支持 `CL`。

```python
n1 = pb.IF(shape=(8, 28), threshold=1)      # Input feature map: (8, 28)
n2 = pb.IF(shape=(16, 26), threshold=1)     # Output feature map: (16, 26)
kernel = np.random.randint(-128, 128, size=(16, 8, 3), dtype=np.int8) # OIL

conv1d = pb.Conv1d(n1, n2, kernel=kernel, stride=1, padding=0, kernel_order="OIL", name="conv1d_1")
```

⚠️ `padding` 目前不支持，默认为0。

#### 2D卷积

全展开形式2D卷积为全连接突触的一种特殊表达。需**严格指定**输入神经元的尺寸与维度、卷积核权重、卷积核维度顺序与步长。对于输出神经元的具体尺寸不做严格要求。

- `kernel`：卷积核权重。
- `stride`：步长，标量或元组格式。当为标量时，对应为 `(x, x)`；当为元组时，则对应为 `(x, y)`。默认为1。
- `padding`：填充，可以为标量或元组。当为标量时，对应为 `(x, x)`；当为元组时，则对应为 `(x, y)`。
- `kernel_order`：指定卷积核维度顺序为 `OIHW` 或 `IOHW` 排列。默认为 `OIHW`。
- 神经元维度顺序仅支持 `CHW`。

```python
n1 = pb.IF(shape=(8, 28, 28), threshold=1)      # Input feature map: (8, 28, 28)
n2 = pb.IF(shape=(16, 26, 26), threshold=1)     # Output feature map: (16, 26, 26)
kernel = np.random.randint(-128, 128, size=(16, 8, 3, 3), dtype=np.int8) # OIHW

conv2d = pb.Conv2d(n1, n2, kernel=kernel, stride=(1, 2), padding=1, kernel_order="OIHW", name="conv2d_1")
```

#### 1D转置卷积

全展开形式1D转置卷积为全连接突触的一种特殊表达。需**严格指定**输入神经元的尺寸与维度、卷积核权重、卷积核维度顺序与步长。对于输出神经元的具体尺寸不做严格要求。

对于 `Conv1d`：

- `kernel`：卷积核权重。
- `stride`：步长，标量。
- `padding`：填充，标量。
- `output_padding`：对输出特征图的一侧进行额外的填充，标量。
- `kernel_order`：指定卷积核维度顺序为 `OIL` 或 `IOL` 排列。
- 神经元维度顺序仅支持 `CL`。
- 参数详细含义参见：[pytorch/ConvTranspose1d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html#torch.nn.ConvTranspose1d)

```python
n1 = pb.IF(shape=(8, 28), threshold=1)      # Input feature map: (8, 28)
n2 = pb.IF(shape=(16, 29), threshold=1)     # Output feature map: (16, 29)
kernel = np.random.randint(-128, 128, size=(16, 8, 3), dtype=np.int8) # OIL

convt1d = pb.ConvTranspose1d(n1, n2, kernel=kernel, stride=1, padding=0, output_padding=1, kernel_order="OIL", name="convt1d_1")
```

#### 2D转置卷积

全展开形式2D转置卷积为全连接突触的一种特殊表达。需**严格指定**输入神经元的尺寸与维度、卷积核权重、卷积核维度顺序与步长。对于输出神经元的具体尺寸不做严格要求。

- `kernel`：卷积核权重。
- `stride`：步长，可以为标量或元组。当为标量时，对应为 `(x, x)`；当为元组时，则对应为 `(x, y)`。
- `padding`：填充，可以为标量或元组。当为标量时，对应为 `(x, x)`；当为元组时，则对应为 `(x, y)`。
- `output_padding`：对输出特征图的一侧进行额外的填充，可以为标量或元组。当为标量时，对应为 `(x, x)`；当为元组时，则对应为 `(x, y)`。
- `kernel_order`：指定卷积核维度顺序为 `OIHW` 或 `IOHW` 排列。
- 神经元维度顺序仅支持 `CHW`。
- 参数详细含义参见：[pytorch/ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d)

```python
n1 = pb.IF(shape=(8, 28, 28), threshold=1)      # Input feature map: (8, 28, 28)
n2 = pb.IF(shape=(16, 55, 55), threshold=1)     # Output feature map: (16, 55, 55)
kernel = np.random.randint(-128, 128, size=(16, 8, 3, 3), dtype=np.int8) # OIHW

convt2d = pb.ConvTranspose2d(n1, n2, kernel=kernel, stride=2, padding=1, output_padding=0, kernel_order="OIHW", name="convt2d_1")
```

### 编码器

PAIBox提供了有状态与无状态编码器。其中，有状态编码器是指编码过程与时间有关，将输入数据编码到一段时间窗口内。而无状态编码器是指编码过程与时间无关，每个时间步，都可以根据输入数据进行编码。

⚠️ 请注意，我们只提供较为简单的编码器，以便用户在不依赖外部库的条件下实现基本编码操作；如果需要更复杂的编码，请直接使用。

#### 无状态编码器

##### 泊松编码

泊松编码是一种常用的无状态编码。以下为一个简单实例：

```python
seed = 1
rng = np.random.RandomState(seed=seed)
x = rng.rand(10, 10).astype(np.float32)
pe = pb.simulator.PoissonEncoder(seed=seed)
out_spike = np.full((20, 10, 10), 0)

for t in range(20):
    out_spike[t] = pe(x)
```

通过调用该编码器，将需编码数据传入，即可得到编码后结果。

##### 直接编码

直接编码使用2D卷积进行特征提取，经过LIF神经元进行编码 。`Conv2dEncoder` 使用示例如下：

```python
kernel = np.random.uniform(-1, 1, size=(1, 3, 3, 3)).astype(np.float32) # OIHW
stride = (1, 1)
padding = (1, 1)
de = pb.simulator.Conv2dEncoder(
    kernel,
    stride,
    padding,
    "OIHW",
    tau=2,
    decay_input=True,
    v_threshold=1,
    v_reset=0.2,
)
x = np.random.uniform(-1, 1, size=(3, 28, 28)).astype(np.float32) # CHW

for t in range(20):
    out_spike = de(x)
```

其中，

- `kernel`：卷积核权重。
- `stride`：步长，可以为标量或元组。当为标量时，对应为 `(x, x)`；当为元组时，则对应为 `(x, y)`。
- `padding`：对输入进行填充，可以为标量或元组。当为标量时，对应为 `(x, x)`；当为元组时，则对应为 `(x, y)`。
- `kernel_order`：指定卷积核维度顺序为 `OIHW` 或 `IOHW` 排列。
- `tau`：膜电位时间常数。
- `decay_input`：输入是否也会参与衰减。
- `v_threshold`：阈值电平。
- `v_reset`：复位电平。
- 待编码数据维度顺序仅支持 `CHW`。

其中，所使用的LIF为SpikingJelly内的 `SimpleLIFNode`。具体原理参见：[SpikingJelly/SimpleLIFNode](https://spikingjelly.readthedocs.io/zh-cn/latest/sub_module/spikingjelly.activation_based.neuron.html#spikingjelly.activation_based.neuron.SimpleLIFNode)。如果需要使用更复杂的编码，请直接使用。

#### 有状态编码器

有状态编码器类别较多。PAIBox提供了几种有状态编码器：周期编码器 `PeriodicEncoder`、延迟编码器 `LatencyEncoder` 。

##### 周期编码器

它以一段脉冲序列为输入，将其循环地在每一个时间步输出。以下为一个简单实例：

```python
# 定义一段脉冲序列
spike = np.full((5, 3), 0)
spike[0, 1] = 1
spike[1, 0] = 1
spike[4, 2] = 1

# 实例化周期性编码器
pe = pb.simulator.PeriodicEncoder(spike)

out_spike = np.full((20, 3), 0)
for t in range(20):
    out_spike[t] = pe()
```

这将仿真20个时间步，周期性地获取输入的脉冲序列并将其输出。

##### 延迟编码器

根据输入数据 `x` 延迟发放脉冲的编码器。当刺激强度越大，发放时间越早，且存在最大脉冲发放时间 `T`。因此对于每一个输入数据，都能得到一段时间步长为 `T` 的脉冲序列，每段序列有且仅有一个脉冲发放。编码类型可为：`linear` 或 `log`。以下为一个简单实例：

```python
N = 6
x = np.random.rand(N)
T = 20

le = pb.simulator.LatencyEncoder(T, "linear")

out_spike = np.zeros((T, N), dtype=np.bool_)
for t in range(T):
    out_spike[t] = le(x)
```

具体编码原理参见：[SpikingJelly/延迟编码器](https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based/2_encoding.html#id5)

### 输入节点

为了支持多样的数据输入形式，同时标明网络模型的输入节点，PAIBox设计了输入节点这一组件。

输入节点可以使用以下方法定义：

```python
inp = pb.InputProj(input=1, shape_out=(4, 4), keep_shape=True, name='inp1')
```

其中，

- `input`：输入节点的输入，支持数值型或函数型输入。即可以为**整型、数组或可调用对象**（函数或者实现了 `__call__` 的对象，例如，编码器等），也可以在例化节点时设置为 `None`。
- `shape_out`：输出数据的尺寸。
- `keep_shape`：在观测节点输出数据时，可以通过该参数确定输出是否保持原始的维度信息，从而更好地进行监测。默认为 `True`。
- `name`：可选参数，为该节点命名。

当在例化输入节点时设置 `input=None`，则可通过如下方式设置输入数据，但**仅限于设置为数值型数据**：

```python
inp.input = np.ones((4, 4), dtype=np.int8)
```

#### 数据类型输入

当输入节点的输出为常量时，可以直接设置 `input` 为常量，并将 `shape_out` 设置为所需输出尺寸。以下为一个简单实例：

```python
# 实例化一个输入节点，使其一直输出2，设置输出尺寸为4*4，并保持其维度信息。
inp = pb.InputProj(2, shape_out=(4, 4), keep_shape=True)

prob = pb.simulator.Probe(inp, "feature_map")  # 实例化一个探针，并观察该输入节点的特征图信息
sim = pb.Simulator(inp)                        # 例化一个仿真器
sim.add_probe(prob)                            # 将探针加入仿真器中
sim.run(2)                                     # 仿真2个时间步

output = sim.data[prob][-1]                    # 获取最后一个时间步的探针数据
print(output)

>>>
    [[2 2 2 2]
    [2 2 2 2]
    [2 2 2 2]
    [2 2 2 2]]

# If keep_shape=False
>>>
[2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
```

当启用 `keep_shape` 时，特征图数据将保持其维度信息。

输入节点的参数也可以是数组。以下为一个简单实例：

```python
x = np.random.randint(0, 5, size=(4, 4))
inp = pb.InputProj(x, shape_out=(4, 4), keep_shape=True)
prob = pb.simulator.Probe(inp, "feature_map")
sim = pb.Simulator(inp)
sim.add_probe(prob)
sim.run(2)

output = sim.data[prob][-1]
print(output)

>>>
    [[4 4 4 0]
    [3 0 2 0]
    [4 0 2 0]
    [2 4 2 4]]
```

#### 函数类型输入

输入节点支持使用函数作为输入。以下为一个简单实例：

```python
def fakeout(*args, **kwargs):
    return np.random.randint(-128, 128, size=(4, 4), dtype=np.int8)

inp = pb.InputProj(fakeout, shape_out=(4, 4), keep_shape=True)
prob = pb.simulator.Probe(inp, "feature_map")
sim = pb.Simulator(inp)
sim.add_probe(prob)
sim.run(2)

output = sim.data[prob][-1]
print(output)

>>>
[[3 3 3 3]
 [3 3 3 3]
 [3 3 3 3]
 [3 3 3 3]]
```

当函数需要时间步信息，则可在函数中声明传入参数 `t` ，输入节点则在前端环境变量 `FRONTEND_ENV` 中获取时间步信息。当需要传入额外的参数时，通过 `FRONTEND_ENV.save()` 保存相关参数至前端环境变量。当函数与时间步或其他参数无关时，需使用 `**kwargs` 代替。以下为一个简单实例：

```python
from paibox import FRONTEND_ENV

def fakeout_with_t(t, bias, **kwargs): # ignore other arguments except `t` & `bias`
    return np.ones((4, 4)) * t + bias

inp = pb.InputProj(input=fakeout_with_t, shape_out=(4, 4), keep_shape=True)
prob = pb.simulator.Probe(inp, "feature_map")

sim = pb.Simulator(inp)
sim.add_probe(prob)
FRONTEND_ENV.save(bias=3) # Passing `bias` to function `fakeout_with_t`
sim.run(4)

output = sim.data[prob][-1]
print(output)

>>>
    # t=3
    [[2 2 2 2]
    [2 2 2 2]
    [2 2 2 2]
    [2 2 2 2]]
    # t=4
    [[3 3 3 3]
    [3 3 3 3]
    [3 3 3 3]
    [3 3 3 3]]
```

当仿真时间不同时，输出结果也不同，表明输入节点的输出与时间步相关。

#### 编码器类型输入

PAIBox提供了一些常用编码器，编码器内部实现了 `__call__` 方法，因此可作为输入节点的输入使用。在作为输入节点的输入使用时，它与一般函数做为输入节点的输入使用存在差别。

在例化 `InputProj` 时，输入节点的输入为编码器。在运行时，还需要通过设置 `inp.input`，**向输入节点输入待编码数据**，节点内部将完成编码并输出。以泊松编码器为例：

```python
pe = pb.simulator.PoissonEncoder()                          # 例化泊松编码器
inp = pb.InputProj(pe, shape_out=(4, 4), keep_shape=True)   # 例化输入节点
input_data = np.random.rand(4, 4).astype(np.float32)        # 生成归一化数据

sim = pb.Simulator(inp)
prob = pb.simulator.Probe(inp, "feature_map")
sim.add_probe(prob)

inp.input = input_data	# 传入数据至输入节点
sim.run(3)

output = sim.data[prob][-1]
print(output)

>>>
    [[ True False  True  True]
    [ True False  True  True]
    [False  True  True  True]
    [ True  True  True False]]
```

## 功能模块

多个基础组件可以组成具有特定功能的模块(module)。在实例化与仿真中，它们作为一个整体，而在后端中则会被拆解，并由多个基础组件构建。不同于基本的神经网络组件——神经元与突触的常规使用模式：

- 在实例化时，其运作机制借鉴了“突触”的连接特性，要求在初始化阶段指定与其相连的前级神经元作为输入源，这些神经元的输出信号将作为进行逻辑运算的操作数。
- 该模块完成运算后的输出效果，则表现为一个具有独立输出能力的“神经元”，其输出接口的设计完全符合神经元的标准形式。这意味着其输出脉冲可作为后继突触的输入。
- 后端构建时，模块将拆分成一或多个神经元节点与突触。所构建的基础组件尺寸由模块连接的操作数尺寸决定。

功能模块均支持 `delay`，`tick_wait_start`，`tick_wait_end`，`keep_shape` 参数。

### 逻辑运算

逻辑运算模块实现了 `numpy` 中的位逻辑运算操作（例如 `&` 与 `numpy.bitwise_and` 等），可对接收到的一或多个输出脉冲进行逻辑运算，并产生脉冲输出。PAIBox提供了逻辑与、或、非、异或：`BitwiseAND`，`BitwiseOR`，`BitwiseNOT`，`BitwiseXOR`。以位与为例：

```python
import paibox as pb

class Net(pb.DynSysGroup):
    def __init__(self):
        super().__init__()
        self.n1 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=1)
        self.n2 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=1)
        self.and1 = pb.BitwiseAND(self.n1, self.n2, delay=1, tick_wait_start=2)
        self.n3 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=self.and1.tick_wait_start + self.and1.external_delay + 1)
        self.s3 = pb.FullConn(self.and1, self.n3, conn_type=pb.SynConnType.All2All)
```

其中：

- `neuron_a`：第一个操作数。
- `neuron_b`：第二个操作数。
- `delay`：设定模块输出的延迟。默认为1，即本时间步的计算结果，**下一时间步**传递至后继节点。
- `tick_wait_start`：设定模块启动时间。模块将在第 `T` 个时间步时启动。0表示不启动。默认为1。
- `tick_wait_end`：设定模块持续工作时长。模块将持续工作 `T` 个时间步。0表示**持续工作**。默认为0。
- `keep_shape`：是否在仿真记录数据时保持尺寸信息，默认为 `False`。实际进行运算的尺寸仍视为一维。
- `name`：模块的名称。可选参数。

⚠️ 模块的属性 `external_delay` 用于表示其相对于外部的内部固有延迟。这是由具体的后端构建形式决定的，不可更改。上述示例中，位与计算结果将输出至 `n3` 中。默认情况下，`n3` 将在位与计算结果输出后启动，因此其启动时间为 `and1` 的启动时间+固有延迟+1。

### 延迟链

用于实现神经元延迟输出。使用方式如下：

```python
n1 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=1)
n1_delay_out = pb.DelayChain(n1, chain_level=5, delay=1, tick_wait_start=2)
n2 = pb.SpikingRelu((10,), delay=1, tick_wait_start=n1_delay_out.tick_wait_start + n1_delay_out.external_delay)
```

其中：

- `neuron`：进行延迟输出的神经元。
- `chain_level`：延迟链的级数，即延迟的时间步。注意，这与 `delay` 含义不同：延迟链内部会建立多级神经元（类似buffer），以实现数据的延迟传递，而 `delay` 会使得神经元输出寄存的位置延后，后继节点的启动时间需要提前，这将导致其在前级**有效输出**前就进行了计算。

### 平均/最大池化

目前仅提供2D池化：`SpikingAvgPool2d`、`SpikingMaxPool2d`。以平均池化为例：

```python
ksize = (3, 3)
stride = None # default is ksize
n1 = pb.SpikingRelu(shape, tick_wait_start=1)
pool2d = pb.SpikingAvgPool2d(n1, ksize, None, tick_wait_start=2)
n2 = pb.SpikingRelu(pool2d.shape_out, delay=1, tick_wait_start=3)
s3 = pb.FullConn(pool2d, n2, conn_type=pb.SynConnType.One2One)
```

其中：

- `neuron`：待池化的神经元。
- `kernel_size`：池化窗口的尺寸，标量或元组格式。当为标量时，对应为 `(x, x)`；当为元组时，则对应为 `(x, y)`。
- `stride`：步长，可选参数，标量或元组格式，默认为 `None`，即池化窗口的尺寸。
- 神经元维度顺序仅支持 `CHW`。

### 脉冲加、减

脉冲加减法与数的加减法存在差异。对脉冲进行加减，运算结果将在较长时间步上体现。例如，在 `T=1` 时刻两神经元均输出1，则将在 `T=2,3` 时刻产生输出脉冲。以下为脉冲加减法运算示例。其中，输入为 `T=12` 脉冲序列，输出为 `T=20` 脉冲序列。

```python
inpa = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1], np.bool_)
inpb = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0], np.bool_)

# 脉冲加结果
>>> np.array([0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], np.bool_)
# 脉冲减结果
>>> np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], np.bool_)
```

`SpikingAdd`，`SpikingSub` 的使用方式与逻辑运算模块相同：

```python
n1 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=1)
n2 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=1)
add1 = pb.SpikingAdd(n1, n2, overflow_strict=False, delay=1, tick_wait_start=2) # n1 + n2
sub1 = pb.SpikingSub(n1, n2, overflow_strict=False, delay=1, tick_wait_start=2) # n1 - n2
```

其中：

- `neuron_a`：第一个操作数。
- `neuron_b`：第二个操作数。在减法中作被减数。
- `overflow_strict`：是否严格检查运算结果溢出。如果启用，则在仿真中，当脉冲加、减运算结果溢出时将报错。默认为 `False`。

### 转置

PAIBox提供了转置模块 `Transpose2d`，`Transpose3d`，用于实现二维、三维矩阵的转置。对于转置，需要**指定**输入神经元的尺寸、转置顺序（仅三维转置需要）。使用方法与逻辑运算模块相同：

```python
n1 = pb.IF((32, 16), 1, 0, delay=1, tick_wait_start=1)
t2d = pb.Transpose2d(n1, tick_wait_start=2)

n2 = pb.IF((32, 16, 24), 1, 0, delay=1, tick_wait_start=1)
t3d = pb.Transpose3d(n2, axes=(1, 2, 0), tick_wait_start=2)
```

其中：

- `neuron`：待转置其输出脉冲的神经元。对于二维转置，支持输入尺寸为1或2维；对于三维转置，支持输入尺寸为2或3维。尺寸不足时，自动补1。
- `axes`：（仅三维转置）如果指定，则必须是包含 `[0,1,…,N-1]` 排列的元组或列表，其中 `N` 是矩阵的轴（维度）数。返回数组的第 `i` 轴将对应于输入的编号为 `axes[i]` 的轴。若未指定，则默认为 `range(N)[::-1]`，这将反转轴的顺序。具体参数含义参见：[numpy.transpose](https://numpy.org/doc/1.26/reference/generated/numpy.transpose.html#numpy.transpose)

## 网络模型

在PAIBox中，可以通过继承 `DynSysGroup`（或 `Network`）来实现，并在其中例化基础组件与功能模块，完成网络模型的构建。以一个简单的两层全连接网络为例：

<p align="center">
    <img src="images/Guide-基础网络搭建-全连接网络示例.png" alt="基础网络搭建-全连接网络示例" style="zoom:50%">
</p>

### 网络构建

要构建上述网络，首先继承 `pb.Network` 并在子类 `fcnet` 中初始化网络。先例化输入节点 `i1` 与两个神经元组 `n1`、 `n2`，然后例化两个突触 `s1`、 `s2` ，将三者连接起来。其中，输入节点为泊松编码器。

```python
import paibox as pb

class fcnet(pb.Network):
    def __init__(self, weight1, weight2):
        super().__init__()
        pe = pb.simulator.PoissonEncoder()

        self.i1 = pb.InputProj(input=pe, shape_out=(784,))
        self.n1 = pb.IF(128, threshold=128, reset_v=0, tick_wait_start=1)
        self.n2 = pb.IF(10, threshold=128, reset_v=0, tick_wait_start=2)
        self.fc1 = pb.FullConn(self.i1, self.n1, weights=weight1, conn_type=pb.SynConnType.All2All)
        self.fc2 = pb.FullConn(self.n1, self.n2, weights=weight2, conn_type=pb.SynConnType.All2All)
```

### 容器类型

PAIBox提供 `NodeList`、`NodeDict` 容器类型，可批量化操作网络基本组件。例如，

```python
import paibox as pb
l1 = pb.NodeList()

for i in range(5):
    l1.append(pb.IF(10, threshold=5, reset_v=0))

for i in range(5):
    l1.append(pb.LIF(10, threshold=5, reset_v=0))
```

如此，我们共例化了10个神经元，包括5个IF神经元、5个LIF神经元。在容器内的基本组件可通过下标进行访问、与其他基本组件连接。这与一般容器类型的用法相同。

### 嵌套网络

有时网络中会重复出现类似的结构，这时先构建子网络，再多次例化复用是个不错的选择。

```python
from typing import Optional
import paibox as pb

class ReusedStructure(pb.Network):
    def __init__(self, weight, tws, name: Optional[str] = None):
        super().__init__(name=name)

        self.pre_n = pb.LIF((10,), 10, tick_wait_start=tws)
        self.post_n = pb.LIF((10,), 10, tick_wait_start=tws+1)
        self.fc = pb.FullConn(
            self.pre_n, self.post_n, conn_type=pb.SynConnType.All2All, weights=weight
        )

class Net(pb.Network):
    def __init__(self, w1, w2):
        super().__init__()

        self.inp1 = pb.InputProj(1, shape_out=(10,))
        self.subnet1 = ReusedStructure(w1, tws=1, name="Reused_Struct_0")
        self.subnet2 = ReusedStructure(w2, tws=3, name="Reused_Struct_1")
        self.fc1 = pb.FullConn(
            self.inp1,
            self.subnet1.pre_n,
            conn_type=pb.SynConnType.One2One,
        )
        self.fc2 = pb.FullConn(
            self.subnet1.post_n,
            self.subnet2.pre_n,
            conn_type=pb.SynConnType.One2One,
        )

w1 = ...
w2 = ...
net = Net(w1, w2)
```

上述示例代码中，我们先创建需复用的子网络 `ReusedStructure`，其结构为 `pre_n` -> `fc` -> `post_n`。而后，在父网络 `Net` 中实例化两个子网络 `subnet1`、 `subnet2`，并与父网络其他部分连接，此时网络结构为：`inp1` -> `fc1` -> `subnet1` -> `fc22` -> `subnet2`。上述示例为一个二级嵌套网络，对于三级或更高级嵌套网络，可参考上述方式构建。

## 仿真

### 仿真器

例化网络后，即可构建仿真器：

```python
w1 = ...
w2 = ...
fcnet = fcnet(w1, w2)
sim = pb.Simulator(fcnet, start_time_zero=False)
```

其中，有如下选项可以配置：

- `target`：网络模型，必须是 `DynamicSys` 类。
- `start_time_zero`：为保持与实际硬件行为的一致性，仿真默认**从时间步1时刻**开始（`T>0` 网络模型才开始工作）。默认为 `False`。

### 探针

在仿真过程中，用户需要检测某一层神经元的膜电位或输出、突触的输出等信息，这可通过**探针**实现。探针告知仿真器在仿真时需要记录哪些信息。使用探针的方式有如下两种：

1. 在构建网络时，直接设置探针，即在网络内部例化探针对象。
2. 在外部例化探针，并调用 `add_probe` 将其添加至仿真器内。仿真器内部将保存所有探针对象。
3. 调用 `remove_probe` 方法可移除探针及其仿真数据。

例化探针时需指定：

- `target`: 监测对象，必须是 `DynamicSys` 类。
- `attr`：监测的属性，如 `spike`、`output` 等，字符串类型，这将监测属性 `target.attr`。

基于上述仿真示例，我们添加几个探针：

```python
class fcnet(pb.Network):
    def __init__(self, weight1, weight2):
        ...
        # 内部探针，记录神经元n1的输出脉冲
        self.probe1 = pb.simulator.Probe(target=self.n1, attr="spike")

fcnet = fcnet(w1, w2)
sim = pb.Simulator(fcnet)

# 外部探针，记录神经元n1的膜电位
probe2 = pb.simulator.Probe(fcnet.n1, "voltage")
sim.add_probe(probe2)
```

可监测的对象包括网络内部所有的属性。例如，神经元及突触的各类属性，常用的监测对象包括：

- 输入节点的 `feature_map`。
- 神经元：脉冲输出 `spike` 、基于硬件寄存器的**输出** `output`（大小为 `256*N` ）、特征图形式的脉冲输出 `feature_map `、膜电位 `voltage`。
- 突触：输出 `output`。

### 仿真机理

在设置完探针后，可为每个输入节点单独输入数据，并进行仿真。仿真结束后可通过 `sim.data[...]` 读取探针监测的数据。

PAIBox仿真器的仿真行为与实际硬件保持一致，在全局时间步 `T>0` 时，网络模型才开始工作。即在例化仿真器时若指定 `start_time_zero=False`，得到的仿真数据 `sim.data[probe1][0]` 为 `T=1` 时刻的模型输出。若指定 `start_time_zero=True`，则仿真数据 `sim.data[probe1][0]` 为 `T=0` 时刻的模型输出（初态）。

```python
# 准备输入数据
input_data = np.random.rand(28, 28).astype(np.float32)
input_data2 = np.random.rand(28, 28).astype(np.float32)

fcnet.inp.input = input_data
sim.run(10, reset=True)

fcnet.inp.input = input_data2
sim.run(5)  # Change the input raw data & continue to simulate for 5 timesteps

# 读取仿真数据
n1_spike_data = sim.data[fcnet.probe1]
n1_v_data = sim.data[fcnet.probe2]
# 重置仿真器
sim.reset()
```

调用 `run` 运行仿真，其中：

- `duration`：指定仿真时间步长。请注意，仿真时需要计算网络的最长路径(delay)，并计入仿真步长中以获取有效的输出。
- `reset`：是否对网络模型中组件进行复位。默认为 `False`。这可实现在一次仿真的不同时间步，输入不同的数据。

## 编译、映射与导出

模型映射将完成网络拓扑解析、映射、分配路由坐标、生成配置文件或帧数据，并最终导出为 `.bin` 或 `.npy` 格式交换文件等一系列工作。

例化 `Mapper`，传入所构建的网络模型，进行编译，最后导出帧即可。

```python
mapper = pb.Mapper()
mapper.build(fcnet)
graph_info = mapper.compile(weight_bit_optimization=True, grouping_optim_target="both")
mapper.export(write_to_file=True, fp="./debug/", format="bin", split_by_coord=False, export_core_params=False)

graph_info.n_core_required
>>> 999

# Clear all the results
mapper.clear()
```

其中，编译时有如下参数可指定：

- `weight_bit_optimization`: 是否对权重精度进行优化处理。这将使得声明时为 INT8 的权重根据实际值当作更小的精度处理（当权重的值均在 [-8, 7] 之间，则可当作 INT4 进行处理）。默认由后端配置项内对应**编译选项**指定（默认开启）。
- `grouping_optim_target`：指定神经元分组的优化目标，可以为 `"latency"`，`"core"` 或 `"both"`，分别代表以延时/吞吐率、占用核资源为优化目标、或二者兼顾。默认由后端配置项内对应**编译选项**指定（默认为 `both`）。
- 同时，该方法将返回字典形式的编译后网络的信息。

导出时有如下参数可指定：

- `write_to_file`: 是否将配置帧导出为文件。默认为 `True`。
- `fp`：导出目录。若未指定，则默认为后端配置选项 `build_directory` 所设置的目录（当前工作目录）。
- `format`：导出交换文件格式，可以为 `bin`、`npy` 或 `txt`。默认为 `bin`。
- `split_by_coord`：是否将配置帧以每个核坐标进行分割，由此生成的配置帧文件命名形如"config_core1"、"config_core2"。默认为 `False`，即最终导出为一个文件。
- `export_core_params`: 是否导出实际使用核参数至json文件，以直观显示实际使用核的配置信息。默认为 `False`。

同时，该方法将返回模型的配置项字典 `GraphInfo`，包括：

- `input`：输入节点信息字典。
- `output`：输出目的地信息字典。
- `memebers`：中间层所在物理核的配置项字典。
- `inherent_timestep`：网络的最长时间步。
- `n_core_required`：网络**需要**的物理核数目。
- `n_core_occupied`：网络**实际占用**的物理核数目。
- `extras`：其他额外的网络信息字典，例如，编译后的网络名称。

### 后端配置项

与后端相关的配置项由 `BACKEND_CONFIG` 统一保存与访问，例如上述**编译选项**、`build_directory`、`target_chip_addr` 等。如下所示，对常用的配置项进行读取与修改：

1. 本地芯片地址 `target_chip_addr`，支持**多芯片配置**。

   ```python
   # Read
   BACKEND_CONFIG.target_chip_addr
   >>> [Coord(0, 0)]

   # Single chip
   BACKEND_CONFIG.target_chip_addr = (1, 1)
   # Multiple chips
   BACKEND_CONFIG.target_chip_addr = [(0, 0), (0, 1), (1, 0)]
   ```

2. 输出芯片地址（测试芯片地址） `output_chip_addr`

   ```python
   # Read
   BACKEND_CONFIG.output_chip_addr
   # or
   BACKEND_CONFIG.test_chip_addr
   >>> Coord(1, 0)

   # Modify
   BACKEND_CONFIG.output_chip_addr = (2, 0)
   # or
   BACKEND_CONFIG.test_chip_addr = (2, 0)
   ```

   ⚠️ 请确保输出芯片地址不与本地芯片地址重叠。

3. 编译后配置信息等文件输出目录路径 `output_dir`，默认为用户当前工作目录

   ```python
   # Read
   BACKEND_CONFIG.output_dir
   >>> Path.cwd() # Default is your current working directory

   # Modify
   BACKEND_CONFIG.output_dir = "path/to/myoutput"
   ```

4. 编译选项

   ```python
   # Set cflag for enabling weight precision optimization
   set_cflag(enable_wp_opt=True, cflag="This is a cflag.")

   # Read
   BACKEND_CONFIG.cflag
   >>> {"enable_wp_opt": True, "cflag": "This is a cflag."}
   ```
