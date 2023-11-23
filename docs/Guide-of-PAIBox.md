<div align="center">

# PAIBox使用指南

</div>

## 快速上手

目前PAIBox处于快速迭代的开发阶段，通过 `git clone` 指定 `dev` 分支以体验PAIBox。

```bash
git clone -b dev https://github.com/PAICookers/PAIBox.git
cd PAIBox
```

PAIBox使用Poetry管理依赖，如果使用Poetry：

```bash
poetry install
```

使用conda等，则手动安装如下依赖至你的Python虚拟环境：

```toml
python = "^3.9"
pydantic = "^2.0"
numpy = "^1.23.0"
```

## 基本组件

PAIBox提供**神经元**与**突触**作为基本组件，用于搭建神经网络。

结合**输入节点**，可以对输入数据进行脉冲编码，并传入网络中进行仿真推理。

### 神经元

PAIBox提供了多种类型的神经元模型，能够实现各种特殊的功能。

#### IF神经元

IF神经元实现了经典的“积分发射”模型，其调用方式及参数如下：

```python
import paibox as pb

n1 = pb.neuron.IF(shape=10, threshold=127, reset_v=0, vjt_init=0, keep_shape=False, name='n1')
```

其中：

- `shape`：代表神经元组的尺寸，其形式可以是整形标量、元组或列表。
- `threshold`：神经元阈值，其形式为整数。
- `reset_v`：神经元的重置电位。
- `vjt_init`：神经元的初始电位。
- `keep_shape`：是否在仿真记录数据时保持尺寸信息，默认为 `False`。实际进行运算的尺寸仍视为一维。
- `name`：可选，为该对象命名。

#### LIF神经元

LIF神经元实现了“泄露-积分-发射”神经元模型，其调用方式及参数如下：

```python
n1 = pb.neuron.LIF(shape=128, threshold=127, reset_v=0, leaky_v=-1, vjt_init=0, keep_shape=False, name='n1')
```

- `leaky_v`：LIF神经元的泄露值。需要注意的是，该值是直接加在神经元的输入上的，因此要实现一般的泄露功能应将其设为**负值**。其他参数含义与IF神经元相同。

#### Tonic Spiking神经元

Tonic Spiking 神经元可以实现对持续脉冲刺激的周期性反应。

```python
n1 = pb.neuron.TonicSpiking(shape=128, fire_step=3, vjt_init=0, keep_shape=False, name='n1')
```

- `fire_step`：发放周期，每接收到 `N` 次刺激后发放脉冲。

以下为一个简单实例：

```python
import paibox as pb
import numpy as np

n1 = pb.neuron.TonicSpiking(shape=1, fire_step=3)
inp_data = np.ones((10,), dtype=np.bool_)
output = np.full((10,), 0, dtype=np.bool_)
voltage = np.full((10,), 0, dtype=np.int32)

for t in range(10):
    output[t] = n1(inp_data[t])
    voltage[t] = n1.voltage

print(output)

>>> [[False]
    [False]
    [ True]
    [False]
    [False]
    [ True]
    [False]
    [False]
    [ True]
    [False]]
```

在持续的脉冲输入下，神经元进行周期性的脉冲发放。

#### Phasic Spiking神经元

Phasic Spiking神经元可以实现，在接受一定数量脉冲后发放，然后保持静息状态，不再发放。

```python
n1 = pb.neuron.PhasicSpiking(shape=128, time_to_fire=3, neg_floor=10, vjt_init=0, keep_shape=False, name='n1')
```

- `time_to_fire`：发放时间。
- `neg_floor`：地板阈值，静息时的膜电位为其负值。

以下为一个简单实例：

```python
import paibox as pb
import numpy as np

n1 = pb.neuron.PhasicSpiking(shape=1, time_to_fire=3)
# [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
inp_data = np.concatenate((np.zeros((2,), np.bool_), np.ones((10,), np.bool_)))
output = np.full((12,), 0, dtype=np.bool_)
voltage = np.full((12,), 0, dtype=np.int32)

for t in range(12):
    output[t] = n1(inp_data[t])
    voltage[t] = n1.voltage

print(output)

>>>
    [[False]
    [False]
    [False]
    [False]
    [ True]
    [False]
    [False]
    [False]
    [False]
    [False]
    [False]
    [False]]
```

当有持续性脉冲输入时，神经元会在 `time_to_step` 个时间步后发放脉冲，而后将一直保持静息状态。

<!-- **Spike Latency神经元**

`Spike Latency`神经元可以实现，在接受一个脉冲后，会在 `fire_time`个时间步后发放。

```python
# 实例化一个Spiking Latency神经元组
n1 = pb.neuron.SpikingLatency(shape=128,fire_time = 3,vjt_init=0, keep_shape=False, name='n1')
```

- `fire_time`：延时时间。

需要注意的是：首先，该神经元需要搭配权重使用，其输入权重应设置为10。在单独使用神经元时，可以在脉冲输入的对应时间步直接输入10，以达到相同的效果。其次，其在延时响应阶段不应有脉冲输入，否则会出现发放错误。

以下为一个简单实例：

```python
import paibox as pb
import numpy as np
n1 = pb.neuron.SpikeLatency(shape=1,fire_time=3)
input = np.array([0,10,0,0,0,10,0,0,0,0,0,0])
output = []
for i in range(12):
    output.append(n1(input[i]))
print(np.array(output))
# 结果如下，可以看到实现了延时功能。
[[False]
 [False]
 [False]
 [False]
 [ True]
 [False]
 [False]
 [False]
 [ True]
 [False]
 [False]
 [False]]
```

**Subthreshold Oscillations 神经元**

`Subthreshold Oscillations` 神经元，即振荡神经元，当有脉冲输入后，神经元会进行发放，之后，其膜电位会一直处于±1的振荡状态。

需要注意的是，该神经元需要搭配权重使用，其输入权重应设置为22。在单独使用神经元时，可以在脉冲输入的对应时间步直接输入22，以达到相同的效果。

以下为一个简单实例：

```python
import paibox as pb
import numpy as np
n1 = pb.neuron.SubthresholdOscillations(shape=1)
input = np.array([0,0,22,0,0,0,0,0,0,0,0,0])
voltage = []
for i in range(12):
    n1(input[i])
    voltage.append(n1.voltage)
print(np.array(voltage))
# 结果如下，通过记录膜电位值，可以看到振荡效果。
[[ 0]
 [ 0]
 [ 1]  # 神经元输出会在这一时刻发放。
 [-1]
 [ 1]
 [-1]
 [ 1]
 [-1]
 [ 1]
 [-1]
 [ 1]
 [-1]]
``` -->

### 突触

PAIBox中，突触用于连接不同神经元组，并包含了连接关系以及权重信息。以全连接类型的突触为实例：

```python
s1= pb.synapses.NoDecay(source=n1, dest=n2, weights=weight1, conn_type=pb.synapses.ConnType.All2All, name='s1')
```

其中：

- `source`：前向神经元组，可以是**神经元或者输入节点**类型。
- `dest`：后向神经元组，只能为**神经元**类型。
- `weights`：突触的权重。
- `conn_type`：连接形式，默认为 `MatConn` 矩阵连接。当设置为 `All2All` 或 `One2One` 时，`weights` 有更简洁的表达。
- `name`：可选，为该对象命名。

突触表达的是两个神经元组之间的连接关系。PAIBox提供了三种主要的连接关系表达：

- `All2All`：全连接
- `One2One`：单对单连接
- `MatConn`：普通的矩阵连接

通常情况下，`MatConn` 适合所有的连接关系，而 `All2All`、`One2One` 则提供了对于特殊连接更为方便的表达。

#### All2All 全连接

对于全连接，其权重 `weights` 有两种输入类型：

- 标量：默认为1。例如，设置为 `X`，则权重矩阵实际为元素均为 `X` 的 `N1*N2` 矩阵。
- 矩阵：尺寸为 `N1*N2` 的矩阵。

其中，`N1` 为前向神经元组数目，`N2` 为后向神经元组数目。

#### One2One 单对单连接

两组神经元之间依次单对单连接，这要求**前向与后向神经元数目相同**。其权重 `weights` 主要有以下几种输入类型：

- 标量：默认为1。这表示前层的各个神经元输出线性地输入到后层神经元。

  ```python
  n1 = pb.neuron.IF(shape=5,threshold=1)
  n2 = pb.neuron.IF(shape=5,threshold=1)
  s1 = pb.synapses.NoDecay(source=n1, dest=n2, conn_type=pb.ConnType.One2One, weights=2, name='s1')

  print(s1.weights)
  >>>
  2
  ```

  其权重以标量的形式储存。由于在运算时标量会随着矩阵进行广播，因此计算正确且节省了存储开销。
- 数组：尺寸要求为 `(N2,)`，可以自定义每组对应神经元之间的连接权重。如下例所示，设置 `weights` 为 `[1, 2, 3, 4, 5]`，

  ```python
  n1 = pb.neuron.IF(shape=5,threshold=1)
  n2 = pb.neuron.IF(shape=5,threshold=1)
  s1 = pb.synapses.NoDecay(source=n1, dest=n2, conn_type=pb.ConnType.One2One, weights=np.arange(1, 6, dtype=np.int8), name='s1')

  print(s1.weights)
  >>>
  [[1, 0, 0, 0, 0],
   [0, 2, 0, 0, 0],
   [0, 0, 3, 0, 0],
   [0, 0, 0, 4, 0],
   [0, 0, 0, 0, 5]]
  ```

  其权重实际上为 `N*N` 矩阵，其中 `N` 为前向/后向神经元组数目。

#### MatConn

普通的神经元连接类型，仅可以通过矩阵设置其权重 `weights`。

### 编码器

对于非脉冲数据，我们需要将其进行脉冲编码，然后输入网络中进行计算。

PAIBox提供了有状态与无状态编码器。其中，有状态编码器是指编码过程与时间有关，将输入数据编码到一段时间窗口内。而无状态编码器是指编码过程与时间无关。每个时间步，都可以根据输入直接进行编码。

#### 无状态编码器

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

#### 有状态编码器

有状态编码器类别较多。但目前来看，使用传统思路进行训练的SNN网络不能使用与时间有关的有状态编码器进行训练。

PAIBox提供了一种有状态编码器，周期性编码器 `PeriodicEncoder`。它以一段脉冲序列为输入，将其循环地在每一个时间步输出。以下为一个简单实例：

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

### 输入节点

为了支持多样的数据输入形式，同时标明网络的输入节点，PAIBox设计了输入节点这一组件。

输入节点可以使用以下方法定义：

```python
# 实例化一个输入节点
inp = pb.InputProj(input=1, shape_out=(4, 4), keep_shape=True, name='inp1')
```

其中，

- `input`：输入节点的数据，可以是整数，数组或可调用对象（函数或者实现了 `__call__` 方法的对象，例如，编码器）。
- `shape_out`：输出数据的尺寸。
- `keep_shape`：在观测节点输出数据时，可以通过该参数确定输出是否保持原始的维度信息，从而更好地进行监测。默认为 `True`。
- `name`：可选参数，为该节点命名。

#### 数据类型输入

当输入节点的输出为常量时，可以直接设置 `input` 为常量，并将 `shape_out` 设置为所需输出尺寸，即可实现。以下为一个简单实例：

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

输入节点的参数也可以是矩阵。以下为一个简单实例：

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

PAIBox支持使用自定义函数作为输入节点的输入。以下为一个简单实例：

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

这可以实现，每个时间步上均产生随机的输出。

当函数需要时间步信息，则可在函数参数中声明 `t` ，输入节点将在前端环境变量中获取当前时间步信息。当函数与时间步无关时，可使用  `*args` 作承接但不使用该参数。以下为一个简单实例：

```python
def fakeout_with_t(t, bias):
    return np.ones((4, 4)) * t + bias

inp = pb.InputProj(input=fakeout_with_t, shape_out=(4, 4), keep_shape=True)
prob = pb.simulator.Probe(inp, "feature_map")

sim = pb.Simulator(inp)
sim.add_probe(prob)
sim.run(4, bias=0)

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

PAIBox支持数据编码。以泊松编码器为例：

```python
pe = pb.simulator.PoissonEncoder()                          # 例化泊松编码器
inp = pb.InputProj(pe, shape_out=(104 4), keep_shape=True)  # 例化输入节点
input_data = np.random.rand(4, 4).astype(np.float32)        # 生成归一化数据

sim = pb.Simulator(inp)
prob = pb.simulator.Probe(inp, "feature_map")
sim.add_probe(prob)
sim.run(3, input=input_data)    # 传入数据至输入节点

output = sim.data[prob][-1]
print(output)

>>>
    [[ True False  True  True]
    [ True False  True  True]
    [False  True  True  True]
    [ True  True  True False]]
```

## 网络搭建DynSysGroup

### 基础模型搭建

在PAIBox中，神经网络搭建可以通过继承 `DynSysGroup` 或 `Network` 来实现。以一个简单的全连接网络为例：

<p align="center">
    <img src="images/Guide-基础网络搭建-全连接网络示例.png" alt="基础网络搭建-全连接网络示例" style="zoom:50%">
</p>

要搭建上述网络，首先继承 `pb.Network` 并在子类 `fcnet` 中初始化网络。先例化输入节点 `i1` 与两个神经元组 `n1`、 `n2`，然后例化两个突触 `s1`、 `s2` ，将三者连接起来：

```python
import paibox as pb

class fcnet(pb.Network):
    def __init__(self, weight1, weight2):
        super().__init__()
        pe = pb.simulator.PoissonEncoder()

        self.i1 = pb.InputProj(input=pe, shape_out=(784,))
        self.n1 = pb.neuron.IF(128, threshold=128, reset_v=0)
        self.n2 = pb.neuron.IF(10, threshold=128, reset_v=0)
        self.s1 = pb.synapses.NoDecay(self.i1, self.n1, weights=weight1, conn_type=pb.synapses.ConnType.All2All)
        self.s2 = pb.synapses.NoDecay(self.n1, self.n2, weights=weight2, conn_type=pb.synapses.ConnType.All2All)

```

<!--
#### **Sequential方法**

也可以使用 `Sequential` 构建 **线性网络**：

```python
# 使用Sequential搭建线性网络
n1 = pb.neuron.TonicSpiking(10, fire_step=3)
n2 = pb.neuron.TonicSpiking(10, fire_step=5)
s1 = pb.synapses.NoDecay(n1, n2, conn_type=pb.synapses.ConnType.All2All)
sequential = pb.network.Sequential(n1, s1, n2)
```

当网络是顺序运行时，不必定义 `update` 方法也可以正确执行。如果网络存在分支，或者需要特别设置网络节点之间的数据更新过程时，则可以通过定义 `update` 方法实现。以下是一个存在分支网络的例子，可以通过定义 `update` 方法设置其数据更新过程。

<img src="C:\Users\baibin\AppData\Roaming\Typora\typora-user-images\image-20230927173640154.png" alt="image-20230927173640154" style="zoom:67%;" />

```python
# 定义一个有分支的网络
class net_using_update(pb.DynSysGroup):
    def __init__(self):
        """
        n1 -> s1
                -> n3
        n2 -> s2
        """
        super().__init__()
        self.n1 = pb.neuron.IF(3,threshold=1,reset_v=0)
        self.n2 = pb.neuron.IF(3,threshold=1,reset_v=0)
        self.n3 = pb.neuron.IF(3 ,threshold=1,reset_v=0)
        self.s1 = pb.synapses.NoDecay(self.n1, self.n3, conn_type=pb.synapses.ConnType.All2All)
        self.s2 = pb.synapses.NoDecay(self.n2, self.n3, conn_type=pb.synapses.ConnType.All2All)
        self.p3 = pb.simulator.Probe(self.n3,'output')
    def update(self,x1,x2):
        y1 = self.n1.update(x1)
        y2 = self.n2.update(x2)
        y1_s1 = self.s1.update(y1)
        y2_s2 = self.s2.update(y2)
        y3 = self.n3.update(y1_s1 + y2_s2)
        return y3
``` -->

<!-- ### **高级算子**

PAIBox也支持在搭建网络中使用高级算子，如池化和卷积等，但首先与芯片结构，我们使用卷积和池化的方式有所变化。

卷积与池化都可以看作是更为稀疏的全连接层。因此，我们可以提供了可以将卷积核展开为全连接层以及生成池化层参数的函数。我们可以将卷积层和池化层转换为突触，并通过调用函数生成对应的权重矩阵送入突触中，从而实现卷积与池化的功能。

TODO -->

## 仿真

### 仿真器

例化网络后，即可构建仿真器：

```python
w1 = ...
w2 = ...
fcnet = fcnet(w1, w2)
sim = pb.Simulator(fcnet)
```

### 探针

在仿真过程中，用户需要检测某一层神经元的膜电位或输出、突触的输出等信息，这可通过**探针**实现。探针告知仿真器在仿真时需要记录哪些信息。使用探针的方式有如下两种：

1. 在构建网络时，直接设置探针，即在网络内部例化探针对象；
2. 在外部例化探针，并调用 `add_probe` 将其添加至仿真器内。仿真器内部将保存所有探针对象。
3. 调用 `Simulator.remove_probe` 方法可移除探针及其仿真数据。

例化探针时需指定：

- `target`: 监测对象，必须是 `DynamicSys` 类。
- `attr`：监测的属性，如 `spike`、`output` 等，字符串类型，这将监测 `target.attr` 属性。
- `subtarget`: 监测对象的子对象。可选，若指定，最终将监测 `target.subtarget.attr` 属性。

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
probe2 = pb.simulator.Probe(net.n1, "voltage")
sim.add_probe(probe2)
```

仿真数据可通过 `sim.data` 取得。可监测的对象包括网络内部所有的属性。例如，神经元及突触的各类属性，常用的监测对象包括：

- 输入节点的 `feature_map`。
- 神经元：脉冲输出 `spike`、脉冲输出（特征图形式） `feature_map`、膜电位 `voltage`。
- 突触：输出 `output`。

在设置完探针后，可输入数据并进行仿真，仿真结束后读取探针监测的数据：
 
```python
# 准备输入数据
input_data = np.random.rand(28, 28).astype(np.float32)
sim.run(10, input=input_data)   # 仿真10个时间步

# 读取仿真数据
n1_spike_data = sim.data[net.probe1]
n1_v_data = sim.data[net.probe2]
# 重置仿真器
sim.reset()
``````