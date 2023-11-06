# PAIBox使用指南

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

PAIBox提供**神经元**与**突触**作为基本组件，用于搭建神经网络。结合**输入节点**，可以将数据传入并进行仿真。

### 神经元

PAIBox中神经元的调用方式（以 `IF` 神经元为例）：

```python
import paibox as pb

# 实例化一个IF神经元组
n1 = pb.neuron.IF(shape=128, threshold=127, reset_v=0, vjt_init=0, keep_shape=False, name='n1')
```

其中：

- `shape`：代表神经元组的大小，其形式可以是整数、元组或列表，可产生一组功能完全相同的神经元。
- `threshold`：神经元阈值，其形式为整数。
- `reset_v`：神经元的重置电位。
- `vjt_init`：神经元的初始电位。
- `keep_shape`：是否在仿真记录数据时保持尺寸信息，默认为 `False`。实际进行运算的尺寸仍为一维。
- `name`：可选，为该对象命名。

除IF外，PAIBox还提供了 `LIF`、`TonicSpiking`、`PhasicSpiking` 等神经元，可支持多样的神经元计算模型。

### 突触

PAIBox中突触的调用方式，以全连接类型突触为例：

```python
# 实例化一个全连接类型突触
s1= pb.synapses.NoDecay(source=n1, dest=n2, weights=weight1, conn_type=pb.ConnType.All2All, name='s1')
# or conn_type=pb.synapses.ConnType.All2All
```

其中：

- `source`：表示突触所连接的前向神经元组，可以是**神经元或者输入节点**类型。
- `dest`：表示突触所连接的后向神经元组，只能为**神经元**类型。
- `weights`：可以通过设置此参数，将自定义权重配置到该突触中。
- `conn_type`：表示突触连接两个神经元组的形式，`All2All` 表示全连接。除此之外，PAIBox还提供了 `One2One` 单对单连接以及 `MaskedLinear` 矩阵连接，满足不同的神经元连接需求。
- `name`：可选，为该对象命名。

### 编码器

大多数情况下，输入数据将经过编码后再输入至网络。PAIBox提供了两类编码器，**有状态**与**无状态**编码器。泊松编码是一种常用的无状态编码，其调用方式如下：

```python
pe = pb.simulator.encoder.PoissonEncoder(shape_out=(10, 10))
x = np.random.randint(-128, 128, (10, 10), dtype=np.int8)

# Simulate 20 time steps
out_spike = np.full((20, 10, 10), 0)

for t in range(20):
    out_spike[t] = pe(x)

# Or use the `run` method
out = pe.run(duration=20, x=x)
```

`Encoder` 内部提供了简单的 `run` 方法用以单独对编码器进行仿真，但注意需要为编码器**显式地传入参数**。

### 输入节点

为了支持多样的数据输入形式，PAIBox设计了输入节点这一组件。输入节点通过如下方式构建：

TODO

<!-- ```python
# 实例化一个输入节点
I1 = pb.projection.InputProj(input=Encoder, shape_out=(784,))
```

可以通过定义 `__call__` 方法来实现想要的函数式输出。

- 如果想要其根据时间变化而一直产生，可以在输入参数中**显式地接收timestep参数**（必须在第一参数位置）。由此，`Encoder` 会产生一个与timestep相关的输出。
- 若输出与timestep无关，则无需接收该参数。由此，输出与timestep无关。

上述 `Encoder` 示例可以实现对输入的NMIST图片进行泊松编码，并将其flatten后输出。

输入节点使用时与神经元类似，需要例化突触将其与其他神经元连接起来，构成网络。主要参数有两个：

- `input`：该参数指定了传入输入节点的数据，它可以是标量整型，numpy数组，返回值为numpy数组的可调用对象（例如函数），或为 `Encoder` 编码器类型。
- `shape_out`：输出尺寸。 -->

## 网络搭建

TODO

<!-- 在PAIBox中，神经网络搭建可以通过继承 `DynSysGroup`（或 `Network` ）来实现。以一个简单的全连接网络为例：

```python
# 定义一个两层全连接网络
class fcnet(pb.DynSysGroup):
    def __init__(self, encoder):
        super().__init__()
        self.n1 = pb.projection.InputProj(encoder)
        self.n2 = pb.neuron.IF(128, threshold=127, reset_v=0)
        self.n3 = pb.neuron.IF(10, threshold=127, reset_v=0)
        self.s1 = pb.synapses.NoDecay(self.n1, self.n2, weights=weight1, conn_type=pb.ConnType.All2All)
        self.s2 = pb.synapses.NoDecay(self.n2, self.n3, weights=weight2, conn_type=pb.ConnType.All2All)
```

根据网络需求实例化输入节点和神经元节点，然后实例化突触将其连接起来。`Sequential` 用以构建**线性网络**：

```python
n1 = pb.neuron.TonicSpiking(10, fire_step=3)
n2 = pb.neuron.TonicSpiking(10, fire_step=5)
s1 = pb.synapses.NoDecay(n1, n2, 1, conn_type=pb.ConnType.All2All)
sequential = pb.network.Sequential(n1, s1, n2)
``` -->

## 仿真器

在构建并实例化网络后，可构建仿真器，对其进行仿真：

```python
net = fcnet(PEncoder(imgs))
sim = pb.Simulator(net)
```

上述代码中，以对图像进行泊松编码后的结果作为输入，实例化了之前定义的 `fcnet` 网络，然后构建了仿真器，并运行5个timestep。

### 探针

在仿真过程中，用户需要检测某一层神经元的膜电位或输出、突触的输出等信息，这可以通过设置**探针**的方式，告知仿真器在仿真时需要记录哪些信息。有两种使用探针的方式：

1. 在构建网络时，直接设置探针，即在网络内部例化探针对象；
2. 在外部例化探针，并调用 `add_probe` 将其添加至仿真器内。仿真器内部将保存所有探针对象。

```python
class fcnet(pb.DynSysGroup):
    def __init__(self, encoder):
        super().__init__()
        self.n1 = pb.projection.InputProj(encoder)
        self.n2 = pb.neuron.IF(128, threshold=127, reset_v=0)
        self.n3 = pb.neuron.IF(10, threshold=127, reset_v=0)
        self.s1 = pb.synapses.NoDecay(self.n1, self.n2, weights=weight1, conn_type=pb.ConnType.All2All)
        self.s2 = pb.synapses.NoDecay(self.n2, self.n3, weights=weight2, conn_type=pb.ConnType.All2All)

        # 内部探针，记录神经元n1的输出脉冲
        self.probe1 = pb.simulator.Probe(self.n1, "spike")

net = fcnet(PEncoder(imgs))
sim = pb.Simulator(net)

# 外部探针，记录神经元n1的膜电位
probe2 = pb.simulator.Probe(net.n1, "voltage")
sim.add_probe(probe1)
```

可监测的对象包括神经元及突触的各类属性，包括但不限于：TODO

### 数据记录

PAIBox提供了多种不同访问数据形式。TODO

<!-- ```python
# 监测仿真过程中的状态变化
probe1 = pb.simulator.Probe(fc_net1.n1, "output")
sim.add_probe(probe1)
sim.run(10)
```

在上述代码中，首先设置了探针 `probe1`，它指向 `fc_net1` 网络的 `n1` 节点，并记录它的输出（脉冲）。监测 `voltage`，可以记录神经元膜电位信息。 -->
