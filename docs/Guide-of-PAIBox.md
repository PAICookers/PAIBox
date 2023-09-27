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

PAIBox提供**神经元**与**突触**作为基本组件，用于搭建神经网络。结合**输入节点**，可以将数据传入并进行仿真运算。

### 神经元

PAIBox中神经元的调用方式（以**IF**神经元为例）：

```python
import paibox as pb

# 实例化一个IF神经元组
n1 = pb.neuron.IF(shape=128, threshold=127, reset_v=0, vjt_init=0, keep_size=False, name='n1')
```

其中：

- `shape`：代表神经元组的大小，其形式可以是整数、元组或列表，可产生一组功能完全相同的神经元。
- `threshold`：神经元阈值，其形式为整数。
- `reset_v`：神经元的重置电位。
- `vjt_init`：神经元的初始电位。
- `keep_size`：布尔型变量，为 `True` 时，神经元组输出会严格保持尺寸。默认为 `False`，便于运算。
- `name`：给神经元命名。

除IF神经元外，PAIBox还提供了**LIF**、**TonicSpiking**、**PhasicSpiking**等，可支持多样的神经元计算模型。

### 突触

PAIBox中突触的调用方式（以**全连接**类型突触为例）：

```python
import paibox as pb

# 实例化一个全连接类型突触
s1 = pb.synapses.NoDecay(source=n1, dest=n2, conn=pb.synapses.All2All(), weights=weight1, name='s1')
```

其中：

- `NoDecay`：指在传递信息过程中没有衰减的突触类型。鉴于芯片并不支持其他类型的突触，因此仅有这一种类型。
- `source`：表示突触所连接的前向神经元组，可以是神经元或者输入节点类型。
- `dest`：表示突触所连接的后向神经元组，只可以是神经元类型。
- `conn`：表示突触连接两个神经元组的形式，`All2All` 表示全连接。除此之外，PAIBox还提供了 `One2One` 单连接等其他连接形式，满足不同的神经元连接需求。
- `weights`：可以通过设置此参数，将自定义权重配置到该突触中，形成全连接层。

### 输入节点

为了支持多样的数据输入形式，PAIBox设计了输入节点这一组件。输入节点通过如下方式构建：

```python
import paibox as pb

# 实例化一个输入节点
I1 = pb.projection.InputProj(val_or_func=Encoder_process, shape=784)
```

输入节点使用时与神经元类似，需要例化突触将其与其他神经元连接起来，构成网络。主要参数有两个：

- `val_or_func`：该参数指定了传入输入节点的数据，它可以是整数，numpy数组，或者是 `Process` 的子类。
- `shape`：输出尺寸。

### Process

如输入节点的示例所示，当传给输入节点的数据是已知的脉冲数据时，`val_or_func` 是对应的整数或numpy数组，**不会随timestep而变化**。如果想要输入数据是一些随时间变化的量，或者需要通过函数生成输入数据，则可以通过继承 `Process` 来实现。

下面以泊松编码为例，完成一个自定义 `Process` 的构建。

```python
# Process基本形式
class NewProcess(pb.base.Process):
    def __init__(self, shape_out):
        super().__init__(shape_out)

    def update(self):
        ...

#可对输入进行泊松编码的process
class PoissonEncoder(pb.base.Process):
    def __init__(self, shape_out, input):
        super().__init__(shape_out)
        self.input = input

    def update(self, t):
        out_spike = torch.rand_like(self.input).le(self.input)
        out_spike = out_spike.numpy().flatten()
        return out_spike
```

可以通过定义 `update` 方法来实现想要的函数式输出。

- 如果想要其根据时间变化而一直产生，可以在输入参数中**显式地接收timestep参数**（必须在第一参数位置）。由此，`Process` 会产生一个与timestep相关的输出。
- 若输出与timestep无关，则无需接收该参数。由此，输出与timestep无关。

上述 `Process` 示例可以实现对输入的NMIST图片进行泊松编码，并将其flatten后输出。

## 网络搭建

在PAIBox中，神经网络搭建可以通过继承 `DynSysGroup` 来实现。以一个简单的全连接网络为例：

```python
# 定义一个两层全连接网络
class fcnet(pb.DynSysGroup):
    def __init__(self, Encoder):
        super().__init__()
        self.n1 = pb.projection.InputProj(Encoder)
        self.n2 = pb.neuron.IF(128, threshold=127, reset_v=0)
        self.n3 = pb.neuron.IF(10, threshold=127, reset_v=0)
        self.l1 = pb.synapses.NoDecay(self.n1, self.n2, pb.synapses.All2All(), weights=weight1)
        self.l2 = pb.synapses.NoDecay(self.n2, self.n3, pb.synapses.All2All(), weights=weight2)
```

首先根据网络需求实例化输入节点和神经元节点，然后实例化突触将其连接起来，即可构成网络。也可以使用 `Sequential` 构建 **线性网络**：

```python
# 使用Sequential搭建线性网络
n1 = pb.neuron.TonicSpiking(10, fire_step=3)
n2 = pb.neuron.TonicSpiking(10, fire_step=5)
s1 = pb.synapses.NoDecay(n1, n2, pb.synapses.All2All())
sequential = pb.network.Sequential(n1, s1, n2)
```

当网络是顺序运行时，不必定义 `update` 方法也可以正确执行。如果网络存在分支，或者需要特别设置网络节点之间的数据更新过程时，则可以通过定义 `update` 方法实现。以下是一个存在分支网络的例子，可以通过定义 `update` 方法设置其数据更新过程。

```python
# 定义一个有分支的网络
class Net_User_Update(pb.DynSysGroup):
    def __init__(self):
        """
        n1 -> s1
                -> n3
        n2 -> s2
        """
        super().__init__()
        self.n1 = pb.neuron.TonicSpiking(3, fire_step=2)
        self.n2 = pb.neuron.TonicSpiking(3, fire_step=2)
        self.n3 = pb.neuron.TonicSpiking(3, fire_step=2)
        self.s1 = pb.synapses.NoDecay(self.n1, self.n3, pb.synapses.One2One())
        self.s2 = pb.synapses.NoDecay(self.n2, self.n3, pb.synapses.One2One())

    def update(self, x1, x2):
        y1 = self.n1.update(x1)
        y2 = self.n2.update(x2)
        y1_s1 = self.s1.update(y1)
        y2_s2 = self.s2.update(y2)
        y3 = self.n3.update(y1_s1 + y2_s2)

        return y3
```

## 仿真

根据实例化好的网络，构建仿真器，即可进行仿真。代码示例如下：

```python
# 使用仿真器进行仿真
fc_net1 = fcnet(PoissonEncoder(784, imgs))
sim = pb.Simulator(fc_net1)
sim.run(5)
```

上述代码中，以对图像进行泊松编码后的结果作为输入，实例化了之前定义的 `fcnet` 网络，然后构建了仿真器，并运行5个timestep。

### 状态监测

在仿真过程中，我们可能需要检测某一层神经元或突触的膜电位或输出，这时我们可以通过设置探针的形式，记录仿真过程中的数据变化。

```python
# 监测仿真过程中的状态变化
probe1 = pb.simulator.Probe(fc_net1.n1, "output")
sim.add_probe(probe1)
sim.run(10)
print(sim.data[probe1]) # sim.data字典保存了所有仿真数据
```

在上述代码中，首先设置了探针 `probe1`，它指向 `fc_net1` 网络的 `n1` 节点，并记录它的输出（脉冲）。监测 `voltage`，可以记录神经元膜电位信息。

在定义网络时，亦可例化一个内部的探针对象，则无需从外部调用 `add_probe`，将探针添加至仿真器对象内。

```python
# 定义一个两层全连接网络并进行监测
class fcnet(pb.DynSysGroup):
    def __init__(self, Encoder):
        super().__init__()
        self.n1 = pb.projection.InputProj(Encoder)
        self.n2 = pb.neuron.IF(128, threshold=127, reset_v=0)
        self.n3 = pb.neuron.IF(10, threshold=127, reset_v=0)
        self.l1 = pb.synapses.NoDecay(self.n1, self.n2, pb.synapses.All2All(), weights=weight1)
        self.l2 = pb.synapses.NoDecay(self.n2, self.n3, pb.synapses.All2All(), weights=weight2)
        # 监测n2的膜电位
        self.n1_acti = pb.simulator.Probe(self.n2, "voltage")
```
