# 量化技术实现与硬件和部署策略 (Quantize Tools)

## 1. 概述与设计背景

传统的 PyTorch 默认量化通常使用特定后端的量化引擎（如QNNPACK）。然而，对于PAICORE，我们需要进行**算子级的指令解构与硬件特征匹配**。
我们的方法实现了：
1. 完全摈弃片上运行时的任何浮点参与和复杂乘法修正。
2. 算子结构转换（如 4-Term 拆分的卷积运算操作）。
3. 使用查找表（LUT）极大地简化层间的反量化和再量化。

---

## 2. 量化基础公式


各个部分的一些数据格式：
*   **网络输入(input)**:量化为`int8`,zp!=0或者 `int8`,zp =0
*   **网络权重 (Weights, 权重张量 $W$)**：量化为`int8`,zp!=0或者 `int8`,zp =0
*   **激活值(ReLU输出)**：量化为`uint8`，zp=0 或者`int8`,zp=0

数学转换映射规则：

**量化：**
$$ X_q =  \text{round}\left( \frac{X_f}{S_x} \right) + Z_x $$

**反量化：**
$$ X_f \approx S_x \times (X_q - Z_x) $$

---

## 3. Linear 与 CNN 的量化计算方法

传统的量化网络中，计算模型面临非对称量化带来的交叉乘积问题。无论全连接层还是卷积层，基础计算都具有线性特征，为了避免在硬件MAC运算前进行复杂的零点减法，我们采用 **4项分解（4-Term Decomposition）**，将所有减法和常数转移出核心算子算符外。

### 3.1 核心分解思想 (Linear / Conv2d 泛化)
我们以操作函数 `Op()`（代表 `Linear(X, W)` 或 `Conv2d(X, W)` 等乘加过程）来描述。
对于非对称量化，原本真正的计算过程为：
$$ Y_{int} = Op(X_q - Z_x, W_q - Z_w) $$

根据算子的线性分配律，这可以被无损拆解为四个独立的操作项：
$$ Y_{acc} = \underbrace{Op(X_q, W_q)}_{\text{Term 1}} - \underbrace{Op(X_q, Z_w)}_{\text{Term 2}} - \underbrace{Op(Z_x, W_q)}_{\text{Term 3}} + \underbrace{Op(Z_x, Z_w)}_{\text{Term 4}} $$

通过这四项分解，硬件底层只需要进行纯无符号与存储固定矩阵的操作。接下来详细解析这四项是如何使用原生结构（特别是矩阵与卷积）实现的。

### 3.2 Linear (全连接层) 的等效操作(@表示矩阵乘法，E表示所有元素都为1的矩阵)
对于全连接操作，我们将 $Op$ 视作一般的内积/矩阵乘：
$$ Y_{int} =(W-z_w*E_w)@(X-z_x*E_x)$$
   即
   $$ Y_{int} =\underbrace{W@X}_{\text{Term 1}}-\underbrace{z_w*E_w@X}_{\text{Term 2}}-\underbrace{z_x*W@E_x}_{\text{Term 3}} +\underbrace{z_w*z_xE_w@E_x}_{\text{Term 4}}$$


*   **Term 1**: $W@X$, core正常配置即可，
*   **Term 2**: $z_w*E_w@X$,因为 $Z_w$ 是标量常量，这等价于对输入 $X_q$ 沿着输入特征维度做求和（即 `sum_X = sum(X_q)` ），再乘以常数 $Z_w$。core中可以算一次（占一个神经元），使用多播传给下一级的累加core
*   **Term 3**: $z_x*W@E_x$。由于 $W$ 在编译期固定，这等同于离线提取时预先对权重 $W_q$ 各个输出通道行进求和，得到一维静态常数向量，维度为output大小。这部分可以放到，Term 1部分的加法泄露中
*   **Term 4**: $z_w*z_xE_w@E_x$。等于`x_zero_point * w_zero_point * x的个数`纯粹的一维偏置项标量静态常数。这部分可以放到，Term 2部分的加法泄露中
  
  ps:因为$w-zp_w$可能超过int8的范围，所以term1,term2不可以合并。综合神经元开销，w+1个，core两个。

***当然，如果使用纯对称量化(已实现)，可以简化这个计算。目前看来，精度下降不大，可以使用一个core解决***


   


### 3.3 CNN (2D卷积层) 的等效卷积操作与维度分析

在卷积网络中，乘加关系更加复杂。我们将由纯求和公式的描述升维，直接将其映射到底层特定的**卷积功能流**。假设基础算符标示为 `Conv2d(Input, Weight)`：

$$ Y_{acc\_int32} = \text{Conv2d}(X - z_x \cdot E_x, W - z_w \cdot E_w) $$
即
$$ Y_{acc\_int32} = \underbrace{\text{Conv2d}(X, W)}_{\text{Term 1}} - \underbrace{\text{Conv2d}(X, z_w \cdot E_w)}_{\text{Term 2}} - \underbrace{\text{Conv2d}(z_x \cdot E_x, W)}_{\text{Term 3}} + \underbrace{\text{Conv2d}(z_x \cdot E_x, z_w \cdot E_w)}_{\text{Term 4}} $$

#### 输入维度定义
假设：
- $B$: Batch Size
- $C_{in}, C_{out}$: 输入/输出通道数
- $H, W$: 输入特征图的高和宽
- $H_{out}, W_{out}$: 输出特征图的高和宽
- $K_h, K_w$: 卷积核的高和宽
- $g$: 分组数 (groups)

主要输入张量的维度为：
- **`x_int` (由 $X_q$ 转换)**: 维度为 $[B, C_{in}, H, W]$
- **`weight_int` (由 $W_q$ 转换)**: 维度为 $[C_{out}, C_{in}, K_h, K_w]$



#### 4-Term 卷积公式与特征图维度

*   **Term 1 (核心特征卷积 MAC)**: `Conv2d(X_q, W_q)`
    直接将原始图像特征图送入 MAC 阵列，执行最原生的定点二维卷积运算。使用神经元$C_{out} \times H_{out}\times W_{out}$个
    - 公式：$Term1 = \text{Conv2d}(X_{int}, W_{int})$
    - **输出维度**: $[B, C_{out}, H_{out}, W_{out}]$

*   **Term 2 (动态激活图滑动和)**: `Conv2d(X_q, Z_w)`
    等价于令权重张量所有内部元素固定为 $Z_w$，对 $X_q$ 进行相同的窗口卷积操作。使用神经元$H_{out}\times W_{out}$个,使用多播传到下一级累加core
    - 公式：$Term2 = \text{Conv2d}(X_{int}, W_{zp\_tensor})$，$W_{zp\_tensor}$ 为全 $Z_w$ 的张量,形状同$W_q$。
    - **输出维度**:$[B, C_{out}, H_{out}, W_{out}]$

*   **Term 3 (静态权重的通道标量和)**: `Conv2d(Z_x, W_q)`
    这里 $Z_x$ 是全图平坦恒定的激活零点常数。
    不需要硬件运行任何乘法。部署时，将卷积核 $W_q$ 沿它的输入通道 $C_{in}$、 $K_h$、$K_w$ 全部求和压平，降维输出为一个仅仅对应于 $C_{out}$ 的静态 1D 常数阵列，接着乘以 $Z_x$。运行时直接作为 Bias 加加器的一部分装载入寄存器。
    - 公式：$Term3 = Z_x \times \sum(W_{int})$
    - **输出维度** : 经过维度调整变为 $[1, C_{out}, 1, 1]$，广播后，使用加法泄露，与term1结合。

*   **Term 4 (零点结构偏置常数)**: `Conv2d(Z_x, Z_w)`
    完全独立预先计算的标量结果，与上文 Term 3 本质相同，在最后被使用加法泄露，与term2结合。
    - 公式：$Term4 = Z_x \times Z_w \times (K_h \times K_w \times \frac{C_{in}}{g})$
    - **输出维度**: 标量 (Scalar)，计算时广播至 `[B, C_{out}, H_{out}, W_{out}]`

#### Bias (偏置项) 加载
公式：$Bias_q = \text{round}\left(\frac{Bias}{Scale_x \times Scale_w}\right)$
- 偏置由于在浮点域是直接与卷积结果相加的，因此其对应的 Scale 是输入与权重 Scale 的乘积。
- 量化 Bias 时，将其形状调整成了 `[1, C_{out}, 1, 1]`。
- **输出维度**: 通过广播机制，最终安全地注入累加器，`Output_Acc` 的维度依然保持 `[B, C_{out}, H_{out}, W_{out}]`，与term3相同，进行合并。

***当然，如果使用纯对称量化(已实现)，可以简化这个计算。目前看来，精度下降不大，可以使用一个core解决***

---
## 4. LUT与激活函数和反量化再量化的深度耦合

在获取到上文分解完毕的 32-bit $Y_{acc\_int32}$ 累加值之后，如果要向下馈送给后续的网络层，我们面临了core最痛苦的阻碍：**重缩放运算  与 非线性激活(ReLU)**。

理论上的严谨后处理公式为：
$$ Y_{q\_out} = \text{clamp}\left( \text{round}\left( Y_{acc\_int32} \times \frac{S_{in} \cdot S_{w}}{S_{out}} \right) + Z_{out}, \quad 0, \quad 255 \right) $$

**痛点分析：** 
系数 $M = \frac{S_{in} \cdot S_{w}}{S_{out}}$ 是一个复杂的浮点数值）。我们芯片纯整型加速器内部无法直接计算。若采用传统的方法将大量的数据搬运到cpu进行反量化、激活函数、再量化三步，将极度的占用大量的时间。为此，我结合LUT的天然量化特性，将这三步融入到精确配置LUT实现。

### 4.1 LUTReLU实现原理
在计算得到整数累加器 `Output_Acc` 后，我们需要将其转换为量化的输出 $q_{out}$。

理想的实数输出 $y$ 应该满足：
$$ y_{float} = S_{out}\times Y_{acc\_int32} $$

同时我们也知道：
$$ S_{out} = S_{in} S_{w}  $$

因此经过ReLU后：
$$ y_{act} = \text{ReLU}( S_{in} S_{w} \cdot Y_{acc\_int32} ) $$
被下一级的校准系数进行再量化：



$$q_{out} = \frac{y_{act}}{S_{out}}+Z_{out}$$ 
一般的$Z_{out}=0$:

$$ q_{out} = \frac{S_{in} S_{w}}{S_{out}} \text{ReLU}(Y_{acc\_int32})$$

由于经过校准，我们可以保证 $max(Y_{acc\_int32})$经过量化后对应255 (配置输出是uint8) 。因此我们只需要配置，LUT的max=lut_scale*255 对应映射为255即可保证$Y_{acc\_int32}$能精确的使用校准参数进行再量化。


### 4.2. LUT 映射表的数学装填态 (由本工具链固化输出，读者可以先跳过)
在我们的工具生成器中（如对应 `LutActivation` 构建等环节），我们对 `LUT 数组` 执行如下填充构建：

$$ LUT[i] = \text{clamp}\left( \text{round}\left( \frac{\text{Activation}( i \times 2^{Shift\_Amount} \times S_{in} \times S_{w} )}{S_{out}} \right) + Z_{out}, \quad 0,\ 255 \right) $$

**针对典型的 `ReLU` 神经元，这个表会表现出极为美妙的应用特性：**
*   当计算得出的 $i < (\text{理论零点分岔位置})$ 时，因为 ReLU 剔除了负浮点数输入，这里产生的表项将直接全部被预填为 $Z_{out}$ 对应的基准 0 值！
*   当偏上计算引发饱和溢出时，工具链的公式自然将其 `clamp` 锁定于常数 `255`。
*   在中间斜坡过度带，$M$ 所引申出的复杂的带偏置、带浮点乘缩放的繁杂换算被凝固在表中一个精准的 8-bit `uint8` 整型数据位里。

**硬件LUT部分：** 
由于整个公式已离线封装好，“繁重”的计算缩减为了周期极短的单指令，仅仅花费 $O(1)$ 指令周期的内存访问，对应地址（索引为 $i$）里安放的8bit的激活值特征就瞬间取出，完美送入了下一层core。

---



## 6. FX Graph 软件架构与执行管道 (Graph Converters & Exporters)

为了让量化参数与上述底层精妙机制完美衔接，我们的工具软件分成了两个紧密扣合的阶段：

### 6.1. 智能提取转换管道 (`FxGraphConverter.py`)
使用 `torch.fx` 从抽象语法树级别捕获：
*   **观测解析 (Observer Parser)**: 当 Pytorch 跑完 Calibration 并打上 MinMaxObserver 等观测节点标签后，截取提炼这些节点的真实 $(S, Z)$ 流。
*   **算子超融合 (Operator Fusion)**: 对如 `[Conv2d -> ReLU, Linear -> ReLU]` 这种高频出现的连接进行图优化融合，并使用 `QuantizedConvReLU2d` 和 `LutLinear` 等特别的包装进行替换。
*   **精确进行算子替换** 使用手写的算子，精确模拟硬件行为，包括4term计算和LUT激活，为后续验证模型量化性能建立仿真基础。

## 7. Res(残差)结构的量化实现

在 ResNet 等带有捷径(Shortcut)的网络结构中，最核心的操作是特征图的逐元素相加与激活，即： $Y = \text{ReLU}(X + \text{Conv2d}(Y_{branch}))$。其中，$X$ 是主干传来的激活值特征图，$Y_{branch}$ 是需要进入分支卷积的特征图。

这个操作在物理硬件与量化部署中面临**三个极大的挑战**：
1. **量化参数不一致带来的对齐难题**：$X$ 直接来自于上一层的输出，具有自己的 $Scale_x$ 与 $ZP_x$。而卷积分支出口的数据，在其量化累加器(Accumulator)里的隐含语义尺度是 $S_{accum} = S_{y\_in} \times S_{w}$。两者的精度刻度完全不同（即 $S_x \neq S_{accum}$），不可能直接相加。
2. **加法不自然** ：本质上的x是激活值，而conv2d(y)的输出是膜电平，二者的累加在目前离线核中根本上是不自然的
3. **多重非线性与重缩放开销**：如果像传统框架那样，把卷积结果先反量化回浮点数，把 $X$ 也反量化回浮点数，再相加、过 ReLU、最后寻找最大最小值重新量化，这会引发极其密集的浮点运算与数据在cpu和离线核搬运的迟滞。我们必须在纯整数域内并结合 LUT 完成所有操作。

为此，我们在 `res_test` 模块中提出了一种基于 **累加器尺度对齐** 和 **近似乘法位移** 的纯INT残差相加方案算法体系。

### 7.1 近似除法：向卷积残差累加器基准对齐

```text
      ┌─────────────────────────────────────────────────────────────┐
      │                                                             │
      │   ┌ - - - - - - - - - - - - - - - - - - - - - - - - - - ┐   ▼
      │   │ ┌──────┐   ┌─────┐   ┌──────┐   ┌──────┐   ┌─────┐  │ ┌───┐   ┌──────┐   ┌──────┐
X^l ──┴───┼►│ Conv ├─► │ BN  ├─► │ ReLU ├─► │ Conv ├─► │ BN  ├──┼►│ + ├─► │ ReLU ├─► │  Y^l │
          │ └──────┘   └─────┘   └──────┘   └──────┘   └─────┘  │ └───┘   └──────┘   └──────┘
          └ - - - - - - - - - - - - F(X^l) - - - - - - - - - - -┘
                                                                             
```

上面展示了典型的 ResNet 基础残差块（Basic block in ResNet）结构，其中主干分支包含了一系列卷积、BN和ReLU操作（即图中的结构 $\mathcal{F}^l(X^l)$ ），其结果与来自源头 $X^l$ 的直连捷径（Shortcut）进行逐元素相加，最后经过一次 ReLU 后输出。

---

为了在量化体系中实现这个过程，与其将主干的卷积累加结果与 $X$ 都还原回浮点或者寻找新的基准，不如**以卷积层最终抛出的高精度 `int32` 累加器（`out_conv_acc`）作为统一的尺度基准**。
卷积累加器的等效尺度为：
$$ S_{accum} = S_{y\_in} \times S_w $$

此时我们要把 $X$ 转换为与 `out_conv_acc` 处于同一衡量体系的值 $X_{aligned}$。从浮点真值相等的角度推导：
$$ X_{aligned} =\frac{(X_{int} - Z_x) \times S_x}{S_{accum}}    $$
因此理想的缩放率为：
$$ Ratio = \frac{S_x}{S_{accum}} $$

为了不在硬件中真正执行浮点除法，我们深度分析离线核的计算流程，决定使用权重相乘配套乘法位移的方案进行近似。我们寻找一对常量整数 $M$ 和 $n$，满足：
$$ Ratio \approx M \times 2^n $$
其中 $M$ 被限制为一个 8 位无符号整数（$1 \le M \le 255$），$n$ 是一个有符号的位移整数（如限制在 -32 到 31 间）。在硬件执行时，即等于将数据先乘 $M$ 再执行乘法泄露。

而对于 $Ratio \times Z_x$这部分，本质上是一个bias，因为硬件的加法泄露发生在乘法之前，所以我们可以直接配置神经元加法泄露为$M \times Z_x$即可。

### 7.2 纯定点加法与 LUT 融合重缩放

完成对齐后，残差分支的两路数据全部进入了纯整数且相同精度，可以直接用纯膜电平累加执行：
$$ q_{add\_int32} = out\_conv\_acc + X_{aligned} $$
这时的 $q_{add\_int32}$ 仍是一个庞大的 32 位整型张量，蕴藏着 $S_{accum}$ 的尺度。

为了量化输出给接下来的网络（例如目标输出的量化参数为 $S_{out}$ 与 $Z_{out}$），我们**继续复用前文提到的 `LutReLU` 特性**。
我们知道最终浮点真值为：
$$ y_{float} = q_{add\_int32} \times S_{accum} $$

进入 ReLU 并映射回目标特性的过程应当为：
$$ q_{out} = \text{ReLU}\left( q_{add\_int32} \times \frac{S_{accum}}{S_{out}} \right) $$



总结下来，我们在残差网络上的量化实现**彻底根绝了硬件运行时的浮点运算**，仅仅依赖**一次整型乘加与移位**进行基准对齐，随后进行**纯整数加法**，最后利用 LUT进行量化操作，将数值锁回下一层期望的正规 8-bit 整型分布域中。

整个过程，我们延续了ir中所期望的 relu(core1膜电平+core2膜电平)的实现路径。

TODOing :目前该功能正在测试当中。