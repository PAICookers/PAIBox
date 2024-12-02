# 算子支持

## 神经元

### 配置项

芯片所支持的神经元配置项如下表所列：

|         支持功能          | 可写 |         取值         | 功能描述                                                                                                                                                                        |
| :-----------------------: | :--: | :------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|         复位模式          |  ✅  | 硬复位/软复位/不复位 | 硬复位，膜电平重置为正/负阈值<br />软复位，膜电平将减正阈值/加负阈值（若负阈值模式为复位模式）<br />不复位，膜电平保持不变                                                      |
|         复位电平          |  ✅  |    30比特有符号数    | 可配置复位电平                                                                                                                                                                  |
|       比较前后泄露        |  ✅  |        前/后         | 阈值比较发生在泄露前/后                                                                                                                                                         |
|          正阈值           |  ✅  |    29比特无符号数    | 可配置正阈值                                                                                                                                                                    |
|          负阈值           |  ✅  |    29比特无符号数    | 可配置负阈值                                                                                                                                                                    |
|         泄露电平          |  ✅  |    30比特有符号数    | 可配置泄露幅值                                                                                                                                                                  |
|         反向泄露          |  ✅  |      开启/关闭       | 若开启，泄露与当前膜电平符号相关：<br />当泄露值为正，膜电平向0收敛<br />当泄露值为负，膜电平偏离0发散                                                                          |
|        负阈值模式         |  ✅  |      复位/饱和       | 当膜电平低于负阈值时：<br />为复位模式，根据复位模式复位<br />为饱和模式，膜电平重置为负阈值                                                                                    |
| 膜电平截取位（仅ANN模式） |  ✅  |        [0,29]        | 输出膜电平的截取位置T，30比特有符号膜电平需截取8比特作为输出：<br />T<8，截取[T-1:0]，低位补0 <br />T=8，截取[7:0]<br />T≤29，截取[T-1:T-8]<br />膜电平大于窗口最高位则截断处理 |
|       随机轴突整合        |  ✅  |      开启/关闭       | 若开启，神经元根据硬件生成的随机数\*过滤一些轴突上的输入，进行选择性累加                                                                                                        |
|         随机泄露          |  ✅  |      开启/关闭       | 若开启，如果泄露幅值小于硬件生成的随机数\*，则此次泄露为0                                                                                                                       |
|         阈值掩码          |  ✅  |        [0,29]        | 若开启，硬件生成的随机数\*将和它求与后得到一个0\~29比特随机阈值，并加至神经元的正、负阈值上                                                                                     |
|          膜电平           |  ❌  |          0           | 只读寄存器，初始值为0                                                                                                                                                           |

\*硬件生成的随机数均为无符号数。

## 突触

芯片不支持Alpha、AMBA、GABA等类型突触。

## 算子

包括突触与突触+神经元组合形式的算子。

|         算子类型         | ANN | SNN |     备注     |
| :----------------------: | :-: | :-: | :----------: |
|          全连接          | ✅  | ✅  |              |
|        2D矩阵乘法        | ✅  | ✅  |              |
|          1D卷积          | ✅  | ✅  |  全展开形式  |
|          2D卷积          | ✅  | ✅  |  全展开形式  |
|        1D转置卷积        | ✅  | ✅  |  全展开形式  |
|        2D转置卷积        | ✅  | ✅  |  全展开形式  |
|           位与           | ❌  | ✅  |              |
|           位或           | ❌  | ✅  |              |
|           位非           | ❌  | ✅  |              |
|          位异或          | ❌  | ✅  |              |
|        1D平均池化        | ❌  | ✅  |    脉冲化    |
| 1D平均池化（膜电位相关） | ❌  | ✅  |    脉冲化    |
|        1D最大池化        | ❌  | ✅  |    脉冲化    |
|        2D平均池化        | ❌  | ✅  |    脉冲化    |
| 2D平均池化（膜电位相关） | ❌  | ✅  |    脉冲化    |
|        2D最大池化        | ❌  | ✅  |    脉冲化    |
|          脉冲加          | ❌  | ✅  | 针对脉冲序列 |
|          脉冲减          | ❌  | ✅  | 针对脉冲序列 |
|          线性层          | ✅  | ❌  |              |
|          2D卷积          | ✅  | ❌  |  半折叠形式  |
|        2D最大池化        | ✅  | ❌  |  半折叠形式  |
|        2D平均池化        | ✅  | ❌  |  半折叠形式  |
|          线性层          | ✅  | ❌  |  半折叠形式  |