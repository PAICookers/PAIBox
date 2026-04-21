# `graph = compile_to_paiir(model, SAMPLE_INPUT)` 分析说明

这份笔记专门解释 [tests/onboard/modelcnn/test_modelcnn_paiir_deploy.py](../tests/onboard/modelcnn/test_modelcnn_paiir_deploy.py) 里这一行代码：

```python
graph = compile_to_paiir(model, SAMPLE_INPUT)
```

结论先说：`compile_to_paiir(...)` 是 PAIIR 的一站式编译入口。它接收一个 PyTorch `nn.Module` 和若干示例输入，把它们逐步转换成一个可交给后端的 `PAIIRGraph`。它不是简单包装 `torch_to_paiir`，而是完整跑完了 tracing、lowering、融合、重写、推导、校验和调度。

## 这行代码的输入

在这个测试里，输入并不是原始训练模型，而是已经做过部署侧替换的 LeNet5：

- `model`：由 `_build_lenet5_model_from_exports()` 构造出来的 `LeNet5` 实例
- `SAMPLE_INPUT`：`torch.zeros((1, 1, 28, 28), dtype=torch.float32)`

`compile_to_paiir` 还能接收很多可选参数，例如 `tick_duration`、`auto_reset`、`tick_overrides`、`input_formats`、`compile_config`、`concrete_args`、`strict`、`enable_avgpool_calibration` 和 `enable_split_avgpool_lif`。默认情况下，这些参数会走内部默认值；如果传了 `compile_config`，则会先用配置对象，再用显式关键字参数覆盖。

## 这行代码的输出

返回值是一个 `PAIIRGraph`，也就是已经编译完成的中间表示图。它不是最终硬件比特流，而是一个已经足够让后端继续处理的图对象，通常已经包含：

- `InputNode` / `OutputNode`
- 融合后的 `OfflineCoreOp` 相关节点，例如 `SequentialOp`、`AccumulateOp`、`PotentialAddOp`
- 路由和布局相关节点，例如 `ReshapeOp`、`ConcatOp`
- 少量仍允许存在的中间态节点，例如 `StandaloneCompOp`、`StandaloneActOp`

最终图会通过 `validate_deployable_graph()` 确认不再残留前端表达层节点，比如 `GeneralAddOp` 和 `SplitOp`。

## `compile_to_paiir` 实际做了什么

下面是它的主要阶段，按代码执行顺序整理：

| 阶段                                      | 输入                                                   | 作用 / 输出                                                                                                        |
| ----------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| 1. 归一化配置                             | `compile_config`、显式关键字参数                       | 先把 `tick_duration`、`auto_reset`、`input_formats` 等参数整理成最终有效值                                         |
| 2. `torch_to_paiir(...)`                  | PyTorch 模型 + `SAMPLE_INPUT`                          | FX tracing + 1:1 lowering，产出原子级 `PAIIRGraph`                                                                 |
| 3. `canonicalize_layout_chains(...)`      | 原子图                                                 | 规范化只影响 layout 的 reshape 链，折叠无意义 reshape                                                              |
| 4. `specialize_general_adds(...)`         | 规范化后的图                                           | 能安全收紧的 `GeneralAddOp` 会变成更部署友好的 `PotentialAddOp`                                                    |
| 5. `elide_layout_invisible_reshapes(...)` | 图                                                     | 去掉芯片不可见的 reshape 包装壳                                                                                    |
| 6. `fuse_to_offline_cores(...)`           | 图                                                     | 把 `StandaloneCompOp` / `StandaloneActOp` 等融合成离线核节点                                                       |
| 7. 结构与语义分析                         | 融合后的图                                             | 运行 `validate_graph()`、`propagate_signal_domain()`、`propagate_data_format()`，并在需要时重写 standalone AvgPool |
| 8. 时序参数分配                           | 图 + `tick_duration` / `auto_reset` / `tick_overrides` | 填充 `tick_start`、`tick_duration`、`tick_initial`                                                                 |
| 9. 可选 AvgPool 标定                      | 图 + `enable_avgpool_calibration`                      | 如果打开，细化共享核 AvgPool+LIF 的阈值                                                                            |
| 10. 最终校验                              | 图                                                     | `validate_compiled_graph()` + `validate_deployable_graph()`，确保图已经是 backend-ready                            |
| 11. 收尾                                  | 图                                                     | 调用 `graph.eval()`，最后返回这个 `PAIIRGraph`                                                                     |

## 最上游：`torch_to_paiir` 做了什么

`compile_to_paiir` 的第一步是调用 [paibox/paiir/lowering/converter.py](../paibox/paiir/lowering/converter.py) 里的 `torch_to_paiir(...)`。这一层主要做的是“把 PyTorch 模型变成 PAIIR 原子图”。

它的关键动作包括：

- 先把模型切到 `eval()` 模式
- 用自定义 FX tracer 做符号追踪
- 把已注册的自定义神经元当成 leaf module
- 把 `Dropout`、`Identity` 之类的模块擦除掉
- 如果传了 `sample_inputs`，就运行 `ShapeProp` 和 `DimsProp`
- 执行 1:1 lowering，创建 `InputNode`、`OutputNode`、`StandaloneCompOp`、`StandaloneActOp`、`ReshapeOp`、`ConcatOp`、`GeneralAddOp` 等原子节点

这里有一个很重要的约束：`sample_inputs` 的 batch size 必须是 1。原因是这个编译路径面向芯片单样本推理，不是批处理训练。

## 中间阶段为什么要反复分析和重写

编译过程中，图并不只是“降一次就完了”。有些 rewrite 会改变拓扑，所以必须重新做分析。

这一部分主要依赖 [paibox/paiir/pipeline/passes.py](../paibox/paiir/pipeline/passes.py) 和 [paibox/paiir/pipeline/rewrite_phase.py](../paibox/paiir/pipeline/rewrite_phase.py)：

- `validate_graph()`：先做结构检查，并清理断开的节点
- `propagate_signal_domain()`：给每个节点补 `output_domain`
- `propagate_data_format()`：给离线核补输入 / 输出 / 权重的数据格式
- `rewrite_standalone_avgpools()`：在语义足够明确时，把 standalone AvgPool 改写成更明确的部署形式
- 如果 rewrite 改了图，就重新跑一轮分析，直到稳定

这也是为什么 `compile_to_paiir` 比 `torch_to_paiir` 更像“真正的编译器”：它不仅降低图，还负责把图补全成可部署状态。

## 在当前 LeNet5 deploy 测试里的实际含义

在这个测试文件里，`model` 不是原始训练模型，而是已经被替换过的部署版本：

- 卷积 / 全连接层的参数换成了导出的量化权重和偏置
- `ReLU` 层换成了 `CustomerLutReLU`
- `CustomerLutReLU` 又通过 `register_neuron(...)` 绑定到 `ANNNodeV25(LutReLUSymmetric(...))`

所以 `compile_to_paiir(model, SAMPLE_INPUT)` 的作用可以理解成：

1. 把这个“部署态 LeNet5”做 FX tracing
2. 把自定义 LUT 激活翻译成 PAIIR 中的 `ANNNodeV25`
3. 把卷积、全连接、激活、池化、reshape 等节点整理成后端可消费的图
4. 给图补齐 signal domain、data format、tick 参数等信息
5. 输出一个可以交给 `Mapper().compile(graph)` 的 `PAIIRGraph`

这也是为什么后面可以直接接：

```python
graph.summary()
mapper = Mapper()
mapper.compile(graph)
```

## 可能遇到的错误

这条 API 不是“无条件成功”的。常见失败点有：

- `UnsupportedOpError`：`strict=True` 时遇到不支持的算子
- `GraphValidationError`：图结构、形状、数据格式或 tick 参数不合法
- `ValueError`：例如 `sample_inputs` 的 batch size 不是 1，或者时序参数非法

## 相关源码入口

- [paibox/paiir/pipeline/compile.py](../paibox/paiir/pipeline/compile.py)
- [paibox/paiir/lowering/converter.py](../paibox/paiir/lowering/converter.py)
- [paibox/paiir/pipeline/passes.py](../paibox/paiir/pipeline/passes.py)
- [paibox/paiir/pipeline/layout_chain_canonicalization.py](../paibox/paiir/pipeline/layout_chain_canonicalization.py)
- [paibox/paiir/pipeline/layout_cross_node_elision.py](../paibox/paiir/pipeline/layout_cross_node_elision.py)
- [paibox/paiir/ir/graph.py](../paibox/paiir/ir/graph.py)

## 一句话总结

`compile_to_paiir(model, SAMPLE_INPUT)` 做的事情，可以概括为：

> 把一个已经准备好部署的 PyTorch 模型，连同一个示例输入，一次性编译成带有布局、语义、格式和时序信息的 `PAIIRGraph`，并确保它已经满足后端可部署约束。