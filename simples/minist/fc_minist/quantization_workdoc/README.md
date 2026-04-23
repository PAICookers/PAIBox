# 量化工作文档

这份文档是给协作同学看的轻量版说明。重点是把量化流程讲清楚，把可运行逻辑拆到单独文件里，避免 README 里塞太多代码。

本文档对应 MNIST 的 LeNet-5 示例，直接复用现有的 [lenet5_minist/model.py](../../lenet5_minist/model.py) 和 [lenet5_minist/checkpoints/best_model.pth](../../lenet5_minist/checkpoints/best_model.pth)。

## 1. 文件说明

- [model.py](model.py)：LeNet-5 模型定义，已复制到本目录，方便独立查看和调用。
- [export_quantized_pth.py](export_quantized_pth.py)：读取现有 `best_model.pth`，执行量化流程，最后导出量化后的模型参数 pth，并同步保存校准数据、层参数 JSON 和计算图 TXT。
- `outputs/`：脚本运行后生成的输出目录。
- `outputs/calibration/`：保存校准样本 `calibration_samples.pt` 和 observer 统计 `calibration_stats.json`。
- `outputs/quantized_model_layers.json`：保存转换后每一层的输入、输出和权重 scale / zp。
- `outputs/quantized_model_graph.txt`：保存量化之后的计算图文本。

## 2. 这版流程做什么

这版流程把量化主线讲清楚：加载 FP32 模型、准备量化配置、做少量校准、转换为量化模型，然后保存量化后的 `state_dict`。

文档只展示最终会落盘的结果，不展开中间的手工算子替换细节。

## 3. 量化脚本的职责

脚本的职责很明确：

- 读取现成的 FP32 权重。
- 使用 MNIST 训练集的少量 batch 做校准。
- 转换得到量化模型。
- 导出校准样本和 observer 统计，方便复现实验和查看量化参数。
- 将量化模型的参数保存为一个 pth 文件。

## 4. 输出结果

默认输出文件是 `outputs/quantized_lenet5_state_dict.pth`，校准数据会另外保存到 `outputs/calibration/`，转换后的层参数和计算图会分别保存到 `outputs/quantized_model_layers.json` 和 `outputs/quantized_model_graph.txt`。

这个 pth 保存的是量化模型的 `state_dict`，适合后续加载、检查或者继续做部署处理。

## 5. CNN 示例的说明

这里的 CNN 示例就是 LeNet-5。

它和原始 LeNet-5 示例保持一致的输入尺寸 `1 x 28 x 28`，量化流程沿用当前脚本，并把结果保存为 pth。对于其他 CNN，只要替换模型和对应权重文件，整体流程也可以沿用。

## 6. 你需要记住的点

- 现有的 `best_model.pth` 已经可以直接拿来读。
- 这份文档不再强调手动量化算子替换。
- 如果以后需要导出校准统计或分层量化信息，可以在脚本里继续扩展，但当前版本先只保留量化模型 pth 的导出。
