# 量化工具使用说明


## 1. 依赖与前期准备

1. **预训练模型文件**：请确保在运行前，已经完成 FP32 模型的训练并保存好了权重。
   
2. **数据集**：数据集供校准及测试精度使用。(可选)

## 2. 使用方法

激活含有依赖（如 `torch`, `torchvision`, 以及当前项目的 `paibox` 等包）的 Python 环境，在终端中运行`export_quantized.py`：


运行过程中控制台将输出以下信息：
1. 模型读取提示。
2. 数据集读取、 Observer 插入提示。
3. FX 校准前向推理的进度（大约抽取 10 个 Batch）。
4. 测试集下的精度评测对比，例如:
   ```text
   => FP32 模型准确率: xx%
   => 量化后模型准确率: xx%
   ```
5. 完成所有产物的导出。一切正常后，你可以在本级目录下看到生成的 `outputs` 文件夹，所有的导出都在该目录下。

## 3. 关键注意事项

⚠️ **强烈要求保持 `get_quantization_config` 函数内容不变**

当前脚本内的 `get_quantization_config` 方法为：

```python
def get_quantization_config(symmetric: bool = True):
    qscheme_act = torch.per_tensor_symmetric if symmetric else torch.per_tensor_affine
    qscheme_weight = torch.per_tensor_symmetric

    my_qconfig = QConfig(
        activation=MinMaxObserver.with_args(
            dtype=torch.quint8,
            qscheme=qscheme_act,
        ),
        weight=MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=qscheme_weight,
        ),
    )

    qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    qconfig_mapping.set_object_type(nn.Conv2d, my_qconfig)
    qconfig_mapping.set_object_type(nn.Linear, my_qconfig)
    qconfig_mapping.set_object_type(nn.ReLU, my_qconfig)
    return qconfig_mapping
```

并保证 symmetric = True

**请勿更改上述量化配置！** 

⚠️ [3] 融合模型并插入 Observer" 

这一步操作中的fuse_fx和prepare_fx顺序不可更改，仿照样例脚本即可。

## 4.output文件夹
txt文件 ：计算图和forward

json文件 ：各个模块的量化参数，可用于配置lut等

export_params :各个模块的权重和偏置

## 5. 量化细节
量化工具会自动识别并替换的结构有一下几个：
- conv+(bn)+relu
- liner+(bn)+relu
- relu(x +conv(y))
  
  在这个当中，会出现M、n两个参数。具体的硬件部署需要用到三个core。第一个core:计算激活值y的卷积并输出膜电平；第二个core：对输入激活值x 乘以权重M ,然后在乘法泄露n后输出膜电平；第三个core；进行膜电平加法并配置LUTReLU


- conv(+bn)   
  
  *注意在工具中conv(+bn) 会在后面自动补用LUTLinear从而输出激活值，而liner(+bn)并未设置。这是因为单独的liner层我们一般认为是模型的最后一个分类层，只输出膜电平即可*
  
  加上上述的自动适配LUTLinear功能，我们可以完整的适配res-block

  ps :由于现在的conv(+bn)自动设置为了输出激活值，因此可以适配conv1(conv2(x))的结构。liner没有添加LUT，因此不适配liner1(liner2(x))。经过测试conv1(conv2(x))结构精度损失较大，不推荐使用。
  
  

