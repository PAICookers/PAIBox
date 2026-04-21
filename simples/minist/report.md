# MNIST 模型量化实验总结


## 1. 数据集 (Dataset)
两个模型均基于经典的 **MNIST 手写数字数据集** 训练和验证：
- **数据格式**：单通道灰度图像，尺寸为 $1 \times 28 \times 28$。
- **预处理**：应用了标准的张量化（ToTensor）以及归一化（Normalize）处理。
```
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
```


## 2. 模型结构描述

### 2.1 全连接网络 (FC_MNIST)
多层感知机模型，其结构主要用于处理展平（flattened）后的 1D 图像特征。
- **Layer 1**: `Linear(784, 512)` -> `BatchNorm1d(512)` -> `ReLU`
- **Layer 2**: `Linear(512, 256)` -> `BatchNorm1d(256)` -> `ReLU`
- **Output Layer**: `Linear(256, 10)`

### 2.2 卷积神经网络 (LeNet-5)
经典卷积神经网络，主要包含卷积特征提取和全连接分类两部分。
- **Feature Extractor**:
  - `Conv2d(1, 6, kernel_size=5, padding=2)` -> `ReLU` -> `MaxPool2d`
  - `Conv2d(6, 16, kernel_size=5)` -> `ReLU` -> `MaxPool2d`
- **Classifier**:
  - 展平特征维度至 `16 * 5 * 5`
  - `Linear(400, 120)` -> `ReLU`
  - `Linear(120, 84)` -> `ReLU`
  - `Linear(84, 10)`

## 3. 量化规格设计 (Quantization Specifications)
整体配置围绕**全局对称量化 (Symmetric Quantization)** 展开。具体规格如下：

- **量化机制 (Scheme)**：
  - 基于 FX 图 (FX Graph) 插桩进行动态或 PTQ 初始化，并插入对应的 `Observer` 机制进行张量统计。
  - 选择全对称方式，`qscheme` 采用 `torch.per_tensor_symmetric` 参数，意味着全对称量化，zp=0。这样方便网络上板部署
- **量化精度 (Data Type)**：
  - **权重 (Weights)**：量化为 `torch.qint8`（有符号 8-bit 整数）。
  - **激活值 (Activations)**：默认映射校准为 `torch.qint8` 或在必要时可作为有符号张量统计。
- **仿真**
  - 完全使用int8进行MAC计算；激活、反量化和再量化使用LUT执行。完全模拟硬件计算行为。


## 4. 模型精度 (Accuracy)

fc模型 ：fp精度-98.52% 、int8量化后精度-98.49%

lenet5模型：fp精度-98.63% 、 int8量化后精度-98.58%

## 5. 文件组成
模型：model.py
fp权重：checkpoints文件夹
int8权重：exported_params_symmetric文件夹
lut配置：readme.txt
其他：量化文件，不需要使用