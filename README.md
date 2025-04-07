# Neural-Network
# CIFAR-10 图像分类神经网络

这个项目实现了一个三层神经网络模型用于CIFAR-10图像分类任务。该模型使用纯NumPy构建，无需深度学习框架，适合学习神经网络基础原理。

## 完整文件结构
```bash
cifar10-classifier/
├── main.py                # 主运行脚本
├── model.py               # 模型文件
├── train.py               # 训练
├── test.py                # 测试
├── hparam_search.py       # 超参数搜索
├── search_models/          # 参数搜索输出目录
├── trained_models/         # 模型训练保存目录（需要自己新建这个文件夹）
│
└── cifar-10-batches-py/   # 必须的数据集目录（需自行下载）
    ├── data_batch_1       # 训练批次1
    ├── data_batch_2       # 训练批次2
    ├── data_batch_3       # 训练批次3
    ├── data_batch_4       # 训练批次4
    ├── data_batch_5       # 训练批次5
    ├── test_batch         # 测试批次
    └── batches.meta       # 标签元数据
```

## 功能概述

该项目包含以下主要功能：

1. **数据加载与预处理**：从CIFAR-10数据集加载图像数据，并进行标准化和one-hot编码处理
2. **三层神经网络模型**：实现了带有一个隐藏层的神经网络，支持ReLU和Sigmoid激活函数
3. **模型训练**：使用SGD优化器训练模型，支持学习率衰减和L2正则化，应用了交叉熵损失
4. **模型评估**：在测试集上评估模型性能
5. **可视化功能**：绘制训练过程中的损失曲线和准确率变化，可视化学习到的权重

## 数据集准备

1. 从[CIFAR-10官方网站](https://www.cs.toronto.edu/~kriz/cifar.html)下载CIFAR-10 Python版本数据集
2. 解压下载的文件到数据集目录，文件位置如“完整文件结构”所示
3. 在`main.py`中修改`data_dir`变量指向您解压的CIFAR-10数据集目录

## 使用方法

### 查找最优参数

通过以下命令来加载数据查找最优参数：

```bash
python main.py --mode search --data_dir cifar-10-batches-py --output_dir search_models
```

### 修改模型参数

可以在`main.py`中修改以下参数来调整模型：

- `hidden_size`：隐藏层的神经元数量
- `activation`：激活函数类型（'relu'或'sigmoid'）
- `epochs`：训练轮数
- `batch_size`：小批量大小
- `lr`：初始学习率
- `reg_lambda`：L2正则化系数
- `lr_decay`：学习率衰减因子

### 训练模型

通过找到的最优参数训练模型：

```bash
python main.py --mode train --data_dir cifar-10-batches-py --output_dir trained_models
```

### 测试模型

利用训练得到的模型来测试数据，评估模型性能

```bash
python main.py --mode test --model_path trained_models/best_model.npz
```

## 代码解释

### 数据处理

- `load_cifar10`函数负责加载和预处理CIFAR-10数据集
- 数据预处理包括：归一化（除以255）、划分验证集、转换为one-hot编码

### 神经网络模型

`ThreeLayerNN`类实现了三层神经网络模型，包括：

- `__init__`：初始化模型参数
- `forward`：前向传播
- `compute_loss`：计算交叉熵损失和正则化损失
- `backward`：反向传播，计算梯度
- `update_params`：使用梯度更新参数
- `save_weights`/`load_weights`：保存和加载模型权重

### 训练与评估

- `train`函数实现了模型训练循环，包括SGD、学习率衰减和模型保存
- `test`函数在测试集上评估模型性能

### 可视化

- `plot_training_curves`：绘制训练和验证损失曲线以及验证准确率曲线
- `visualize_weights`：可视化第一层权重

