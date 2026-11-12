---
id: simple-inference
leetgpuId: 41
titleEn: Simple Inference
titleZh: 简单模型推理
description: 使用 PyTorch 对已加载的线性层执行前向推理，并将结果写入平台提供的输出张量
problemUrl: https://leetgpu.com/challenges/simple-inference
updatedAt: 2026-11-12
---
## 题目定义

对一个已经训练并加载完成的 `torch.nn.Linear` 模型执行前向推理给定输入张量与模型，计算：

```text
output = input @ weight.T + bias
```

其中 `weight` 的形状为 `[output_size, input_size]`，`bias` 的形状为 `[output_size]`

### 实现要求

- 使用 PyTorch 内置函数和运算
- `solve` 函数签名保持不变
- 最终结果写入 `output` 张量
- 模型已加载完成，可以直接推理

### 示例 1

```text
输入：input = [[1.0, 2.0]]
      batch_size = 1，input_size = 2
      weight = [[0.5, 1.0], [1.5, 0.5]]
      bias = [0.1, 0.2]
输出：output = [[2.6, 2.7]]
```

### 示例 2

```text
输入：input = [[1.0], [2.0], [3.0]]
      batch_size = 3，input_size = 1
      weight = [[2.0]]
      bias = [1.0]
输出：output = [[3.0], [5.0], [7.0]]
```

### 约束

- `1 ≤ batch_size ≤ 1000`
- `1 ≤ input_size ≤ 1000`
- `1 ≤ output_size ≤ 1000`
- 输入值范围为 `[-10.0, 10.0]`
- 性能测试使用 `batch_size = 1000`

## 模板代码解析

这道题的本地平台数据只提供 JAX 和 PyTorch starter，没有 CUDA、CuTe、Mojo 或 Triton starter题目要求使用 PyTorch 内置运算并把结果写入 `output`，因此参考实现采用 PyTorch，严格保持其真实 starter 签名：

```python
import torch
import torch.nn as nn


# input, model, and output are on the GPU
def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
    pass
```

- `input`：位于 GPU 上的输入张量，形状为 `[batch_size, input_size]`
- `model`：已经加载且可直接调用的 `nn.Module`，题目中是线性层
- `output`：平台提供的 GPU 输出张量，形状为 `[batch_size, output_size]`

与返回新张量的接口不同，这个 PyTorch starter 要求写入传入的 `output`对象模板中没有尺寸参数，也不需要自行配置 Grid 或线程

## 实现思路

### 1. 调用模型完成前向传播

PyTorch 模型实现了调用接口：

```python
result = model(input)
```

对于题目给定的线性层，这等价于批量计算：

```text
result[b,o] = bias[o] + Σ input[b,k] × weight[o,k]
```

其中 `b` 遍历 batch，`o` 遍历输出特征，`k` 遍历输入特征

### 2. 写入已有输出张量

函数签名没有返回值位置，题面要求结果存入 `output`，因此使用：

```python
output.copy_(result)
```

`copy_` 是原地操作，会保留平台传入的输出张量对象

### 3. 关闭梯度记录

这是推理任务，不需要构建反向传播图可以使用：

```python
with torch.no_grad():
```

它不会改变模型计算结果，同时避免记录推理过程的梯度信息

### 4. 数据布局与边界

输入、权重、偏置和输出的维度由 PyTorch 模型负责匹配题目保证形状合法，矩阵乘法、偏置广播和 GPU 并行均由 PyTorch 内置算子执行，因此不需要手写元素索引或线程边界

## 参考实现

```python
import torch
import torch.nn as nn


# input, model, and output are on the GPU
def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
    with torch.no_grad():
        output.copy_(model(input))
```
