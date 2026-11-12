---
id: batch-normalization
leetgpuId: 40
titleEn: Batch Normalization
titleZh: 批归一化
description: 对 N×C 输入按特征列计算总体均值与方差，再使用 gamma、beta 和 epsilon 完成 BatchNorm 前向传播
problemUrl: https://leetgpu.com/challenges/batch-normalization
updatedAt: 2026-11-12
---
## 题目定义

输入是形状 `[N, C]` 的二维张量，其中 `N` 是 batch 大小，`C` 是特征数Batch Normalization 对每个特征列 `c` 独立计算统计量：

```text
mean[c] = sum(input[n,c]) / N
variance[c] = sum((input[n,c] - mean[c])²) / N
output[n,c] = gamma[c] * (input[n,c] - mean[c])
              / sqrt(variance[c] + eps) + beta[c]
```

方差分母是 `N`，即总体方差，不使用无偏估计的 `N - 1`

### 实现要求与约束

- 不允许使用外部库
- `solve` 函数签名必须保持不变
- 结果写入形状 `[N, C]` 的 `output`
- `1 ≤ N ≤ 10,000`
- `1 ≤ C ≤ 1,024`
- `eps = 1e-5`
- `input` 范围为 `[-100, 100]`
- `gamma` 范围为 `[0.1, 10]`
- `beta` 范围为 `[-10, 10]`
- 性能测试使用 `N = 5000`、`C = 512`

### 示例

```text
input = [[1, 2], [3, 4], [5, 6]]
gamma = [1, 1]
beta = [0, 0]
eps = 1e-5
output ≈ [[-1.224, -1.224], [0, 0], [1.224, 1.224]]
```

两个特征列分别沿 batch 维归一化

## 模板代码解析

CUDA starter 的签名为：

```cpp
#include <cuda_runtime.h>

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma,
                      const float* beta, float* output,
                      int N, int C, float eps) {}
```

`input` 与 `output` 按行主序存储，位置 `(n,c)` 的下标是 `n * C + c`，`gamma` 和 `beta` 的长度都是 `C`

参考实现让一个线程负责一个特征列这个线程先遍历 `N` 个样本计算均值和方差，再遍历一次写出该列的全部结果这种映射不需要额外统计量缓冲，也没有跨线程写冲突

## 实现思路

### 1. 按特征列归约

线程 `c` 访问：

```text
input[c], input[C+c], input[2C+c], ...
```

第一遍求和得到 `mean`；第二遍累计平方差得到总体方差

`N = 1` 时方差为零，分母仍有 `sqrt(eps)`，而 `input - mean` 为零，所以输出恰好是 `beta[c]`

### 2. 应用仿射变换

归一化后再乘 `gamma[c]`、加 `beta[c]`参数按特征共享，不随样本 `n` 改变

### 3. 数值与并行性

输入输出保持 FP32，但均值与平方差使用双精度累加，避免大偏置、低方差数据在顺序 FP32 求和时发生明显消减`eps` 加在总体方差内部再开平方；优化版还可以让一个 Block 协作处理一列并做配对归约

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void batch_norm(const float* input, const float* gamma,
                           const float* beta, float* output,
                           int N, int C, float eps) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    double sum = 0.0;
    for (int n = 0; n < N; ++n) {
        sum += input[n * C + c];
    }
    double mean = sum / N;

    double squared = 0.0;
    for (int n = 0; n < N; ++n) {
        double diff = (double)input[n * C + c] - mean;
        squared += diff * diff;
    }
    double variance = squared / N;
    double inverseStd = 1.0 / sqrt(variance + eps);

    for (int n = 0; n < N; ++n) {
        int index = n * C + c;
        double normalized = ((double)input[index] - mean) * inverseStd;
        output[index] = (float)(gamma[c] * normalized + beta[c]);
    }
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma,
                      const float* beta, float* output,
                      int N, int C, float eps) {
    int threads = 256;
    int blocks = (C + threads - 1) / threads;
    batch_norm<<<blocks, threads>>>(input, gamma, beta, output, N, C, eps);
    cudaDeviceSynchronize();
}
```
