---
id: swish-gated-linear-unit
leetgpuId: 54
titleEn: Swish-Gated Linear Unit
titleZh: SWiGLU 激活函数
description: 使用 CUDA 将浮点向量分成两半并计算 SWiGLU，练习半区索引、SiLU 门控和输出长度映射
problemUrl: https://leetgpu.com/challenges/swish-gated-linear-unit
updatedAt: 2026-11-12
---
## 题目定义

对形状为 `[N]` 的 `float32` 输入执行 SWiGLU（Swish-Gated Linear Unit）前向计算`N` 为偶数，输入分成长度相同的两半 `x₁` 和 `x₂`：

```text
SiLU(x₁) = x₁ × σ(x₁)
σ(x) = 1 / (1 + e^(-x))
SWiGLU(x₁, x₂) = SiLU(x₁) × x₂
```

输出长度为 `N / 2`

### 实现要求

- 只使用原生功能，不允许使用外部库
- `solve` 函数签名保持不变
- 最终结果写入 `output`

### 示例 1

```text
输入：[1.0, 2.0, 3.0, 4.0]，N = 4
输出：[2.1931758, 7.0463767]
```

输入两半分别为 `x₁ = [1, 2]`、`x₂ = [3, 4]`

### 示例 2

```text
输入：[0.5, 1.0]，N = 2
输出：[0.31122968]
```

### 约束

- `1 ≤ N ≤ 100,000`
- `N` 是偶数
- 输入值范围为 `[-100.0, 100.0]`
- 性能测试使用 `N = 100,000`

## 模板代码解析

平台 CUDA starter 已在 CPU 端计算半区长度：

```cpp
#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
```

`input` 和 `output` 都位于 GPU 显存中`solve` 接收完整输入长度 `N`，计算 `halfN = N / 2`，再按输出长度启动线程kernel 接收的是 `halfN`，而不是原始 `N`需要补全空的 `swiglu_kernel` 函数体

## 实现思路

### 1. 一个线程负责一个输出元素

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

输出只有 `halfN` 个元素，因此合法范围是：

```cpp
if (i < halfN)
```

### 2. 定位输入的两半

输入的一维布局为：

```text
[x₁[0], x₁[1], ..., x₁[halfN-1], x₂[0], x₂[1], ..., x₂[halfN-1]]
```

线程 `i` 读取：

```cpp
float x1 = input[i];
float x2 = input[i + halfN];
```

### 3. 计算门控输出

先计算第一半元素的 SiLU，再乘以第二半对应元素：

```cpp
float sigmoid = 1.0f / (1.0f + expf(-x1));
output[i] = x1 * sigmoid * x2;
```

相邻线程读取两个半区中的相邻元素并写入相邻输出不同线程负责不同下标，无需同步

### 4. 数值范围

题目输入范围为 `[-100, 100]`当 `x1` 很小时，`expf(-x1)` 可能溢出为正无穷，直接公式会把极小的负 SiLU 值下溢近似为带符号零，误差远低于题目容差；当 `x1` 很大时，Sigmoid 趋近 1，输出趋近 `x1 × x2`

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < halfN) {
        float x1 = input[i];
        float x2 = input[i + halfN];
        float sigmoid = 1.0f / (1.0f + expf(-x1));
        output[i] = x1 * sigmoid * x2;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
```
