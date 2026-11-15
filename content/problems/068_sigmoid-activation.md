---
id: sigmoid-activation
leetgpuId: 68
titleEn: Sigmoid Activation
titleZh: Sigmoid 激活函数
description: 使用 CUDA 对大型 FP32 向量逐元素计算数值稳定的 Sigmoid，避免极端负值导致指数溢出
problemUrl: https://leetgpu.com/challenges/sigmoid-activation
updatedAt: 2026-11-12
---
## 题目定义

给定长度为 `N` 的 FP32 向量 `X`，逐元素计算 Sigmoid 并写入 `Y`：

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

Sigmoid 将实数映射到 `(0, 1)`

### 实现要求

- 不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写入 `Y`

### 示例 1

```text
输入：X = [0.0, 1.0, -1.0, 2.0]
输出：Y ≈ [0.5, 0.7311, 0.2689, 0.8808]
```

### 示例 2

```text
输入：X = [0.5, -0.5, 3.0, -3.0]
输出：Y ≈ [0.6225, 0.3775, 0.9526, 0.0474]
```

### 约束

- `1 ≤ N ≤ 100,000,000`
- 输入是有限的 32 位浮点数
- 功能测试包含 `±1000`、零、正负混合值及非 Block 整数倍长度 `N = 255`
- 性能测试使用 `N = 50,000,000`，输入范围为 `[-10, 10]`
- 判定的绝对误差和相对误差均为 `1e-5`

## 模板代码解析

平台提供了空 kernel 和完整启动框架：

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(
    const float* X,
    float* Y,
    int N
) {}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (N + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
```

- `X`、`Y` 是长度为 `N` 的设备数组
- starter 已按向上取整计算 Grid 大小
- 作答需要补全逐元素 kernel，并保留边界判断

官方 `challenge.py` 的运行时签名把 `N` 描述为 `size_t`，但当前 CUDA starter 的固定签名使用 `int N`。本页参考实现严格保持 starter 签名；题目上限一亿，可由 `int` 安全表示

## 实现思路

一个线程处理一个元素。为了让极端正负输入都保持数值稳定，按符号使用等价公式：

```text
x ≥ 0：sigmoid(x) = 1 / (1 + exp(-x))
x < 0：sigmoid(x) = exp(x) / (1 + exp(x))
```

当 `x` 很负时，直接计算 `exp(-x)` 可能溢出。分支公式只计算 `exp(x)`，它会趋近于 0，不会产生 `inf / inf` 或无效结果

同理，正数分支中的 `exp(-x)` 不大于 1。极端输入 `1000` 和 `-1000` 会稳定得到接近 1 和 0 的结果

线程在访问前判断 `index < N`，覆盖 `N = 255` 等非 Block 整数倍长度。时间复杂度为 `O(N)`，额外空间为 `O(1)`

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(
    const float* X,
    float* Y,
    int N
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) return;

    float x = X[index];
    if (x >= 0.0f) {
        Y[index] = 1.0f / (1.0f + expf(-x));
    } else {
        float exponential = expf(x);
        Y[index] = exponential / (1.0f + exponential);
    }
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (N + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
```
