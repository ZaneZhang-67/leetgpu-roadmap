---
id: gelu-activation
leetgpuId: 124
titleEn: GELU Activation
titleZh: GELU 激活函数
description: 使用 CUDA erf 精确公式逐元素计算 Gaussian Error Linear Unit
problemUrl: https://leetgpu.com/challenges/gelu-activation
updatedAt: 2026-11-12
---
## 题目定义

对长度为 `N` 的 FP32 向量逐元素计算精确 GELU：

```text
GELU(x) = x/2 * (1 + erf(x/sqrt(2)))
```

运行时使用 `torch.nn.functional.gelu(..., approximate="none")`，因此不能替换为 tanh 近似。

### 示例

```text
X=[0,1,-1,2]       → Y=[0,0.8413,-0.1587,1.9545]
X=[0.5,-0.5,3,-3] → Y=[0.3457,-0.1543,2.9959,-0.0041]
```

**约束与测试**

- `1≤N≤100,000,000`，输入为有限 FP32
- 测试覆盖零、正负值、绝对值 50 和非二次幂长度 255
- 性能测试 `N=50,000,000`，容差 `atol=rtol=1e-5`

## 模板代码解析

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* X, float* Y, int N) {}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
```

官方 starter 已提供空 kernel、256 线程启动配置和同步调用，只需补全逐元素 kernel。

## 实现思路

一个线程处理一个元素，检查下标后调用单精度 `erff`。常量 `1/sqrt(2)` 写为 FP32，表达式严格对应 exact GELU。线程之间没有数据依赖，读写均连续。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* X, float* Y, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        float x = X[index];
        Y[index] = 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
    }
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
```
