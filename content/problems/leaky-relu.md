---
id: leaky-relu
leetgpuId: 23
titleEn: Leaky ReLU
titleZh: Leaky ReLU 激活函数
description: 使用 CUDA 对浮点向量逐元素计算 Leaky ReLU，理解固定泄漏系数与分段函数实现
problemUrl: https://leetgpu.com/challenges/leaky-relu
updatedAt: 2026-11-12
---
## 题目定义

对浮点向量逐元素执行 Leaky ReLU 激活函数本题固定泄漏系数 `α = 0.01`：

```text
f(x) = x         当 x > 0
f(x) = 0.01 × x  当 x ≤ 0
```

与普通 ReLU 不同，Leaky ReLU 会保留负数分支的一个小斜率

### 实现要求

- 不允许使用外部库
- `solve` 函数签名保持不变
- 最终结果写入向量 `output`
- 使用 `α = 0.01`

### 示例 1

```text
输入：x = [1.0, -2.0, 3.0, -4.0]
输出：y = [1.0, -0.02, 3.0, -0.04]
```

### 示例 2

```text
输入：x = [-1.5, 0.0, 2.5, -3.0]
输出：y = [-0.015, 0.0, 2.5, -0.03]
```

### 约束

- `1 ≤ N ≤ 100,000,000`
- `-1000.0 ≤ input[i] ≤ 1000.0`
- 性能测试使用 `N = 50,000,000`

## 模板代码解析

平台提供以下 CUDA 模板：

```cpp
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

`input` 是只读输入向量，`output` 是输出向量，两者都在 GPU 显存中`N` 表示元素数量模板使用一维 Grid，每个 Block 含 256 个线程，并通过向上取整覆盖全部元素需要作答的部分是空的 kernel 函数体

## 实现思路

Leaky ReLU 是逐元素运算，每个线程只处理一个输入值

### 1. 建立线程与元素的映射

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

线程 `i` 对应 `input[i]` 和 `output[i]`

### 2. 保护数组边界

启动线程数可能大于 `N`，因此先判断：

```cpp
if (i < N)
```

只有合法线程才能读取和写入数组

### 3. 实现分段函数

```cpp
float x = input[i];
output[i] = x > 0.0f ? x : 0.01f * x;
```

题目把 `x = 0` 归入 `x ≤ 0` 分支，但 `0.01 × 0` 仍然为零

### 4. 数据布局与并行关系

输入和输出都是连续的一维 `float` 数组相邻线程读取相邻输入并写入相邻输出不同元素之间没有依赖，不需要线程同步

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float x = input[i];
        output[i] = x > 0.0f ? x : 0.01f * x;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```
