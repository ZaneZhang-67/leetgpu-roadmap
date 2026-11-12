---
id: relu
leetgpuId: 21
titleEn: ReLU
titleZh: ReLU 激活函数
description: 使用 CUDA 对浮点向量逐元素计算 ReLU，练习一维线程映射、边界判断和分段函数
problemUrl: https://leetgpu.com/challenges/relu
updatedAt: 2026-11-12
---
## 题目定义

对一个由 32 位浮点数组成的向量执行 ReLU 激活函数负数变为零，非负数保持原值：

```text
ReLU(x) = max(0, x)
```

### 实现要求

- 不允许使用外部库
- `solve` 函数签名保持不变
- 最终结果写入 `output`

### 示例 1

```text
输入：input = [-2.0, -1.0, 0.0, 1.0, 2.0]
输出：output = [0.0, 0.0, 0.0, 1.0, 2.0]
```

### 示例 2

```text
输入：input = [-3.5, 0.0, 4.2]
输出：output = [0.0, 0.0, 4.2]
```

### 约束

- `1 ≤ N ≤ 100,000,000`
- 性能测试使用 `N = 25,000,000`

## 模板代码解析

平台提供空的 GPU kernel 和完整的启动框架：

```cpp
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

`input` 和 `output` 都指向 GPU 显存，`input` 带有 `const`，表示 kernel 只读取它`N` 是向量元素数每个 Block 启动 256 个线程，Block 数通过向上取整保证线程数量覆盖全部元素需要补全的是 `relu_kernel` 函数体

## 实现思路

ReLU 对每个元素独立计算，可以让一个线程负责一个下标

### 1. 计算全局下标

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

线程 `i` 读取 `input[i]` 并写入 `output[i]`

### 2. 检查边界

Block 数向上取整后，最后一个 Block 可能包含多余线程，因此访问数组前判断：

```cpp
if (i < N)
```

### 3. 计算 ReLU

```cpp
float x = input[i];
output[i] = x > 0.0f ? x : 0.0f;
```

对于负数输出零；零和正数分别输出零与原值不同线程处理不同元素，没有数据依赖或写冲突

### 4. 数据布局

输入和输出都是长度为 `N` 的连续一维浮点数组相邻线程访问相邻元素，地址连续

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float x = input[i];
        output[i] = x > 0.0f ? x : 0.0f;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```
