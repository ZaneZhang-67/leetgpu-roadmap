---
id: dot-product
leetgpuId: 17
titleEn: Dot Product
titleZh: 向量点积
description: 使用 CUDA 计算两个浮点向量的点积，通过线程局部乘加与两级共享内存归约稳定得到标量结果
problemUrl: https://leetgpu.com/challenges/dot-product
updatedAt: 2026-11-12
---
## 题目定义

给定两个长度相同的 32 位浮点向量 `A` 和 `B`，计算点积并把标量结果写入 `result[0]`

```text
result[0] = Σ A[i] × B[i]，其中 0 ≤ i < N
```

### 实现要求

- 只使用 GPU 原生功能，不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写入输出变量 `result`

### 示例 1

```text
输入：A = [1.0, 2.0, 3.0, 4.0]
      B = [5.0, 6.0, 7.0, 8.0]
输出：result = 70.0
```

```text
1.0×5.0 + 2.0×6.0 + 3.0×7.0 + 4.0×8.0 = 70.0
```

### 示例 2

```text
输入：A = [0.5, 1.5, 2.5]
      B = [2.0, 3.0, 4.0]
输出：result = 15.5
```

### 约束

- `A` 和 `B` 的长度相同
- `1 ≤ N ≤ 100,000,000`
- 性能测试使用 `N = 4,194,304`

## 模板代码解析

平台提供固定的 CUDA 入口：

```cpp
#include <cuda_runtime.h>

// A, B, result are device pointers
extern "C" void solve(
    const float* A,
    const float* B,
    float* result,
    int N
) {}
```

- `A`、`B` 是 GPU 上长度为 `N` 的只读向量
- `result` 指向 GPU 上保存最终标量的空间
- `N` 是向量长度

模板没有提供 kernel、结果初始化和启动配置，这些均由实现补充

## 实现思路

点积先逐元素相乘，再把全部乘积求和结构与 Reduction 相同，只是线程局部累加的值改为 `A[i] * B[i]`

### 1. 线程局部乘加

第一阶段启动多个 Block，每个线程使用 Grid-stride loop 处理多个位置：

```cpp
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

float sum = 0.0f;
for (int i = index; i < N; i += stride) {
    sum = fmaf(A[i], B[i], sum);
}
```

`i < N` 同时保护两个输入数组访问相邻线程读取相邻元素，形成连续访存

### 2. Block 内归约

每个线程把局部和写入共享内存，然后采用折半归约：

```text
256 个局部和 → 128 → 64 → 32 → ... → 1 个 Block 部分和
```

每轮结束后调用 `__syncthreads()`，确保下一轮读取的是完整结果第一阶段把每个 Block 的结果写入独立的 `partials[blockIdx.x]`，不存在跨 Block 写冲突

### 3. 第二级确定性归约

第二个 kernel 只启动一个 Block，由 256 个线程读取 `partials`，再次执行同样的共享内存归约，最后由 0 号线程写入 `result[0]`

这样不需要跨 Block 的浮点 `atomicAdd`，部分和的合并次序固定，重复运行不会因 Block 调度顺序不同而改变结果，也更适合题目的严格误差阈值

### 4. 覆盖最大输入

第一阶段最多启动 1024 个 Block，更多输入由 Grid-stride loop 处理：

```cpp
int blocksPerGrid = ceil(N / 256);
blocksPerGrid = min(blocksPerGrid, 1024);
```

`partials` 只需保存至多 1024 个浮点数第二阶段的单个 Block 通过跨步循环读取这些部分和，因此 `N = 100,000,000` 时仍能覆盖全部元素

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

constexpr int kThreads = 256;

__global__ void dot_product_partials(
    const float* A,
    const float* B,
    float* partials,
    int N
) {
    __shared__ float shared[kThreads];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = index; i < N; i += stride) {
        sum = fmaf(A[i], B[i], sum);
    }

    shared[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) partials[blockIdx.x] = shared[0];
}

__global__ void reduce_partials(
    const float* partials,
    float* result,
    int count
) {
    __shared__ float shared[kThreads];

    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < count; i += blockDim.x) {
        sum += partials[i];
    }

    shared[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) result[0] = shared[0];
}

// A, B, result are device pointers
extern "C" void solve(
    const float* A,
    const float* B,
    float* result,
    int N
) {
    int blocksPerGrid = (N + kThreads - 1) / kThreads;
    if (blocksPerGrid > 1024) blocksPerGrid = 1024;

    float* partials = nullptr;
    cudaMalloc(&partials, static_cast<size_t>(blocksPerGrid) * sizeof(float));

    dot_product_partials<<<blocksPerGrid, kThreads>>>(
        A, B, partials, N
    );
    reduce_partials<<<1, kThreads>>>(partials, result, blocksPerGrid);

    cudaDeviceSynchronize();
    cudaFree(partials);
}
```
