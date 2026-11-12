---
id: count-3d-array-element
leetgpuId: 45
titleEn: Count 3D Array Element
titleZh: 统计三维数组元素
description: 将 N×M×K 的 int32 数组线性化后使用 CUDA 归约，统计目标值 P 的出现次数
problemUrl: https://leetgpu.com/challenges/count-3d-array-element
updatedAt: 2026-11-12
---
## 题目定义

给定形状为 `N × M × K` 的 32 位整数数组 `input` 和目标值 `P`，统计数组中等于 `P` 的元素数量，并写入 `output[0]`

**实现要求**

- 只使用原生功能，不允许使用外部库
- `solve` 函数签名保持不变
- 最终结果写入 `output`

**示例**

```text
输入：input = [[[1, 2, 3], [4, 5, 1]],
               [[1, 1, 1], [2, 2, 2]]]
      N = 2，M = 2，K = 3，P = 1
输出：5
```

**约束与测试**

- `1 ≤ N, M, K ≤ 1000`
- `1 ≤ input[i], P ≤ 100`
- 输入形状为 `(N, M, K)`，输出形状为 `(1,)`，类型均为 `int32`
- 功能测试包含 `100 × 200 × 300`
- 性能测试使用 `N = M = K = 500`、`P = 2`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output,
                      int N, int M, int K, int P) {}
```

这里 `K` 是第三个维度长度，待统计值是 `P`，两者不能混淆输入在显存中连续存储，模板没有提供 kernel 和启动配置

## 实现思路

三维坐标 `(n, m, k)` 的线性下标为：

```text
(n × M + m) × K + k
```

统计只需遍历所有元素，可以把输入视为长度 `N × M × K` 的一维数组三个维度的理论乘积最大为十亿，仍低于 `int32` 上限，但使用 `long long` 计算总数和索引可避免中间乘法风险

每个线程通过 grid-stride 循环处理多个元素，Block 内共享内存归约后，只执行一次 `atomicAdd`输出来自 `torch.empty(1)`，必须先清零最大计数为十亿，仍可由 `int32` 表示

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void count_3d_kernel(const int* input, int* output,
                                long long total, int P) {
    __shared__ int counts[256];
    int tid = threadIdx.x;
    long long index = (long long)blockIdx.x * blockDim.x + tid;
    long long stride = (long long)blockDim.x * gridDim.x;
    int local = 0;

    for (long long i = index; i < total; i += stride) {
        local += input[i] == P;
    }
    counts[tid] = local;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) counts[tid] += counts[tid + offset];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, counts[0]);
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output,
                      int N, int M, int K, int P) {
    cudaMemset(output, 0, sizeof(int));
    long long total = (long long)N * M * K;
    int threads = 256;
    long long required = (total + threads - 1) / threads;
    int blocks = required > 1024 ? 1024 : (int)required;
    count_3d_kernel<<<blocks, threads>>>(input, output, total, P);
    cudaDeviceSynchronize();
}
```
