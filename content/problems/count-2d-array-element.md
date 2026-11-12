---
id: count-2d-array-element
leetgpuId: 44
titleEn: Count 2D Array Element
titleZh: 统计二维数组元素
description: 将 N×M 的 int32 数组展平后使用 CUDA 并行归约，统计等于 K 的元素数量
problemUrl: https://leetgpu.com/challenges/count-2d-array-element
updatedAt: 2026-11-12
---
## 题目定义

给定形状为 `N × M` 的 32 位整数数组 `input` 和目标值 `K`，统计所有位置中等于 `K` 的元素数量，并写入 `output[0]`

**实现要求**

- 只使用原生功能，不允许使用外部库
- `solve` 函数签名保持不变
- 最终结果写入 `output`

**示例**

```text
输入：input = [[1, 2, 3],
               [4, 5, 1]]
      K = 1
输出：2
```

**约束与测试**

- `1 ≤ N, M ≤ 10,000`
- `1 ≤ input[i], K ≤ 100`
- 输入是形状为 `(N, M)` 的连续 `int32` 张量，输出是形状为 `(1,)` 的 `int32` 张量
- 功能测试覆盖全相同值和 `1000 × 1000` 数组
- 性能测试使用 `N = M = 10,000`、`K = 1`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int M, int K) {}
```

`N` 是行数，`M` 是列数，数组在显存中按连续行存储模板只规定入口，需要自行初始化输出并启动统计 kernel

## 实现思路

二维位置 `(row, col)` 的线性下标为 `row * M + col`统计不依赖坐标本身，因此可直接把数组看作长度为 `N × M` 的一维连续数组

`N × M` 最大为一亿，使用 `long long` 计算总元素数和 grid-stride 下标，避免维度乘法在中间表达式中产生窄化每个线程先累加局部匹配数，再在共享内存中归约；每个 Block 对输出执行一次原子加法

`challenge.py` 使用 `torch.empty(1)` 创建输出，因此启动前必须调用 `cudaMemset` 清零输出最大匹配数为一亿，可存入 `int32`

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void count_2d_kernel(const int* input, int* output,
                                long long total, int K) {
    __shared__ int counts[256];
    int tid = threadIdx.x;
    long long index = (long long)blockIdx.x * blockDim.x + tid;
    long long stride = (long long)blockDim.x * gridDim.x;
    int local = 0;

    for (long long i = index; i < total; i += stride) {
        local += input[i] == K;
    }
    counts[tid] = local;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) counts[tid] += counts[tid + offset];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, counts[0]);
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    cudaMemset(output, 0, sizeof(int));
    long long total = (long long)N * M;
    int threads = 256;
    long long required = (total + threads - 1) / threads;
    int blocks = required > 1024 ? 1024 : (int)required;
    count_2d_kernel<<<blocks, threads>>>(input, output, total, K);
    cudaDeviceSynchronize();
}
```
