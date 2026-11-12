---
id: count-array-element
leetgpuId: 43
titleEn: Count Array Element
titleZh: 统计一维数组元素
description: 使用 CUDA 归约统计一维 int32 数组中等于目标值 K 的元素数量
problemUrl: https://leetgpu.com/challenges/count-array-element
updatedAt: 2026-11-12
---
## 题目定义

给定长度为 `N` 的 32 位整数数组 `input` 和目标值 `K`，统计数组中等于 `K` 的元素数量，并把结果写入 `output[0]`

**实现要求**

- 只使用原生功能，不允许使用外部库
- `solve` 函数签名保持不变
- 最终结果写入 `output`

**示例**

```text
输入：input = [1, 2, 3, 4, 1]，K = 1
输出：2
```

**约束与测试**

- `1 ≤ N ≤ 100,000,000`
- 题面给出的 `input[i]` 与 `K` 范围为 `[1, 100,000]`
- `challenge.py` 还会使用未出现在输入取值范围内的 `K = 501010`，此时正确结果为 0
- 输出是形状为 `(1,)` 的 `int32` 张量
- 性能测试使用 `N = 100,000,000`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int K) {}
```

`input` 和 `output` 都位于 GPU 显存`input` 含 `N` 个 `int32`，`output` 只有一个元素模板没有提供 kernel、临时缓冲区或启动配置

## 实现思路

每个线程用 grid-stride 循环检查多个元素，在寄存器中累加局部计数；随后在 Block 内用共享内存归约，每个 Block 只对 `output[0]` 执行一次 `atomicAdd`这样既覆盖一亿个元素，也比每次匹配都执行全局原子加法更少争用

启动计数前必须把 `output[0]` 清零，因为官方测试用 `torch.empty(1)` 创建输出，初值未定义比较条件直接使用 `input[i] == K`，因此目标值不在输入范围内时自然得到 0

最大计数为一亿，未超过 32 位有符号整数上限

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void count_kernel(const int* input, int* output, int N, int K) {
    __shared__ int counts[256];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    int local = 0;

    for (int i = index; i < N; i += stride) {
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
extern "C" void solve(const int* input, int* output, int N, int K) {
    cudaMemset(output, 0, sizeof(int));
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    count_kernel<<<blocks, threads>>>(input, output, N, K);
    cudaDeviceSynchronize();
}
```
