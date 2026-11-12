---
id: subarray-sum
leetgpuId: 47
titleEn: Subarray Sum
titleZh: 一维子数组求和
description: 使用 CUDA 对闭区间内的 int32 元素执行并行归约，介绍区间映射、Grid-stride 遍历和 Block 部分和合并
problemUrl: https://leetgpu.com/challenges/subarray-sum
updatedAt: 2026-11-12
---
## 题目定义

给定长度为 `N` 的 32 位整数数组 `input`，以及从零开始的下标 `S` 和 `E`，计算闭区间 `input[S..E]` 的元素总和，并写入 `output[0]`

```text
output[0] = Σ input[i]，其中 S ≤ i ≤ E
```

### 实现要求

- 只使用原生功能，不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写入 `output`

### 示例 1

```text
输入：input = [1, 2, 1, 3, 4]，S = 1，E = 3
输出：output = 6
```

### 示例 2

```text
输入：input = [1, 2, 3, 4]，S = 0，E = 3
输出：output = 10
```

### 约束

- `1 ≤ N ≤ 100,000,000`
- `1 ≤ input[i] ≤ 10`
- `0 ≤ S ≤ E ≤ N - 1`
- 性能测试使用 `N = 100,000,000`、`S = 17,651`、`E = 98,765,431`

## 模板代码解析

平台提供固定的 CUDA 入口：

```cpp
#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(
    const int* input,
    int* output,
    int N,
    int S,
    int E
) {}
```

- `input` 是 GPU 上长度为 `N` 的 `int32` 数组
- `output` 是 GPU 上长度为 1 的 `int32` 数组
- `S` 和 `E` 都包含在求和范围内
- `N` 保留完整输入长度，实际参与计算的元素数量是 `E - S + 1`

模板没有提供 kernel、输出初始化或启动配置

## 实现思路

让每个线程先累加多个区间元素，再在 Block 内归约为一个部分和每个 Block 最后只对输出执行一次原子加法

### 1. 将局部下标映射到原数组

区间长度为：

```cpp
int count = E - S + 1;
```

线程使用局部下标 `offset` 遍历 `[0, count)`，实际地址为 `input[S + offset]`这样边界条件只需判断 `offset < count`

### 2. Grid-stride 遍历

线程按整个 Grid 的线程数跨步读取：

```cpp
for (int offset = globalThread; offset < count; offset += gridStride) {
    local += input[S + offset];
}
```

即使区间包含近一亿个元素，也能由固定数量的 Block 完整覆盖

### 3. Block 内归约

256 个线程将局部和写入共享内存，随后每轮把有效数量减半同步确保本轮写入完成后才进入下一轮

### 4. 合并部分和

启动 kernel 前用 `cudaMemset` 将 `output[0]` 清零每个 Block 的 0 号线程对 `output` 执行一次 `atomicAdd`

reference 的输入和输出类型均为 `int32`，参考实现保持相同的整数加法与输出语义

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void subarray_sum_kernel(
    const int* input,
    int* output,
    int S,
    int count
) {
    __shared__ int shared[256];

    int tid = threadIdx.x;
    int offset = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    int local = 0;

    for (int i = offset; i < count; i += stride) {
        local += input[S + i];
    }

    shared[tid] = local;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            shared[tid] += shared[tid + step];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(
    const int* input,
    int* output,
    int N,
    int S,
    int E
) {
    (void)N;
    constexpr int threadsPerBlock = 256;
    int count = E - S + 1;
    int blocksPerGrid =
        (count + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 1024) blocksPerGrid = 1024;

    cudaMemset(output, 0, sizeof(int));
    subarray_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, S, count
    );
    cudaDeviceSynchronize();
}
```
