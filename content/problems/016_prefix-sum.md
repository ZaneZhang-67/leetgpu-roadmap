---
id: prefix-sum
leetgpuId: 16
titleEn: Prefix Sum
titleZh: 前缀和
description: 使用 CUDA 分层扫描计算包含式前缀和，通过 Block 内 Hillis-Steele 扫描、Block 总和递归扫描和偏移回填覆盖大规模输入
problemUrl: https://leetgpu.com/challenges/prefix-sum
updatedAt: 2026-11-12
---
## 题目定义

给定长度为 `N` 的 32 位浮点数组 `input`，计算包含式前缀和并写入 `output`

```text
input  = [a, b, c, d, ...]
output = [a, a+b, a+b+c, a+b+c+d, ...]
```

即：

```text
output[i] = Σ input[j]，其中 0 ≤ j ≤ i
```

### 实现要求

- 只使用 GPU 原生功能，不允许使用外部库
- `solve` 函数签名必须保持不变
- 结果必须写入 `output`

### 示例 1

```text
输入：[1.0, 2.0, 3.0, 4.0]
输出：[1.0, 3.0, 6.0, 10.0]
```

### 示例 2

```text
输入：[5.0, -2.0, 3.0, 1.0, -4.0]
输出：[5.0, 3.0, 6.0, 7.0, 3.0]
```

### 约束

- `1 ≤ N ≤ 100,000,000`
- `-1000.0 ≤ input[i] ≤ 1000.0`
- 输出数组中的最大值可以用 32 位浮点数表示
- 性能测试使用 `N = 250,000`

## 模板代码解析

平台提供固定的 CUDA 入口：

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {}
```

- `input` 是 GPU 上长度为 `N` 的只读数组
- `output` 是 GPU 上长度为 `N` 的输出数组
- `N` 是元素数量

模板没有提供 kernel、临时缓冲区或启动配置，需要在 `solve` 中组织完整的分层扫描

## 实现思路

单个 Block 无法覆盖最多一亿个元素参考实现把数组分块扫描，再扫描各 Block 的总和，最后把前面所有 Block 的累计和加回当前 Block

### 1. Block 内包含式扫描

每个 Block 处理 256 个元素线程先把输入载入共享内存，没有对应输入的线程写入 0：

```cpp
shared[tid] = index < N ? input[index] : 0.0f;
```

随后使用 Hillis-Steele 形式逐步扩大依赖距离：

```text
offset = 1, 2, 4, 8, ...
```

每轮中，线程 `tid` 先把自身旧值和 `shared[tid - offset]` 都读入寄存器；所有线程完成读取后再写回新值由于读写同一共享数组，读取与写回之间、相邻轮次之间都需要同步

扫描完成后，合法线程把结果写入 `output`，Block 最后一个有效元素同时写入 `blockSums[blockIdx.x]`

### 2. 递归扫描 Block 总和

若共有多个 Block，仅完成局部扫描还不够例如第二个 Block 的每个结果都需要加上第一个 Block 的总和

因此，对 `blockSums` 使用同一个分层扫描过程：

```text
blockSums → scannedBlockSums
```

当 Block 总和数组仍超过 256 个元素时继续递归，直到顶层只需一个 Block

递归深度是 `O(log₂₅₆N)`，即使 `N = 100,000,000` 也只有少量层级

### 3. 回填前置 Block 的偏移

对原数组第 `block` 个 Block：

```text
block = 0：偏移为 0
block > 0：偏移为 scannedBlockSums[block - 1]
```

每个线程把偏移加到自己的局部前缀和值上：

```cpp
output[index] += scannedBlockSums[blockIdx.x - 1];
```

这样得到全局包含式前缀和

### 4. 边界处理

最后一个 Block 可能不足 256 个元素无效线程以 0 参与扫描，Block 总和取实际最后一个元素：

```cpp
int validCount = min(256, N - blockStart);
blockSums[blockIdx.x] = shared[validCount - 1];
```

因此任意正整数 `N` 都能正确处理，不要求数组长度是 2 的幂或 Block 大小的倍数

## 参考实现

```cpp
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;

__global__ void scan_blocks(
    const float* input,
    float* output,
    float* blockSums,
    int N
) {
    __shared__ float shared[kBlockSize];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int blockStart = blockIdx.x * blockDim.x;

    shared[tid] = index < N ? input[index] : 0.0f;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        float value = shared[tid];
        float addend = tid >= offset ? shared[tid - offset] : 0.0f;
        __syncthreads();
        shared[tid] = value + addend;
        __syncthreads();
    }

    if (index < N) output[index] = shared[tid];

    if (blockSums != nullptr && tid == 0) {
        int validCount = N - blockStart;
        if (validCount > blockDim.x) validCount = blockDim.x;
        blockSums[blockIdx.x] = shared[validCount - 1];
    }
}

__global__ void add_block_offsets(
    float* output,
    const float* scannedBlockSums,
    int N
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N && blockIdx.x > 0) {
        output[index] += scannedBlockSums[blockIdx.x - 1];
    }
}

void scan_recursive(const float* input, float* output, int N) {
    int blocks = (N + kBlockSize - 1) / kBlockSize;

    if (blocks == 1) {
        scan_blocks<<<1, kBlockSize>>>(input, output, nullptr, N);
        return;
    }

    float* blockSums = nullptr;
    float* scannedBlockSums = nullptr;
    cudaMalloc(&blockSums, static_cast<size_t>(blocks) * sizeof(float));
    cudaMalloc(&scannedBlockSums, static_cast<size_t>(blocks) * sizeof(float));

    scan_blocks<<<blocks, kBlockSize>>>(
        input, output, blockSums, N
    );
    scan_recursive(blockSums, scannedBlockSums, blocks);
    add_block_offsets<<<blocks, kBlockSize>>>(
        output, scannedBlockSums, N
    );

    cudaFree(blockSums);
    cudaFree(scannedBlockSums);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    scan_recursive(input, output, N);
    cudaDeviceSynchronize();
}
```
