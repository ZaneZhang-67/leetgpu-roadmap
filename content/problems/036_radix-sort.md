---
id: radix-sort
leetgpuId: 36
titleEn: Radix Sort
titleZh: 基数排序
description: 使用 CUDA 对完整 uint32 范围执行稳定的逐位基数排序，通过零位计数、前缀扫描和稳定散射生成升序结果
problemUrl: https://leetgpu.com/challenges/radix-sort
updatedAt: 2026-11-12
---
## 题目定义

给定长度为 `N` 的 32 位无符号整数数组 `input`，使用基数排序按数值升序排列，并将结果写入 `output`

### 实现要求

- 不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写入 `output`
- 必须使用基数排序
- 按升序排列

### 示例 1

```text
输入：[170, 45, 75, 90, 2, 802, 24, 66]
输出：[2, 24, 45, 66, 75, 90, 170, 802]
```

### 示例 2

```text
输入：[1, 4, 1, 3, 555, 1000, 2]
输出：[1, 1, 2, 3, 4, 555, 1000]
```

### 约束

- `1 ≤ N ≤ 100,000,000`
- `0 ≤ input[i] ≤ 4,294,967,295`
- 输入和输出元素类型均为 `uint32`
- 功能测试包含重复值、单元素、已排序、逆序及 `UINT32_MAX`
- 性能测试使用 `N = 50,000,000`

题目和 reference 均按无符号整数排序，不涉及浮点数、负数或 NaN 语义最高位与其余位相同，都是数值位

## 模板代码解析

平台提供固定的 CUDA 入口：

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(
    const unsigned int* input,
    unsigned int* output,
    int N
) {}
```

- `input` 是 GPU 上长度为 `N` 的 `uint32` 数组
- `output` 是同长度的排序结果
- 输入与输出分离

模板没有提供扫描缓冲区、kernel 或启动配置

## 实现思路

采用最低有效位优先的二进制基数排序，从第 0 位到第 31 位执行 32 轮稳定分区每轮把当前位为 0 的元素放在前面，当前位为 1 的元素放在后面

### 1. 生成零位标志

对于当前 `bit`：

```cpp
zeroFlag[i] = ((source[i] >> bit) & 1u) == 0u ? 1u : 0u;
```

随后对 `zeroFlag` 计算包含式前缀和 `zeroPrefix`

### 2. 计算稳定散射位置

若当前位为 0：

```text
position = zeroPrefix[i] - 1
```

若当前位为 1，`zeroPrefix[i]` 已包含当前位置的零标志，因此它前面的一位元素数量为：

```text
i - zeroPrefix[i]
```

因此位置为：

```text
position = totalZeros + (i - zeroPrefix[i])
```

相同位值的元素保持原相对次序，这正是 LSD 基数排序逐轮正确的条件重复值也会完整保留

### 3. 分层扫描标志数组

前缀扫描使用 Block 内 Hillis-Steele 扫描、递归扫描 Block 总和、偏移回填的结构，可覆盖最多一亿个元素，不要求 `N` 是 2 的幂或 Block 大小的倍数

### 4. 轮间交换缓冲区

每轮稳定散射后交换源与目标32 轮是偶数，最终结果回到初始工作缓冲区，再复制到 `output`

输入先复制到内部工作区，保证 `input` 始终只读，并让算法无论平台是否复用内存都遵守 starter 语义

## 参考实现

```cpp
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;

__global__ void make_zero_flags(
    const unsigned int* source,
    unsigned int* flags,
    int N,
    int bit
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        flags[index] = ((source[index] >> bit) & 1u) == 0u ? 1u : 0u;
    }
}

__global__ void scan_blocks_u32(
    const unsigned int* input,
    unsigned int* output,
    unsigned int* blockSums,
    int N
) {
    __shared__ unsigned int shared[kBlockSize];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int blockStart = blockIdx.x * blockDim.x;
    shared[tid] = index < N ? input[index] : 0u;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        unsigned int value = shared[tid];
        unsigned int addend = tid >= offset ? shared[tid - offset] : 0u;
        __syncthreads();
        shared[tid] = value + addend;
        __syncthreads();
    }

    if (index < N) output[index] = shared[tid];

    if (blockSums != nullptr && tid == 0) {
        int valid = N - blockStart;
        if (valid > blockDim.x) valid = blockDim.x;
        blockSums[blockIdx.x] = shared[valid - 1];
    }
}

__global__ void add_scan_offsets_u32(
    unsigned int* output,
    const unsigned int* scannedBlockSums,
    int N
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N && blockIdx.x > 0) {
        output[index] += scannedBlockSums[blockIdx.x - 1];
    }
}

void scan_u32(
    const unsigned int* input,
    unsigned int* output,
    int N
) {
    int blocks = (N + kBlockSize - 1) / kBlockSize;
    if (blocks == 1) {
        scan_blocks_u32<<<1, kBlockSize>>>(input, output, nullptr, N);
        return;
    }

    unsigned int* blockSums = nullptr;
    unsigned int* scannedBlockSums = nullptr;
    cudaMalloc(&blockSums, static_cast<size_t>(blocks) * sizeof(unsigned int));
    cudaMalloc(&scannedBlockSums, static_cast<size_t>(blocks) * sizeof(unsigned int));

    scan_blocks_u32<<<blocks, kBlockSize>>>(input, output, blockSums, N);
    scan_u32(blockSums, scannedBlockSums, blocks);
    add_scan_offsets_u32<<<blocks, kBlockSize>>>(
        output, scannedBlockSums, N
    );

    cudaFree(blockSums);
    cudaFree(scannedBlockSums);
}

__global__ void stable_scatter(
    const unsigned int* source,
    unsigned int* destination,
    const unsigned int* zeroPrefix,
    int N,
    int bit
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) return;

    unsigned int totalZeros = zeroPrefix[N - 1];
    bool isZero = ((source[index] >> bit) & 1u) == 0u;
    unsigned int position = isZero
        ? zeroPrefix[index] - 1u
        : totalZeros + static_cast<unsigned int>(index) - zeroPrefix[index];

    destination[position] = source[index];
}

// input, output are device pointers
extern "C" void solve(
    const unsigned int* input,
    unsigned int* output,
    int N
) {
    size_t bytes = static_cast<size_t>(N) * sizeof(unsigned int);
    unsigned int* first = nullptr;
    unsigned int* second = nullptr;
    unsigned int* flags = nullptr;
    unsigned int* prefix = nullptr;

    cudaMalloc(&first, bytes);
    cudaMalloc(&second, bytes);
    cudaMalloc(&flags, bytes);
    cudaMalloc(&prefix, bytes);
    cudaMemcpy(first, input, bytes, cudaMemcpyDeviceToDevice);

    int blocks = (N + kBlockSize - 1) / kBlockSize;
    unsigned int* source = first;
    unsigned int* destination = second;

    for (int bit = 0; bit < 32; ++bit) {
        make_zero_flags<<<blocks, kBlockSize>>>(source, flags, N, bit);
        scan_u32(flags, prefix, N);
        stable_scatter<<<blocks, kBlockSize>>>(
            source, destination, prefix, N, bit
        );

        unsigned int* temporary = source;
        source = destination;
        destination = temporary;
    }

    cudaMemcpy(output, source, bytes, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    cudaFree(first);
    cudaFree(second);
    cudaFree(flags);
    cudaFree(prefix);
}
```
