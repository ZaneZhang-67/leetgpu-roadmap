---
id: rainbow-table
leetgpuId: 24
titleEn: Rainbow Table
titleZh: 彩虹表迭代哈希
description: 使用 CUDA 对 32 位整数数组并行执行多轮 FNV-1a 哈希，理解元素级并行与轮次内串行依赖
problemUrl: https://leetgpu.com/challenges/rainbow-table
updatedAt: 2026-11-12
---
## 题目定义

使用题目提供的 FNV-1a 哈希函数，对一个 32 位整数数组并行执行 `R` 轮哈希同一元素的后一轮输入是前一轮输出：

```text
value₀ = input[i]
value₁ = fnv1a_hash(value₀)
value₂ = fnv1a_hash(value₁)
...
valueR = fnv1a_hash(valueR-1)
output[i] = valueR
```

数组中的不同元素互不依赖，可以并行处理；单个元素的 `R` 轮计算存在先后依赖

### 实现要求

- 不允许使用外部库
- `solve` 函数签名保持不变
- 最终结果写入数组 `output`

### 示例 1

```text
输入：numbers = [123, 456, 789]，R = 2
输出：hashes = [1636807824, 1273011621, 2193987222]
```

### 示例 2

```text
输入：numbers = [0, 1, 2147483647]，R = 3
输出：hashes = [96754810, 3571711400, 2006156166]
```

### 约束

- `1 ≤ N ≤ 10,000,000`
- `1 ≤ R ≤ 100`
- `0 ≤ input[i] ≤ 2147483647`
- 性能测试使用 `N = 5,000,000`

## 模板代码解析

模板已经提供设备端哈希函数、空 kernel 和 CPU 端入口：

```cpp
#include <cuda_runtime.h>

__device__ unsigned int fnv1a_hash(unsigned int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;

    unsigned int hash = OFFSET_BASIS;

    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFFu;
        hash = (hash ^ byte) * FNV_PRIME;
    }

    return hash;
}

__global__ void fnv1a_hash_kernel(
    const int* input,
    unsigned int* output,
    int N,
    int R
) {}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
    cudaDeviceSynchronize();
}
```

`fnv1a_hash` 已由平台实现它把一个 32 位无符号整数按从低到高的顺序拆成四个字节，每处理一个字节便先异或、再乘以 FNV 质数

`input` 的元素类型是 `int`，`output` 的元素类型是 `unsigned int`；`N` 是数组长度，`R` 是迭代轮数；CPU 端以 256 个线程组成一个 Block，并向上取整计算 Block 数需要补全的部分只有 `fnv1a_hash_kernel` 函数体，不应改动平台提供的哈希函数或 `solve` 签名

## 实现思路

### 1. 一个线程负责一个输入元素

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

线程 `i` 读取 `input[i]`，独立完成该元素的全部哈希轮次，最后写入 `output[i]`

### 2. 检查边界

Grid 向上取整后可能产生多余线程，因此先判断：

```cpp
if (i < N)
```

### 3. 处理有符号输入与无符号哈希值

题目约束保证输入非负，但哈希函数与输出使用 32 位无符号整数线程先转换并保存当前值：

```cpp
unsigned int value = static_cast<unsigned int>(input[i]);
```

数组以连续一维布局存储，相邻线程分别访问相邻元素

### 4. 串行执行 R 轮哈希

同一元素的下一轮依赖上一轮结果，因此在线程内部循环：

```cpp
for (int round = 0; round < R; ++round) {
    value = fnv1a_hash(value);
}
```

不同线程之间不共享 `value`，无需同步每个线程完成后写回：

```cpp
output[i] = value;
```

### 5. 32 位溢出语义

`fnv1a_hash` 使用 `unsigned int` 执行异或和乘法无符号 32 位运算会按模 `2³²` 保留低 32 位，这正是模板哈希函数的计算语义参考实现应继续使用 `unsigned int`，避免改变中间结果的位宽

## 参考实现

```cpp
#include <cuda_runtime.h>

__device__ unsigned int fnv1a_hash(unsigned int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;

    unsigned int hash = OFFSET_BASIS;

    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFFu;
        hash = (hash ^ byte) * FNV_PRIME;
    }

    return hash;
}

__global__ void fnv1a_hash_kernel(
    const int* input,
    unsigned int* output,
    int N,
    int R
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        unsigned int value = static_cast<unsigned int>(input[i]);

        for (int round = 0; round < R; ++round) {
            value = fnv1a_hash(value);
        }

        output[i] = value;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
    cudaDeviceSynchronize();
}
```
