---
id: reduction
leetgpuId: 4
titleEn: Reduction
titleZh: 并行归约求和
description: 使用 CUDA 对浮点数组执行并行归约，通过线程局部累加、Block 内共享内存归约和原子加法得到总和
problemUrl: https://leetgpu.com/challenges/reduction
updatedAt: 2026-11-12
---
## 题目定义

给定一个包含 `N` 个 32 位浮点数的数组 `input`，在 GPU 上计算所有元素之和，并把唯一结果写入 `output[0]`

```text
output[0] = input[0] + input[1] + ... + input[N - 1]
```

### 实现要求

- 只使用 GPU 原生功能，不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写入 `output`

### 示例 1

```text
输入：[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
输出：36.0
```

### 示例 2

```text
输入：[-2.5, 1.5, -1.0, 2.0]
输出：0.0
```

### 约束

- `1 ≤ N ≤ 100,000,000`
- `-1000.0 ≤ input[i] ≤ 1000.0`
- 最终总和始终可以用 32 位浮点数表示
- 性能测试使用 `N = 4,194,304`

## 模板代码解析

平台提供的 CUDA 模板只有头文件、设备指针说明和固定的 `solve` 签名：

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {}
```

- `input` 指向 GPU 上长度为 `N` 的只读数组
- `output` 指向 GPU 上保存最终标量结果的空间
- `N` 是输入元素数量
- 平台直接调用 `solve`，因此函数名、参数顺序和参数类型都需要保持原样

模板没有提供 kernel、启动配置或输出初始化，这些都需要在 `solve` 中完成

## 实现思路

直接让每个线程对同一个 `output[0]` 执行一次 `atomicAdd` 虽然容易实现，但会产生大量原子操作本题先让每个线程累加多个元素，再在 Block 内归约为一个部分和，每个 Block 最后只执行一次原子加法

### 1. 先把输出清零

不同 Block 最终通过 `atomicAdd` 合并结果，因此启动 kernel 前必须把 `output[0]` 初始化为零：

```cpp
cudaMemset(output, 0, sizeof(float));
```

若省略初始化，原子加法会累加到输出内存原有的值上

### 2. Grid-stride loop

线程的全局下标和整个 Grid 的线程总数分别为：

```cpp
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
```

每个线程按照 `stride` 跨步读取输入：

```cpp
float sum = 0.0f;
for (int i = index; i < N; i += stride) {
    sum += input[i];
}
```

这样即使输入规模大于一次启动的线程总数，所有元素仍会被覆盖每个线程只把自己的局部和写入共享内存，不会与其他线程冲突

### 3. Block 内共享内存归约

每个 Block 有 256 个线程，对应 256 个局部和：

```cpp
__shared__ float shared[256];
shared[threadIdx.x] = sum;
__syncthreads();
```

随后每轮把有效元素数量减半：

```cpp
for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
        shared[threadIdx.x] += shared[threadIdx.x + offset];
    }
    __syncthreads();
}
```

第一次把后 128 个值加到前 128 个位置，下一次处理 64 对，直到 `shared[0]` 保存整个 Block 的部分和同步用于保证本轮所有写入完成后，线程才进入下一轮

### 4. 合并各 Block 的部分和

每个 Block 的 0 号线程执行一次：

```cpp
if (threadIdx.x == 0) {
    atomicAdd(output, shared[0]);
}
```

原子加法保证多个 Block 同时更新 `output[0]` 时不会丢失结果与“每个输入元素一次原子加法”相比，原子操作数量降低为 Block 数量

### 5. 启动配置

使用 256 个线程组成一个 Block，并限制最多启动 1024 个 Block：

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
if (blocksPerGrid > 1024) blocksPerGrid = 1024;
```

Block 数受限后，Grid-stride loop 继续覆盖剩余输入，不影响正确性

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void reduction_kernel(const float* input, float* output, int N) {
    __shared__ float shared[256];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = index; i < N; i += stride) {
        sum += input[i];
    }

    shared[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    constexpr int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 1024) blocksPerGrid = 1024;

    cudaMemset(output, 0, sizeof(float));
    reduction_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```
