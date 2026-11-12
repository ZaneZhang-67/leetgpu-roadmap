---
id: stream-compaction
leetgpuId: 72
titleEn: Stream Compaction
titleZh: 数据流压缩
description: 使用 CUDA 稳定地筛选正数，并把结果紧凑写入输出数组前部
problemUrl: https://leetgpu.com/challenges/stream-compaction
updatedAt: 2026-11-12
---
## 题目定义

给定长度为 `N` 的 FP32 数组 `A`，筛选所有严格大于零的元素，按原输入顺序紧凑写入 `out` 的前部。

### 示例

```text
输入： A   = [1.0, -2.0, 3.0, 0.0, -1.0, 4.0]
输出： out = [1.0,  3.0, 4.0, 0.0,  0.0, 0.0]
```

零不满足谓词。输出数组的长度仍为 `N`，接口不会单独返回有效长度。若正元素数量为 `k`，则 `out[0:k]` 保存筛选结果，`out[k:N]` 必须为零。约束为 `1≤N≤100,000,000`、`A[i]∈[-1000,1000]`。`out` 是预分配的 FP32 数组，长度为 `N`，运行时初始值为零；参考实现仍主动清零，保证代码可以独立提交。官方测试覆盖无正数、全正数、精确零和非二次幂长度。性能用例为 `N=50,000,000`，要求精确匹配。

## 模板代码解析

CUDA starter 的真实签名为：

```text
extern "C" void solve(const float* A, int N, float* out)
```

注意参数顺序是 `A,N,out`。稳定顺序是 reference 的布尔索引自然产生的语义；直接让线程对全局计数器 `atomicAdd` 会按执行时序写入，从而破坏输入顺序。

## 实现思路

参考实现先为每个元素生成 0/1 标志，再执行全局独占扫描：`positions[i]` 表示 `i` 之前有多少个正元素。若 `A[i]>0`，就把它写到 `out[positions[i]]`，因此顺序稳定且写入紧凑。

扫描使用经典的 Hillis-Steele 双缓冲，每轮偏移依次为 `1,2,4...`。每轮启动一个 kernel，以 kernel 边界提供全局同步；读取 `source`、写入 `destination` 后，再交换两个指针。扫描得到的是 inclusive 计数，因此散射位置为 `prefix[i]-1`。

`solve` 在散射前主动清零整个输出，未被正元素覆盖的尾部自然保持为零。同一默认流中的清零、扫描和散射会按提交顺序执行。该实现的时间复杂度为 `O(N log N)`，临时空间为两个长度为 N 的 INT32 数组，可以覆盖最大 N。这是保证稳定顺序和完整输出语义的正确性基线，并非面向最大性能用例的最优压缩扫描实现。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void make_flags(const float* A, int* flags, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) flags[i] = A[i] > 0.0f ? 1 : 0;
}

__global__ void scan_step(const int* source, int* destination, int N, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int value = source[i];
    if (i >= offset) value += source[i - offset];
    destination[i] = value;
}

__global__ void scatter_positive(const float* A, const int* prefix, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && A[i] > 0.0f) out[prefix[i] - 1] = A[i];
}

// A and out are device pointers
extern "C" void solve(const float* A, int N, float* out) {
    constexpr int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t bytes = static_cast<size_t>(N) * sizeof(int);
    int* first = nullptr;
    int* second = nullptr;
    cudaMalloc(&first, bytes);
    cudaMalloc(&second, bytes);
    cudaMemset(out, 0, static_cast<size_t>(N) * sizeof(float));

    make_flags<<<blocks, threads>>>(A, first, N);
    int* source = first;
    int* destination = second;
    for (int offset = 1; offset < N; offset <<= 1) {
        scan_step<<<blocks, threads>>>(source, destination, N, offset);
        int* temporary = source;
        source = destination;
        destination = temporary;
    }
    scatter_positive<<<blocks, threads>>>(A, source, out, N);
    cudaDeviceSynchronize();
    cudaFree(second);
    cudaFree(first);
}
```
