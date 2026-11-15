---
id: segmented-exclusive-prefix-sum
leetgpuId: 70
titleEn: Segmented Exclusive Prefix Sum
titleZh: 分段独占前缀和
description: 使用 CUDA 按段起始标志并行计算 FP64 累计、FP32 输出的独占前缀和
problemUrl: https://leetgpu.com/challenges/segmented-exclusive-prefix-sum
updatedAt: 2026-11-12
---
## 题目定义

给定 FP32 数组 `values` 和 INT32 数组 `flags`。`flags[i]==1` 表示下标 `i` 是新分段的起点，且官方输入保证 `flags[0]==1`。输出为每段内部的独占前缀和：

```text
output[i] = 当前分段中位于 i 之前的 values 之和
```

所以每个分段首元素的输出都是 0。

### 示例

```text
values:   [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
flags:    [  1,   0,   0,   1,   0,   1]
segments: [1.0, 2.0, 3.0] | [4.0, 5.0] | [6.0]
output:   [0.0, 1.0, 3.0,   0.0, 4.0,   0.0]
```

约束为 `1≤N≤100,000,000`、`flags[0]=1`、`flags[i]∈{0,1}`，且 `values[i]∈[-100,100]`。官方性能用例 `N=50,000,000`，每 256 个元素开始新段；reference 使用 FP64 计算全局前缀后转回 FP32，判定容差为 `1e-3`。

## 模板代码解析

CUDA starter 的签名为：

```text
extern "C" void solve(const float* values, const int* flags, float* output, int N)
```

三者均为设备指针。`flags` 不是段编号，而是起点标志。独占语义要求当前元素不计入自己的输出；分段起点的输出必须为零。

## 实现思路

使用全局 Hillis-Steele 分段扫描。初始化时，为每个位置构造一个二元状态 `(sum,head)`：

```text
head[i] = flags[i] != 0
sum[i]  = 0                              i==0 或 head[i]
          values[i-1]                    其他位置
```

这相当于把独占前缀需要的贡献向右移动一位，同时在每个分段起点清零。

每轮偏移为 `1,2,4,...`。当前位置状态为 `cur`，左侧状态为 `left`：

```text
newSum  = curHead ? curSum : leftSum + curSum
newHead = leftHead | curHead
```

`curHead` 表示从左侧位置到当前位置之间已经出现分段起点，此时不能把左侧和跨过边界加入。两个 `double` 缓冲区和两个 `unsigned char` 缓冲区交替作为每轮的输入与输出。每轮启动一个独立 kernel，由 kernel 边界提供全局同步，因此跨越任意 Block 的分段也能正确计算。

算法执行 `ceil(log2 N)` 轮，时间复杂度为 `O(N log N)`，并行深度为 `O(log N)`。临时空间包括两份 FP64 数组和两份字节标志数组，共约 `18N` 字节。这是保证全局分段语义与 FP64 累计精度的正确性基线，并非工作量最优的扫描实现。FP64 累加与官方 reference 一致，最终统一转换为 FP32。`N=1`、连续分段起点、单一长分段和跨 Block 分段都使用同一套逻辑。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void initialize_segmented_scan(const float* values, const int* flags,
                                          double* sums, unsigned char* heads, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    unsigned char head = flags[i] != 0 ? 1 : 0;
    heads[i] = head;
    sums[i] = (i == 0 || head != 0) ? 0.0 : static_cast<double>(values[i - 1]);
}

__global__ void segmented_scan_step(const double* source_sums,
                                    const unsigned char* source_heads,
                                    double* destination_sums,
                                    unsigned char* destination_heads,
                                    int N, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double current_sum = source_sums[i];
    unsigned char current_head = source_heads[i];
    if (i >= offset) {
        double left_sum = source_sums[i - offset];
        unsigned char left_head = source_heads[i - offset];
        destination_sums[i] = current_head != 0 ? current_sum : left_sum + current_sum;
        destination_heads[i] = static_cast<unsigned char>(left_head | current_head);
    } else {
        destination_sums[i] = current_sum;
        destination_heads[i] = current_head;
    }
}

__global__ void store_segmented_scan(const double* sums, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = static_cast<float>(sums[i]);
}

// values, flags, output are device pointers
extern "C" void solve(const float* values, const int* flags, float* output, int N) {
    constexpr int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t sum_bytes = static_cast<size_t>(N) * sizeof(double);
    size_t head_bytes = static_cast<size_t>(N) * sizeof(unsigned char);

    double* sums_a = nullptr;
    double* sums_b = nullptr;
    unsigned char* heads_a = nullptr;
    unsigned char* heads_b = nullptr;
    cudaMalloc(&sums_a, sum_bytes);
    cudaMalloc(&sums_b, sum_bytes);
    cudaMalloc(&heads_a, head_bytes);
    cudaMalloc(&heads_b, head_bytes);

    initialize_segmented_scan<<<blocks, threads>>>(values, flags, sums_a, heads_a, N);
    double* source_sums = sums_a;
    double* destination_sums = sums_b;
    unsigned char* source_heads = heads_a;
    unsigned char* destination_heads = heads_b;

    for (int offset = 1; offset < N; offset <<= 1) {
        segmented_scan_step<<<blocks, threads>>>(source_sums, source_heads,
                                                  destination_sums, destination_heads,
                                                  N, offset);
        double* temporary_sums = source_sums;
        source_sums = destination_sums;
        destination_sums = temporary_sums;
        unsigned char* temporary_heads = source_heads;
        source_heads = destination_heads;
        destination_heads = temporary_heads;
    }

    store_segmented_scan<<<blocks, threads>>>(source_sums, output, N);
    cudaDeviceSynchronize();
    cudaFree(heads_b);
    cudaFree(heads_a);
    cudaFree(sums_b);
    cudaFree(sums_a);
}
```
