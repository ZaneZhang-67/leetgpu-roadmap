---
id: top-k-selection
leetgpuId: 29
titleEn: Top K Selection
titleZh: Top-K 选择
description: 使用 CUDA 分块维护候选集并逐层归并，保留重复值并按降序输出任意合法数量的最大元素
problemUrl: https://leetgpu.com/challenges/top-k-selection
updatedAt: 2026-11-12
---
## 题目定义

给定长度为 `N` 的 32 位浮点数组 `input`，选出最大的 `k` 个元素，将它们按降序写入长度为 `k` 的 `output`

重复值按出现次数保留若第 `k` 大的位置落在重复值中，输出需要包含足够多份该值，使结果长度恰好为 `k`

### 实现要求

- 不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写入 `output`

### 示例 1

```text
输入：input = [1.0, 5.0, 3.0, 2.0, 4.0]，N = 5，k = 3
输出：output = [5.0, 4.0, 3.0]
```

### 示例 2

```text
输入：input = [7.2, -1.0, 3.3, 8.8, 2.2]，N = 5，k = 2
输出：output = [8.8, 7.2]
```

### 约束

- `1 ≤ N ≤ 100,000,000`
- `1 ≤ k ≤ N`
- 输入元素均为 32 位浮点数
- reference 测试覆盖负数、全相等、`k = N`、大量重复最大值、跨步分布和集中在数组末尾的候选值
- 性能测试使用 `N = 50,000,000`、`k = 100`

## 模板代码解析

平台提供固定的 CUDA 入口：

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(
    const float* input,
    float* output,
    int N,
    int k
) {}
```

- `input` 是 GPU 上长度为 `N` 的只读数组
- `output` 是 GPU 上长度为 `k` 的输出数组
- `k` 可能等于 `N`

模板没有提供候选缓冲区、kernel 或启动配置

## 实现思路

每个分区只保留一个最大值会丢失同一分区中的其他优胜元素性能用例的 `k = 100` 适合使用分区候选集；公开约束还允许 `k > 100`，参考实现为这种情况保留完整降序归并排序路径

### 1. 构造小 K 分区候选集

当 `k ≤ 100` 时，将输入划分成固定大小的连续分区，每个 Block 负责一个分区Block 内由 0 号线程维护长度为 `k` 的降序列表，对分区中的所有元素执行插入

插入时使用严格比较：

```cpp
while (position > 0 && shared[position - 1] < value)
```

相等值不会被合并或去重，因此重复最大值会按份数保留

### 2. 两两归并候选列表

每个分区的前 `k` 已经降序排列下一层 kernel 将相邻两个列表像归并排序一样合并，但只写出前 `k` 个元素

若某层列表数为奇数，最后一个列表直接与空列表归并归并过程继续保留相等值，并保证降序

### 3. 逐层缩减

列表数量每轮约减半：

```text
numLists → ceil(numLists / 2) → ... → 1
```

最终唯一列表就是全局前 `k` 大结果这是因为任何分区中排名低于该分区前 `k` 的元素都不可能进入全局前 `k`

### 4. 覆盖任意合法 K

连续分区中的全部元素都会被扫描，因此优胜值无论集中在一个区域、位于数组末尾还是按较大步长分散都不会遗漏候选列表不去重，可处理第 `k` 大值重复、大量相同最大值和全相等输入

固定 Shared Memory 数组只服务于 `k ≤ 100` 的常见路径当 `k > 100` 时，参考实现对完整输入执行自底向上的降序归并排序，再复制前 `k` 项这一分支覆盖 `k = N` 等公开约束，不会越过固定候选数组边界

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

constexpr int kPartitionSize = 4096;
constexpr int kMaxK = 100;

__global__ void select_partition_topk(
    const float* input,
    float* candidates,
    int N,
    int k
) {
    __shared__ float top[kMaxK];

    if (threadIdx.x != 0) return;

    int begin = blockIdx.x * kPartitionSize;
    int end = begin + kPartitionSize;
    if (end > N) end = N;

    for (int i = 0; i < k; ++i) top[i] = -CUDART_INF_F;

    for (int i = begin; i < end; ++i) {
        float value = input[i];
        if (value < top[k - 1]) continue;

        int position = k - 1;
        while (position > 0 && top[position - 1] < value) {
            top[position] = top[position - 1];
            --position;
        }
        top[position] = value;
    }

    for (int i = 0; i < k; ++i) {
        candidates[blockIdx.x * k + i] = top[i];
    }
}

__global__ void merge_candidate_lists(
    const float* source,
    float* destination,
    int numLists,
    int k
) {
    if (threadIdx.x != 0) return;

    int pair = blockIdx.x;
    int leftList = pair * 2;
    if (leftList >= numLists) return;

    const float* left = source + leftList * k;
    const float* right =
        leftList + 1 < numLists ? source + (leftList + 1) * k : nullptr;
    float* result = destination + pair * k;

    int i = 0;
    int j = 0;
    for (int out = 0; out < k; ++out) {
        float leftValue = i < k ? left[i] : -CUDART_INF_F;
        float rightValue = right != nullptr && j < k
            ? right[j]
            : -CUDART_INF_F;

        if (leftValue >= rightValue) {
            result[out] = leftValue;
            ++i;
        } else {
            result[out] = rightValue;
            ++j;
        }
    }
}

__global__ void merge_full_lists(
    const float* source,
    float* destination,
    int N,
    int width
) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int left = pair * width * 2;
    if (left >= N) return;

    int middle = left + width;
    if (middle > N) middle = N;
    int right = left + width * 2;
    if (right > N) right = N;

    int i = left;
    int j = middle;
    int out = left;
    while (i < middle && j < right) {
        if (source[i] >= source[j]) {
            destination[out++] = source[i++];
        } else {
            destination[out++] = source[j++];
        }
    }
    while (i < middle) destination[out++] = source[i++];
    while (j < right) destination[out++] = source[j++];
}

void full_sort_topk(
    const float* input,
    float* output,
    int N,
    int k
) {
    size_t bytes = static_cast<size_t>(N) * sizeof(float);
    float* first = nullptr;
    float* second = nullptr;
    cudaMalloc(&first, bytes);
    cudaMalloc(&second, bytes);
    cudaMemcpy(first, input, bytes, cudaMemcpyDeviceToDevice);

    float* source = first;
    float* destination = second;
    constexpr int threads = 256;
    for (int width = 1; width < N; ) {
        int pairs = (N + 2 * width - 1) / (2 * width);
        int blocks = (pairs + threads - 1) / threads;
        merge_full_lists<<<blocks, threads>>>(source, destination, N, width);

        float* temporary = source;
        source = destination;
        destination = temporary;
        if (width > N / 2) break;
        width *= 2;
    }

    cudaMemcpy(
        output,
        source,
        static_cast<size_t>(k) * sizeof(float),
        cudaMemcpyDeviceToDevice
    );
    cudaDeviceSynchronize();
    cudaFree(first);
    cudaFree(second);
}

// input, output are device pointers
extern "C" void solve(
    const float* input,
    float* output,
    int N,
    int k
) {
    if (k > kMaxK) {
        full_sort_topk(input, output, N, k);
        return;
    }

    int numLists = (N + kPartitionSize - 1) / kPartitionSize;
    size_t bytes = static_cast<size_t>(numLists) * k * sizeof(float);

    float* first = nullptr;
    float* second = nullptr;
    cudaMalloc(&first, bytes);
    cudaMalloc(&second, bytes);

    select_partition_topk<<<numLists, 1>>>(
        input, first, N, k
    );

    float* source = first;
    float* destination = second;

    while (numLists > 1) {
        int nextLists = (numLists + 1) / 2;
        merge_candidate_lists<<<nextLists, 1>>>(
            source, destination, numLists, k
        );

        float* temporary = source;
        source = destination;
        destination = temporary;
        numLists = nextLists;
    }

    cudaMemcpy(
        output,
        source,
        static_cast<size_t>(k) * sizeof(float),
        cudaMemcpyDeviceToDevice
    );
    cudaDeviceSynchronize();

    cudaFree(first);
    cudaFree(second);
}
```
