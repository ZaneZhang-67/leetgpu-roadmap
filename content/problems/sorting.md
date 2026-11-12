---
id: sorting
leetgpuId: 15
titleEn: Sorting
titleZh: 并行排序
description: 使用 CUDA 自底向上归并排序处理最多一百万个浮点数，通过并行归并和缓冲区交换得到升序结果
problemUrl: https://leetgpu.com/challenges/sorting
updatedAt: 2026-11-12
---
## 题目定义

给定一个包含 `N` 个 32 位浮点数的数组 `data`，在 GPU 上将它按升序排列，并把结果写回原数组

### 实现要求

- 只使用原生功能，不允许使用外部库
- `solve` 函数签名必须保持不变
- 排序结果必须写回输入数组 `data`

### 示例

```text
输入：data = [5.0, 2.0, 8.0, 1.0, 9.0, 4.0]，N = 6
输出：data = [1.0, 2.0, 4.0, 5.0, 8.0, 9.0]
```

### 约束

- `1 ≤ N ≤ 1,000,000`
- 性能测试使用 `N = 1,000,000`

## 模板代码解析

平台提供固定的 CUDA 入口：

```cpp
#include <cuda_runtime.h>

// data is device pointer
extern "C" void solve(float* data, int N) {}
```

- `data` 是 GPU 上长度为 `N` 的可写数组
- `N` 是元素数量
- 排序必须原地反映到 `data`，但实现过程可以申请临时缓冲区

模板没有指定排序算法，也没有提供 kernel 或启动配置

## 实现思路

参考实现采用自底向上的归并排序它不依赖输入规模是 2 的幂，能覆盖 `N = 1,000,000` 的完整约束

### 1. 从长度为 1 的有序段开始

单个元素天然有序第一轮把相邻的两个长度为 1 的段合并，之后段宽依次变为：

```text
1, 2, 4, 8, ...
```

当段宽 `width` 不小于 `N` 时，整个数组已经有序

### 2. 一个线程归并一对有序段

第 `pair` 个线程负责区间：

```text
left  = pair × 2 × width
mid   = min(left + width, N)
right = min(left + 2 × width, N)
```

两个输入段分别为 `[left, mid)` 和 `[mid, right)`线程使用两个游标选择较小元素并写入目标缓冲区

数组尾部不足两个完整段时，`min` 会把边界截断，算法仍然正确

### 3. 每轮交换源与目标

归并不能安全地在同一数组内并行覆盖源数据，因此 `solve` 申请一个与输入等长的临时缓冲区：

```cpp
cudaMalloc(&buffer, N * sizeof(float));
```

每轮结束后交换 `source` 与 `destination`，下一轮从刚生成的有序段继续归并kernel 启动之间天然形成顺序，另外调用 `cudaDeviceSynchronize()` 及时检查并完成该轮

### 4. 将最终结果写回 `data`

归并轮数可能是奇数，也可能是偶数若最终有序数据位于临时缓冲区，需要复制回平台要求的 `data`：

```cpp
cudaMemcpy(data, source, N * sizeof(float), cudaMemcpyDeviceToDevice);
```

随后释放临时空间

### 5. 复杂度

总共进行 `ceil(log₂N)` 轮，每轮读取并写入 `N` 个元素：

```text
时间复杂度：O(N log N)
额外空间：O(N)
```

这比单线程插入排序或冒泡排序适合百万级输入，并且对任意 `N` 都能得到正确升序结果

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void merge_pass(
    const float* source,
    float* destination,
    int N,
    int width
) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int left = pair * width * 2;

    if (left >= N) return;

    int mid = left + width;
    if (mid > N) mid = N;

    int right = left + width * 2;
    if (right > N) right = N;

    int i = left;
    int j = mid;
    int out = left;

    while (i < mid && j < right) {
        if (source[i] <= source[j]) {
            destination[out++] = source[i++];
        } else {
            destination[out++] = source[j++];
        }
    }

    while (i < mid) destination[out++] = source[i++];
    while (j < right) destination[out++] = source[j++];
}

// data is device pointer
extern "C" void solve(float* data, int N) {
    if (N <= 1) return;

    float* buffer = nullptr;
    cudaMalloc(&buffer, static_cast<size_t>(N) * sizeof(float));

    float* source = data;
    float* destination = buffer;
    constexpr int threadsPerBlock = 256;

    for (int width = 1; width < N; ) {
        int pairs = (N + 2 * width - 1) / (2 * width);
        int blocksPerGrid =
            (pairs + threadsPerBlock - 1) / threadsPerBlock;

        merge_pass<<<blocksPerGrid, threadsPerBlock>>>(
            source, destination, N, width
        );
        cudaDeviceSynchronize();

        float* temp = source;
        source = destination;
        destination = temp;

        if (width > N / 2) break;
        width *= 2;
    }

    if (source != data) {
        cudaMemcpy(
            data,
            source,
            static_cast<size_t>(N) * sizeof(float),
            cudaMemcpyDeviceToDevice
        );
    }

    cudaFree(buffer);
}
```
