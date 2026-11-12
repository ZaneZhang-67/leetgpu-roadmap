---
id: nearest-neighbor
leetgpuId: 38
titleEn: Nearest Neighbor
titleZh: 最近邻
description: 为每个三维点寻找距离最近的其他点，理解展平坐标、平方欧氏距离、自身排除与确定的平局规则
problemUrl: https://leetgpu.com/challenges/nearest-neighbor
updatedAt: 2026-11-12
---
## 题目定义

输入包含 `N` 个三维点，展平存储为长度 `N × 3` 的 `float` 数组 `points`对于每个点 `i`，找出距离最近且满足 `j != i` 的点，并将其下标写入 `indices[i]`

比较平方欧氏距离即可，不需要计算平方根：

```text
d²(i,j) = (xi-xj)² + (yi-yj)² + (zi-zj)²
```

### 实现要求与约束

- 不允许使用外部库
- `solve` 函数签名必须保持不变
- 结果写入 `int32` 数组 `indices`
- `1 ≤ N ≤ 100,000`
- 坐标是范围 `[-1000, 1000]` 内的 32 位浮点数
- 性能测试使用 `N = 10,000`

参考评测使用 `argmin`，距离相等时返回最小下标朴素实现按 `j` 从小到大扫描，并且只在 `distance < bestDistance` 时更新，因此自然遵循相同的平局规则

### 示例

```text
输入：points = [(0,0,0), (1,0,0), (5,5,5)]
输出：indices = [1, 0, 1]
```

点 0 与点 1 互为最近邻；点 2 到点 1 的距离小于到点 0 的距离

## 模板代码解析

CUDA starter 为：

```cpp
#include <cuda_runtime.h>

// points and indices are device pointers
extern "C" void solve(const float* points, int* indices, int N) {}
```

点 `i` 的三个坐标位于：

```text
points[3*i + 0]
points[3*i + 1]
points[3*i + 2]
```

`indices` 有 `N` 个元素一个线程可以独立处理一个查询点，因此 CPU 端按 `N` 启动一维 Grid

题面允许 `N = 1`，但此时不存在满足 `j != i` 的候选点平台参考流程把对角距离设为正无穷后执行 `argmin`，单元素输入会得到下标 `0`；参考实现沿用这一评测语义

## 实现思路

### 1. 一个线程负责一个点

线程 `i` 先读取点 `i` 的三个坐标，再扫描所有候选点 `j`

```cpp
if (j == i) continue;
```

这样排除点到自身的零距离

### 2. 比较平方距离

平方根是单调函数，比较 `d²` 与比较 `d` 得到的顺序相同，因此直接累加三个坐标差的平方，减少不必要的开方

初始化：

```cpp
float bestDistance = CUDART_INF_F;
int bestIndex = 0;
```

只有距离严格更小时才更新，这会保留最先遇到的候选，即平局时下标最小的点

### 3. 复杂度

每个点扫描 `N` 个候选，时间复杂度为 `O(N²)`，额外空间为 `O(1)`相邻线程会反复读取整份点集，可进一步使用 Shared Memory 分块加载候选点；不过无论如何优化，都应保持自身排除和平局规则不变

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void nearest_neighbor(const float* points, int* indices, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x = points[3 * i];
    float y = points[3 * i + 1];
    float z = points[3 * i + 2];
    float bestDistance = CUDART_INF_F;
    int bestIndex = 0;

    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        float dx = x - points[3 * j];
        float dy = y - points[3 * j + 1];
        float dz = z - points[3 * j + 2];
        float distance = dx * dx + dy * dy + dz * dz;
        if (distance < bestDistance) {
            bestDistance = distance;
            bestIndex = j;
        }
    }
    indices[i] = bestIndex;
}

// points and indices are device pointers
extern "C" void solve(const float* points, int* indices, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    nearest_neighbor<<<blocks, threads>>>(points, indices, N);
    cudaDeviceSynchronize();
}
```
