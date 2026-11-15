---
id: all-pairs-shortest-paths
leetgpuId: 73
titleEn: All-Pairs Shortest Paths
titleZh: 全源最短路径
description: 使用 Floyd-Warshall 在 GPU 上计算有向带权图的全源最短路径，正确处理阶段同步与无穷大
problemUrl: https://leetgpu.com/challenges/all-pairs-shortest-paths
updatedAt: 2026-11-12
---
## 题目定义

输入 `dist` 是按行主序展平的 `N × N` 距离矩阵，`dist[i*N+j]` 表示从顶点 `i` 到顶点 `j` 的直接边权。不可达位置为正无穷，主对角线为 0。要求把任意两点之间的最短距离写入 `output`

运行时 reference 使用 Floyd-Warshall：

```text
for k in [0,N):
    d[i,j] = min(d[i,j], d[i,k] + d[k,j])
```

### 官方示例

```text
输入：N = 4
      dist = [
        0,   5, inf,  10,
        inf, 0,   3, inf,
        inf, inf, 0,   1,
        inf, inf, inf, 0
      ]
输出：output = [
        0,   5,   8,   9,
        inf, 0,   3,   4,
        inf, inf, 0,   1,
        inf, inf, inf, 0
      ]
```

其中 `0→2` 经由顶点 1，距离为 8；`0→3` 经由 `1→2`，距离为 9；`1→3` 经由顶点 2，距离为 4

### 完整约束

- `1 ≤ N ≤ 4096`
- 边权是有限 `float32` 或表示无直接边的正无穷
- 输入图没有负环
- 对所有 `i`，均有 `dist[i*N+i] = 0`
- `dist` 与 `output` 都是长度 `N*N` 的 FP32 行主序数组
- 性能测试使用 `N=2048`
- 运行时评测还包括 `N=1`、断连图、零权边、无负环的负权边和非二次幂规模
- 判定容差为 `atol=0.01, rtol=0.01`

## 模板代码解析

CUDA starter 的真实入口为：

```cpp
#include <cuda_runtime.h>

// dist, output are device pointers
extern "C" void solve(const float* dist, float* output, int N) {}
```

`dist` 是只读设备指针，`output` 是长度为 `N*N` 的设备输出。题目没有传入额外缓冲区，因此可以先把输入复制到输出，再原地执行 Floyd-Warshall

每个 `k` 都依赖上一个阶段更新后的完整矩阵。Block 内同步不足以覆盖整个 Grid，阶段之间必须有全局同步。最直接的做法是每个 `k` 启动一次 kernel；默认 Stream 中的 kernel 按提交顺序执行

## 实现思路

一个二维线程负责一个 `(i,j)`，读取当前阶段的 `output[i,k]` 与 `output[k,j]`，再更新 `output[i,j]`

由于输入图主对角线为 0 且不存在负环，同一 `k` 阶段原地更新第 `k` 行和第 `k` 列不会改变它们：

```text
d[i,k] = min(d[i,k], d[i,k] + d[k,k]) = d[i,k]
d[k,j] = min(d[k,j], d[k,k] + d[k,j]) = d[k,j]
```

因此 kernel 可直接跳过 `row==k` 或 `col==k` 的元素，既保持 reference 结果，也避免同一阶段对枢轴行列同时读写

正无穷满足 `inf + x = inf`，使用普通浮点加法与 `fminf` 会自然保留不可达状态，无需把 INF 替换成有限哨兵。负权边也按同一公式处理

时间复杂度为 `O(N³)`，额外设备空间为 `O(1)`；输出矩阵本身占 `O(N²)`

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void floyd_stage(float* distances, int N, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N || row == k || col == k) return;

    int index = row * N + col;
    float through_k = distances[row * N + k] + distances[k * N + col];
    distances[index] = fminf(distances[index], through_k);
}

// dist, output are device pointers
extern "C" void solve(const float* dist, float* output, int N) {
    size_t bytes = static_cast<size_t>(N) * N * sizeof(float);
    cudaMemcpy(output, dist, bytes, cudaMemcpyDeviceToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);
    for (int k = 0; k < N; ++k) {
        floyd_stage<<<blocks, threads>>>(output, N, k);
    }
    cudaDeviceSynchronize();
}
```
