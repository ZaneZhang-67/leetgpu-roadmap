---
id: connected-components
leetgpuId: 79
titleEn: Connected Components
titleZh: 连通分量
description: 使用 CUDA 标签传播求无向图连通分量中的最小顶点编号
problemUrl: https://leetgpu.com/challenges/connected-components
updatedAt: 2026-11-12
---
## 题目定义

给定一个含 `N` 个顶点和 `M` 条边的无向图，其中边 `e` 由 INT32 数组中的 `(src[e],dst[e])` 表示。需要为每个顶点写出所属连通分量的标签，且标签必须等于该分量中的最小顶点编号。

### 示例

```text
N=5，M=3
src=[0,1,3]
dst=[1,2,4]
边：0-1、1-2、3-4
labels=[0,0,0,3,3]
```

约束为 `1≤N≤1,000,000`、`1≤M≤5,000,000`，顶点编号位于 `[0,N)`。图中允许自环和重复边，所有数组均为 INT32。性能用例为 `N=1,000,000`、`M=5,000,000`。

## 模板代码解析

CUDA starter 的签名为：

```text
extern "C" void solve(const int* src, const int* dst, int* labels, int N, int M)
```

`src` 和 `dst` 的长度均为 `M`，`labels` 的长度为 `N`。每条边都是无向边，因此传播时必须同时考虑 `src→dst` 和 `dst→src`。孤立点最终保留自己的编号。

## 实现思路

先令 `labels[v]=v`。每轮都让每条边用两个端点中较小的标签更新另一个端点，更新通过 `atomicMin` 完成；随后执行一次指针跳跃 `labels[v]=labels[labels[v]]`。设备端的 `changed` 标志记录本轮是否发生变化，host 端每轮取回该标志，直到收敛或达到 `N` 轮。

标签只会减小，最小编号会沿无向边传播。收敛后，每个分量中的所有顶点都得到该分量的最小编号。自环和重复边不会改变结果。该实现不必构建邻接表，额外空间只有一个设备标志。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void initialize_labels(int* labels, int N) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < N) labels[v] = v;
}

__global__ void propagate_edges(const int* src, const int* dst, int* labels, int* changed, int M) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= M) return;
    int u = src[e], v = dst[e];
    int lu = labels[u], lv = labels[v];
    if (atomicMin(&labels[u], lv) > lv) atomicExch(changed, 1);
    if (atomicMin(&labels[v], lu) > lu) atomicExch(changed, 1);
}

__global__ void compress_labels(int* labels, int* changed, int N) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    int old_label = labels[v];
    int new_label = labels[old_label];
    if (new_label < old_label) {
        labels[v] = new_label;
        atomicExch(changed, 1);
    }
}

// src, dst, labels are device pointers
extern "C" void solve(const int* src, const int* dst, int* labels, int N, int M) {
    constexpr int threads = 256;
    int vertex_blocks = (N + threads - 1) / threads;
    int edge_blocks = (M + threads - 1) / threads;
    int* changed = nullptr;
    cudaMalloc(&changed, sizeof(int));
    initialize_labels<<<vertex_blocks, threads>>>(labels, N);

    for (int iteration = 0; iteration < N; ++iteration) {
        cudaMemset(changed, 0, sizeof(int));
        propagate_edges<<<edge_blocks, threads>>>(src, dst, labels, changed, M);
        compress_labels<<<vertex_blocks, threads>>>(labels, changed, N);
        int host_changed = 0;
        cudaMemcpy(&host_changed, changed, sizeof(int), cudaMemcpyDeviceToHost);
        if (host_changed == 0) break;
    }
    cudaFree(changed);
}
```
