---
id: bfs-shortest-path
leetgpuId: 46
titleEn: BFS Shortest Path
titleZh: BFS 最短路径
description: 使用 CUDA 在含障碍物的二维网格中逐层执行 BFS，求四方向移动的最短步数
problemUrl: https://leetgpu.com/challenges/bfs-shortest-path
updatedAt: 2026-11-12
---
## 题目定义

给定展平的二维网格、起点和终点，求从起点到终点的最少步数网格值 `0` 表示可通行，`1` 表示障碍物；每一步只能向上、下、左、右移动

若起点与终点相同，结果为 0；若终点不可达，结果为 -1

**实现要求**

- 不允许使用外部库
- `solve` 函数签名保持不变
- 最终结果写入 `result[0]`
- 返回精确整数结果，官方容差为 0

**约束与测试**

- `1 ≤ rows, cols ≤ 1,000`
- `grid` 是长度为 `rows × cols` 的展平 `int32` 数组，元素只取 `0` 或 `1`
- `result` 是形状为 `(1,)` 的 `int32` 数组
- 起点和终点坐标均保证在网格范围内且位于可通行格
- 功能测试包含无路径、起终点相同、`1 × 1` 网格和复杂迷宫
- 性能测试使用 `500 × 500` 网格
- `challenge.py` 会直接把起点标记为已访问；当起终点相同时，无论该格原值如何都返回 0

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// grid, result are device pointers
extern "C" void solve(const int* grid, int* result, int rows, int cols, int start_row,
                      int start_col, int end_row, int end_col) {}
```

`grid` 和 `result` 位于 GPU 显存，坐标使用行列形式，线性下标为 `row * cols + col`；模板没有提供队列、访问数组或 kernel，需要在 `solve` 中分配 BFS 工作空间

## 实现思路

使用两个长度为 `rows × cols` 的队列，分别保存当前层和下一层；当前层中的节点距离相同，因此每完成一轮便令 `distance += 1`

每个线程处理当前层的一个格子，并检查四个邻居；合法邻居必须坐标未越界、满足 `grid[index] == 0`，并且尚未访问使用 `atomicCAS(&visited[index], 0, 1)` 原子认领邻居，避免多个父节点重复入队；认领成功后，再通过 `atomicAdd` 获取下一层队列位置

每轮 kernel 提交到默认流后，阻塞式 D2H `cudaMemcpy` 会等待本轮扩展完成；CPU 读取下一层长度后再交换队列，因此各层保持严格顺序；第一次发现终点时，对应层数就是最短距离

起终点相同时，在分配工作区前直接写 0；若队列耗尽仍未发现终点，则写 -1；`challenge.py` 保证起点合法，搜索从起点开始，不再检查起点格值

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void expand_frontier(const int* grid, int rows, int cols,
                                const int* current, int current_size,
                                int* next, int* next_size, int* visited,
                                int end_index, int* found) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= current_size) return;

    int index = current[i];
    int row = index / cols;
    int col = index % cols;
    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};

    for (int direction = 0; direction < 4; ++direction) {
        int nr = row + dr[direction];
        int nc = col + dc[direction];
        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;

        int neighbor = nr * cols + nc;
        if (grid[neighbor] != 0) continue;
        if (atomicCAS(&visited[neighbor], 0, 1) == 0) {
            int position = atomicAdd(next_size, 1);
            next[position] = neighbor;
            if (neighbor == end_index) atomicExch(found, 1);
        }
    }
}

// grid, result are device pointers
extern "C" void solve(const int* grid, int* result, int rows, int cols, int start_row,
                      int start_col, int end_row, int end_col) {
    int start = start_row * cols + start_col;
    int end = end_row * cols + end_col;
    if (start == end) {
        int zero = 0;
        cudaMemcpy(result, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        return;
    }

    int total = rows * cols;
    int *frontier_a, *frontier_b, *visited, *next_size, *found;
    cudaMalloc(&frontier_a, (size_t)total * sizeof(int));
    cudaMalloc(&frontier_b, (size_t)total * sizeof(int));
    cudaMalloc(&visited, (size_t)total * sizeof(int));
    cudaMalloc(&next_size, sizeof(int));
    cudaMalloc(&found, sizeof(int));
    cudaMemset(visited, 0, (size_t)total * sizeof(int));
    cudaMemset(found, 0, sizeof(int));
    cudaMemcpy(frontier_a, &start, sizeof(int), cudaMemcpyHostToDevice);
    int one = 1;
    cudaMemcpy(visited + start, &one, sizeof(int), cudaMemcpyHostToDevice);

    int current_size = 1;
    int distance = 0;
    int answer = -1;
    while (current_size > 0) {
        cudaMemset(next_size, 0, sizeof(int));
        int threads = 256;
        int blocks = (current_size + threads - 1) / threads;
        expand_frontier<<<blocks, threads>>>(grid, rows, cols,
            frontier_a, current_size, frontier_b, next_size, visited, end, found);

        int host_next_size = 0;
        int host_found = 0;
        cudaMemcpy(&host_next_size, next_size, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&host_found, found, sizeof(int), cudaMemcpyDeviceToHost);
        ++distance;
        if (host_found) {
            answer = distance;
            break;
        }

        int* temporary = frontier_a;
        frontier_a = frontier_b;
        frontier_b = temporary;
        current_size = host_next_size;
    }

    cudaMemcpy(result, &answer, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaFree(found);
    cudaFree(next_size);
    cudaFree(visited);
    cudaFree(frontier_b);
    cudaFree(frontier_a);
}
```
