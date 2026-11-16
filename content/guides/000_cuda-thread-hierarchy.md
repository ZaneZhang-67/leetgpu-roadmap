---
id: cuda-thread-hierarchy
order: 0
difficulty: easy
titleEn: CUDA Thread Hierarchy
titleZh: CUDA 线程层级
description: 通过一张图认识 CUDA 中 Grid、Block、Thread 的关系，并理解一维全局下标的计算方式
updatedAt: 2026-11-15
---
## Grid、Block 和 Thread

调用一次 CUDA kernel 时，GPU 会启动许多线程。CUDA 使用三层结构组织这些线程：一个 Grid 包含多个 Block，一个 Block 包含多个 Thread

![CUDA 一维 Grid、Block 和 Thread 的索引关系](uploads/cuda-grid-block-thread-indexing.png)

图中每种颜色代表一个 Block，每个小格代表一个 Thread。同一个 Block 内，线程编号 `threadIdx.x` 从 `0` 开始；不同 Block 使用 `blockIdx.x` 区分

图片来源：[NVIDIA Developer Blog](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)

## 计算全局下标

`threadIdx.x` 只是线程在当前 Block 内的编号。要找到线程在整个 Grid 中负责的位置，还需要加上前面所有 Block 包含的线程数：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

图中高亮线程位于 Block 2，每个 Block 有 256 个线程，它在 Block 内的编号是 3，因此：

```text
i = 2 × 256 + 3 = 515
```

这个全局下标通常直接对应数组下标。处理一维数组时，线程 `i` 负责第 `i` 个元素

## Kernel 启动参数

CUDA 使用三尖括号指定 Grid 中的 Block 数和每个 Block 的 Thread 数：

```cpp
kernel<<<blocksPerGrid, threadsPerBlock>>>(...);
```

例如 `kernel<<<4, 256>>>` 会启动 4 个 Block，每个 Block 有 256 个 Thread，总共 1024 个线程

下一篇向量加法会使用这套映射，让每个线程独立计算一个输出元素
