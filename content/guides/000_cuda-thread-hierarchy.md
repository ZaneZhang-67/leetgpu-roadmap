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

调用一次 CUDA kernel 时，GPU 会启动许多线程。CUDA 使用三层结构组织这些线程：一个 Grid 包含多个 Block，一个 Block 包含多个 Thread。

![Grid、Block 和 Thread 间的关系](uploads/grid-of-thread-blocks.png "Grid、Block 和 Thread 间的关系")

[](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)

## 怎么定位一个 thread？

## 线程的位置与 Block 的尺寸

一个 Grid 包含多个 Block，每个 Block 又包含多个 Thread。要确定一个线程的位置，需要回答两个问题：

1. **它属于 Grid 中的哪个 Block？**
2. **它位于这个 Block 内的哪个位置？**

CUDA 提供了几个内置变量，线程可以在 kernel 中直接读取：

| 变量 | 表示什么 | 可以理解为 |
|---|---|---|
| `blockIdx` | 当前 Block 在 Grid 中的坐标 | 我属于哪一块 |
| `threadIdx` | 当前 Thread 在 Block 内的坐标 | 我在块内的什么位置 |
| `blockDim` | 每个 Block 在各个方向上的线程数 | 一块有多大 |
| `gridDim` | Grid 在各个方向上的 Block 数 | 总共排列了多少块 |

其中，`Idx` 表示**索引或坐标**，从 `0` 开始；`Dim` 表示**尺寸或数量**。

这些变量都有 `.x`、`.y`、`.z` 三个分量。例如：

- `threadIdx.x`：线程在 Block 内沿 x 方向的坐标
- `blockIdx.x`：Block 在 Grid 内沿 x 方向的坐标
- `blockDim.x`：每个 Block 沿 x 方向有多少个线程

**`blockDim.x = 256` 表示 x 方向有 256 个线程，不是说 Block 有 256 个维度。**

一维和二维描述的是程序如何组织线程坐标，并不代表 GPU 上的线程在物理上排成一条线或一张表。

## 1. 一维 Block：线程沿一个方向编号

处理向量等一维数据时，可以使用一维 Block：

```cpp
kernel<<<4, 256>>>(...);
```

这次启动的含义是：

- Grid 中有 `4` 个 Block，`blockIdx.x` 的范围是 `0～3`
- 每个 Block 有 `256` 个线程，`threadIdx.x` 的范围是 `0～255`
- 每个线程读取到的 `blockDim.x` 都是 `256`
- 整个 Grid 一共包含 `4 × 256 = 1024` 个线程

**每个 Block 都有自己的一套线程编号。** 因此，Block 0 中有 `threadIdx.x = 3` 的线程，Block 1、Block 2 中也都有。

仅凭 `threadIdx.x`，还不足以区分整个 Grid 中的线程。

### 从块内编号计算全局下标

如果让各个 Block 按编号依次负责一段连续数组：

```text
Block 0 → 数组下标   0～255
Block 1 → 数组下标 256～511
Block 2 → 数组下标 512～767
Block 3 → 数组下标 768～1023
```

那么，当前 Block 对应的起始位置是：

```cpp
blockIdx.x * blockDim.x
```

再加上线程在 Block 内的偏移，就得到：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

以图中高亮的线程为例：

```text
blockIdx.x  = 2       当前线程属于 Block 2
blockDim.x  = 256     每个 Block 有 256 个线程
threadIdx.x = 3       当前线程在块内的编号是 3

i = 2 × 256 + 3 = 515
```

在“一线程处理一个数组元素”的映射中，这个线程就负责数组的第 `515` 号元素。

## 2. 二维 Block：线程用两个坐标定位

处理矩阵或图像时，可以把 Block 内的线程组织成二维形式：

```cpp
dim3 block(4, 2);
```

这表示一个 Block 沿 x 方向有 `4` 个线程，沿 y 方向有 `2` 个线程，总共包含 `4 × 2 = 8` 个线程。

用 `(threadIdx.x, threadIdx.y)` 表示每个线程的位置：

```text
                    threadIdx.x
               0       1       2       3

threadIdx.y 0  (0,0)   (1,0)   (2,0)   (3,0)
            1  (0,1)   (1,1)   (2,1)   (3,1)
```

此时：

```text
blockDim.x = 4
blockDim.y = 2

threadIdx.x 的范围：0～3
threadIdx.y 的范围：0～1
```

注意，**`blockDim.x` 此时只表示一行有多少个线程，不再等于整个 Block 的线程总数。** 二维 Block 的线程总数是：

```cpp
blockDim.x * blockDim.y
```

### Grid 也可以按二维排列 Block

在上述 Block 的基础上，再定义一个二维 Grid：

```cpp
dim3 block(4, 2);  // 每块 4 列、2 行线程，共 8 个线程
dim3 grid(3, 2);   // Grid 中有 3 列、2 行 Block，共 6 个 Block

kernel<<<grid, block>>>(...);
```

这里存在两套不同层级的坐标：

- `blockIdx.x`、`blockIdx.y`：定位 **Block**
- `threadIdx.x`、`threadIdx.y`：定位 **Block 内的线程**

### 从块内坐标计算全局坐标

假设让 x 方向对应矩阵的列，y 方向对应矩阵的行，那么两个方向分别计算：

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

推导方式与一维相同：

- `blockIdx.x * blockDim.x`：当前 Block 的起始列
- `blockIdx.y * blockDim.y`：当前 Block 的起始行
- 再分别加上线程在块内的列偏移、行偏移

例如，当前线程位于 Block `(2, 1)`，块内坐标为 `(3, 1)`：

```text
col = 2 × 4 + 3 = 11
row = 1 × 2 + 1 = 3
```

这个线程对应全局坐标**第 3 行、第 11 列**，行列编号都从 `0` 开始。

如果矩阵以行主序存储在一维数组中，还需要根据矩阵的实际列数 `width` 转换成数组下标：

```cpp
int i = row * width + col;
```

这里的 `width` 是**矩阵的实际列数**，不是 `blockDim.x`，也不一定等于 Grid 覆盖的总宽度。Grid 向上取整覆盖数据时，边缘线程还需要检查：

```cpp
if (row < height && col < width) {
    // 访问矩阵元素
}
```

**Block 和 Grid 的维度可以独立设置。** 二维 Block 并不要求 Grid 也必须是二维；上面的示例将两者都设为二维，是为了让线程坐标与矩阵的行、列对应起来。
