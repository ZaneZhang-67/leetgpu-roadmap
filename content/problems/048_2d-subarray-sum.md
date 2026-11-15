---
id: 2d-subarray-sum
leetgpuId: 48
titleEn: 2D Subarray Sum
titleZh: 二维子数组求和
description: 使用 CUDA 对行主序矩阵中的闭区间矩形执行并行归约，介绍局部坐标映射、行跨度和部分和合并
problemUrl: https://leetgpu.com/challenges/2d-subarray-sum
updatedAt: 2026-11-12
---
## 题目定义

给定形状为 `N × M` 的 32 位整数矩阵 `input`，计算行区间 `[S_ROW, E_ROW]` 与列区间 `[S_COL, E_COL]` 组成的闭区间矩形之和

```text
output[0] = Σ input[row, col]
```

其中：

```text
S_ROW ≤ row ≤ E_ROW
S_COL ≤ col ≤ E_COL
```

### 实现要求

- 只使用原生功能，不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写入 `output`

### 示例 1

```text
input = [[1, 2, 3],
         [4, 5, 1]]
S_ROW = 0，E_ROW = 1，S_COL = 1，E_COL = 2
输出：11
```

### 示例 2

```text
input = [[5, 10],
         [5,  2]]
S_ROW = 0，E_ROW = 0，S_COL = 1，E_COL = 1
输出：10
```

### 约束

- `1 ≤ N, M ≤ 10,000`
- `1 ≤ input[i] ≤ 10`
- `0 ≤ S_ROW ≤ E_ROW ≤ N - 1`
- `0 ≤ S_COL ≤ E_COL ≤ M - 1`
- 性能测试使用 `N = M = 10,000`，行区间 `[0, 9998]`，列区间 `[1, 9999]`

## 模板代码解析

平台提供固定的 CUDA 入口：

```cpp
#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(
    const int* input,
    int* output,
    int N,
    int M,
    int S_ROW,
    int E_ROW,
    int S_COL,
    int E_COL
) {}
```

- `input` 是 GPU 上按行主序保存的 `N × M` 矩阵
- `output` 是长度为 1 的 `int32` 数组
- 四个边界下标都包含在求和区域内

模板没有提供 kernel、输出初始化或启动配置

## 实现思路

将目标矩形看作紧凑的一维任务空间，每个线程处理若干个局部元素，再把局部和归约

### 1. 目标矩形尺寸

```cpp
int rows = E_ROW - S_ROW + 1;
int cols = E_COL - S_COL + 1;
```

目标元素数量为 `rows × cols`，官方上限为 `10,000²=100,000,000`；参考实现统一使用 64 位整数计算任务数量和线性下标，保持乘法与地址表达式的宽类型一致，并为维度扩展保留余量

### 2. 局部线性下标转坐标

对目标区域内的局部下标 `localIndex`：

```cpp
int localRow = localIndex / cols;
int localCol = localIndex % cols;
int row = S_ROW + localRow;
int col = S_COL + localCol;
```

原矩阵行主序地址为：

```cpp
input[row * M + col]
```

注意行跨度是完整矩阵列数 `M`，而不是目标矩形宽度 `cols`

### 3. Grid-stride 遍历与归约

线程以整个 Grid 的线程数为步长覆盖目标元素，然后在共享内存中执行折半归约每个 Block 最终只产生一个部分和

### 4. 输出合并

`solve` 先清零 `output[0]`，各 Block 的 0 号线程通过 `atomicAdd` 合并结果

reference 使用 `input[S_ROW:E_ROW+1, S_COL:E_COL+1]`，因此单行、单列、完整矩阵和非零起点矩形都由同一套闭区间映射处理

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void subarray_sum_2d_kernel(
    const int* input,
    int* output,
    int M,
    int S_ROW,
    int S_COL,
    int cols,
    long long total
) {
    __shared__ int shared[256];

    int tid = threadIdx.x;
    long long index =
        static_cast<long long>(blockIdx.x) * blockDim.x + tid;
    long long stride =
        static_cast<long long>(blockDim.x) * gridDim.x;
    int local = 0;

    for (long long i = index; i < total; i += stride) {
        int localRow = static_cast<int>(i / cols);
        int localCol = static_cast<int>(i % cols);
        int row = S_ROW + localRow;
        int col = S_COL + localCol;
        local += input[static_cast<long long>(row) * M + col];
    }

    shared[tid] = local;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            shared[tid] += shared[tid + step];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(
    const int* input,
    int* output,
    int N,
    int M,
    int S_ROW,
    int E_ROW,
    int S_COL,
    int E_COL
) {
    (void)N;
    constexpr int threadsPerBlock = 256;
    int rows = E_ROW - S_ROW + 1;
    int cols = E_COL - S_COL + 1;
    long long total = static_cast<long long>(rows) * cols;
    long long needed =
        (total + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = needed > 1024 ? 1024 : static_cast<int>(needed);

    cudaMemset(output, 0, sizeof(int));
    subarray_sum_2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, M, S_ROW, S_COL, cols, total
    );
    cudaDeviceSynchronize();
}
```
