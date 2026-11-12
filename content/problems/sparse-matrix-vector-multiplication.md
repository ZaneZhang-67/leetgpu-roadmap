---
id: sparse-matrix-vector-multiplication
leetgpuId: 18
titleEn: Sparse Matrix-Vector Multiplication
titleZh: 稀疏矩阵向量乘法
description: 使用 CUDA 计算行主序矩阵与稠密向量的乘积，让一个 Block 协作处理一行并跳过零元素
problemUrl: https://leetgpu.com/challenges/sparse-matrix-vector-multiplication
updatedAt: 2026-11-12
---
## 题目定义

给定行主序矩阵 `A`，形状为 `M × N`，以及长度为 `N` 的稠密向量 `x`，计算：

```text
y = A × x
```

输出向量 `y` 的长度为 `M`，每个元素为：

```text
y[i] = Σ A[i, j] × x[j]，其中 0 ≤ j < N
```

题面说明矩阵约有 60%～70% 的元素为零，`nnz` 表示非零元素数量不过平台传入的 `A` 仍是完整的行主序矩阵，不是 CSR 等压缩格式

### 实现要求

- 只使用 GPU 原生功能，不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写入向量 `y`

### 示例

```text
A（3 × 4）：
[[5.0, 0.0, 0.0, 1.0],
 [0.0, 2.0, 3.0, 0.0],
 [0.0, 0.0, 0.0, 4.0]]

x = [1.0, 2.0, 3.0, 4.0]

y = [9.0, 13.0, 16.0]
```

例如：

```text
y[0] = 5×1 + 0×2 + 0×3 + 1×4 = 9
y[1] = 0×1 + 2×2 + 3×3 + 0×4 = 13
y[2] = 0×1 + 0×2 + 0×3 + 4×4 = 16
```

### 约束

- `1 ≤ M, N ≤ 10,000`
- `A` 约有 60%～70% 的元素为零
- 性能测试使用 `M = 1,000`、`N = 10,000`

## 模板代码解析

平台提供固定的 CUDA 入口：

```cpp
#include <cuda_runtime.h>

// A, x, y are device pointers
extern "C" void solve(
    const float* A,
    const float* x,
    float* y,
    int M,
    int N,
    int nnz
) {}
```

- `A` 指向 GPU 上行主序的 `M × N` 完整矩阵
- `x` 指向长度为 `N` 的只读向量
- `y` 指向长度为 `M` 的输出向量
- `M`、`N` 是矩阵行数和列数
- `nnz` 是矩阵中非零元素总数

签名没有传入行偏移、列索引或非零值数组，因此不能把 `A` 当作 CSR 数据解释`nnz` 可描述稀疏程度，但不足以定位各个非零元素，正确实现仍需按 `A[row * N + col]` 访问矩阵

## 实现思路

让一个 Block 协作计算一行输出每个线程处理该行中的若干列，累加局部乘积，再在共享内存中归约

### 1. Block 与矩阵行的映射

```cpp
int row = blockIdx.x;
```

Grid 启动 `M` 个 Block，因此每一行都有且只有一个 Block 负责，最终由该 Block 写入 `y[row]`

### 2. 线程跨步遍历列

Block 内线程 `tid` 处理：

```text
col = tid, tid + 256, tid + 512, ...
```

行主序地址为：

```cpp
A[row * N + col]
```

读取矩阵值后，可以跳过零元素：

```cpp
float value = A[row * N + col];
if (value != 0.0f) {
    sum += value * x[col];
}
```

这不会减少读取完整矩阵的带宽，因为输入不是压缩格式，但能避免零元素对应的乘加

### 3. Block 内归约

各线程把局部和写入共享内存，随后执行折半归约：

```text
256 个局部和 → 128 → 64 → ... → 1 个行结果
```

每轮都调用 `__syncthreads()` 保证线程间可见性

### 4. 写入输出

Block 的 0 号线程写入：

```cpp
y[row] = shared[0];
```

不同 Block 写入不同的 `y[row]`，没有写冲突，不需要原子操作

### 5. 边界和规模

线程循环条件为 `col < N`，因此任意 `N` 都能处理，包括 `N` 小于 256 或不是 256 的倍数

Grid 只有 `M` 个 Block，题目给定 `M ≤ 10,000`，能直接覆盖全部输出行参数 `nnz` 保留在 `solve` 签名中以符合平台接口，但在这种稠密存储布局下无需参与地址计算

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void spmv_kernel(
    const float* A,
    const float* x,
    float* y,
    int M,
    int N
) {
    __shared__ float shared[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;

    if (row < M) {
        for (int col = tid; col < N; col += blockDim.x) {
            float value = A[row * N + col];
            if (value != 0.0f) {
                sum += value * x[col];
            }
        }
    }

    shared[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0 && row < M) {
        y[row] = shared[0];
    }
}

// A, x, y are device pointers
extern "C" void solve(
    const float* A,
    const float* x,
    float* y,
    int M,
    int N,
    int nnz
) {
    (void)nnz;
    constexpr int threadsPerBlock = 256;
    spmv_kernel<<<M, threadsPerBlock>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}
```
