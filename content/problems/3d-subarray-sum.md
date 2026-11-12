---
id: 3d-subarray-sum
leetgpuId: 49
titleEn: 3D Subarray Sum
titleZh: 三维子数组求和
description: 使用 CUDA 对三维张量中的闭区间长方体执行并行归约，介绍局部三维坐标与完整张量行主序地址的转换
problemUrl: https://leetgpu.com/challenges/3d-subarray-sum
updatedAt: 2026-11-12
---
## 题目定义

给定形状为 `N × M × K` 的 32 位整数张量 `input`，计算以下闭区间长方体内的元素总和：

```text
S_DEP ≤ depth ≤ E_DEP
S_ROW ≤ row ≤ E_ROW
S_COL ≤ col ≤ E_COL
```

结果写入 `output[0]`

### 实现要求

- 只使用原生功能，不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写入 `output`

### 示例 1

```text
input = [[[1, 2, 3],
          [4, 5, 1]],
         [[1, 1, 1],
          [2, 2, 2]]]
N = 2，M = 2，K = 3
S_DEP = 0，E_DEP = 1
S_ROW = 0，E_ROW = 0
S_COL = 1，E_COL = 2
输出：7
```

### 示例 2

```text
input = [[[5, 10],
          [5,  2],
          [2,  2]]]
N = 1，M = 3，K = 2
S_DEP = 0，E_DEP = 0
S_ROW = 0，E_ROW = 2
S_COL = 1，E_COL = 1
输出：14
```

### 约束

- `1 ≤ N, M, K ≤ 500`
- `1 ≤ input[i] ≤ 10`
- `0 ≤ S_DEP ≤ E_DEP ≤ N - 1`
- `0 ≤ S_ROW ≤ E_ROW ≤ M - 1`
- `0 ≤ S_COL ≤ E_COL ≤ K - 1`
- 性能测试使用 `N = M = K = 500`，深度区间 `[11, 498]`、行区间 `[0, 499]`、列区间 `[1, 489]`

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
    int K,
    int S_DEP,
    int E_DEP,
    int S_ROW,
    int E_ROW,
    int S_COL,
    int E_COL
) {}
```

- `input` 是 GPU 上按 `(N, M, K)` 行主序保存的 `int32` 张量
- `output` 是长度为 1 的 `int32` 数组
- `N`、`M`、`K` 分别对应深度、行和列维度
- 六个边界下标都包含在求和区域内

模板没有提供 kernel、输出初始化或启动配置

## 实现思路

将目标长方体压平为一维任务空间，线程从局部线性下标恢复深度、行和列，再访问完整张量

### 1. 目标区域尺寸

```cpp
int depths = E_DEP - S_DEP + 1;
int rows = E_ROW - S_ROW + 1;
int cols = E_COL - S_COL + 1;
```

目标元素数量为 `depths × rows × cols`，官方上限为 `500³=125,000,000`；参考实现仍从第一次乘法起使用 `long long` 计算数量和地址，避免维度乘法或后续线性寻址被窄类型截断

### 2. 恢复局部三维坐标

令一个目标深度切片包含 `rows × cols` 个元素：

```cpp
localDepth = index / (rows * cols);
remainder = index % (rows * cols);
localRow = remainder / cols;
localCol = remainder % cols;
```

再加上三个起点，得到完整张量坐标

### 3. 完整张量地址

`input` 的连续维度顺序与 reference 的 `(N, M, K)` 一致，行主序地址为：

```text
((depth × M) + row) × K + col
```

步长使用完整维度 `M` 和 `K`，不能使用目标区域的 `rows`、`cols` 代替

### 4. 并行归约

线程通过 Grid-stride loop 累加多个目标元素，Block 内使用共享内存折半归约，最后每个 Block 执行一次 `atomicAdd`

这套映射同时覆盖单个元素、单个深度切片、完整张量以及三个维度都从非零位置开始的子区域

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void subarray_sum_3d_kernel(
    const int* input,
    int* output,
    int M,
    int K,
    int S_DEP,
    int S_ROW,
    int S_COL,
    int rows,
    int cols,
    long long total
) {
    __shared__ int shared[256];

    int tid = threadIdx.x;
    long long index =
        static_cast<long long>(blockIdx.x) * blockDim.x + tid;
    long long stride =
        static_cast<long long>(blockDim.x) * gridDim.x;
    long long plane = static_cast<long long>(rows) * cols;
    int local = 0;

    for (long long i = index; i < total; i += stride) {
        int localDepth = static_cast<int>(i / plane);
        long long remainder = i % plane;
        int localRow = static_cast<int>(remainder / cols);
        int localCol = static_cast<int>(remainder % cols);

        int depth = S_DEP + localDepth;
        int row = S_ROW + localRow;
        int col = S_COL + localCol;
        long long address =
            (static_cast<long long>(depth) * M + row) * K + col;
        local += input[address];
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
    int K,
    int S_DEP,
    int E_DEP,
    int S_ROW,
    int E_ROW,
    int S_COL,
    int E_COL
) {
    (void)N;
    constexpr int threadsPerBlock = 256;
    int depths = E_DEP - S_DEP + 1;
    int rows = E_ROW - S_ROW + 1;
    int cols = E_COL - S_COL + 1;
    long long total = static_cast<long long>(depths) * rows * cols;
    long long needed =
        (total + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = needed > 1024 ? 1024 : static_cast<int>(needed);

    cudaMemset(output, 0, sizeof(int));
    subarray_sum_3d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input,
        output,
        M,
        K,
        S_DEP,
        S_ROW,
        S_COL,
        rows,
        cols,
        total
    );
    cudaDeviceSynchronize();
}
```
