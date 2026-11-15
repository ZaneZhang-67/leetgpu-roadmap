---
id: sparse-matrix-dense-matrix-multiplication
leetgpuId: 75
titleEn: Sparse Matrix-Dense Matrix Multiplication
titleZh: 稀疏矩阵与稠密矩阵乘法
description: 按运行时真实布局计算含零元素的稠密 A 与稠密 B 的矩阵乘法，并利用零值跳过无效乘加
problemUrl: https://leetgpu.com/challenges/sparse-matrix-dense-matrix-multiplication
updatedAt: 2026-11-12
---
## 题目定义

运行时传入的 `A` 是形状 `[M,N]` 的 FP32 矩阵，零元素表示稀疏位置；`B` 是 `[N,K]` 的稠密矩阵，输出 `C` 为 `[M,K]`：

```text
C[i,j] = Σ A[i,n] * B[n,j]
```

尽管题名包含 Sparse Matrix，官方 runtime 并未传入 CSR 的 `row_ptr`、`col_idx` 和 `values`。`A` 仍以完整的行主序 `M*N` 数组传入，`nnz` 只是独立的整数参数，reference 也直接执行稠密 `A @ B`

### 官方示例

```text
输入：A = [[2,0,0,1],
           [0,3,0,0],
           [0,0,4,0]]
      B = [[1,2],
           [3,4],
           [5,6],
           [7,8]]
输出：C = [[9,12],
           [9,12],
           [20,24]]
```

### 完整约束

- `1 ≤ M,N,K ≤ 8192`
- `A`、`B`、`C` 均为 FP32 行主序矩阵
- `A` 与 `B` 的元素范围为 `[-10,10]`
- 题面描述 `A` 约有 60%–70% 元素为零
- `nnz` 表示 `A` 中非零元素的数量，但不携带 CSR 索引；运行时仍传入完整的 `M*N` 数组
- 性能测试使用 `M=4096,N=2048,K=512`
- 运行时评测覆盖 `nnz=0`、单位矩阵、负值和非二次幂维度
- 判定容差为 `atol=0.001, rtol=0.001`

## 模板代码解析

真实 CUDA 签名为：

```cpp
#include <cuda_runtime.h>

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C,
                      int M, int N, int K, int nnz) {}
```

三个矩阵均为行主序设备数组，参数顺序不能按常见 GEMM 的 `M,K,N` 误读。`nnz` 不携带非零元素的位置，不能据此恢复 CSR，也不能只遍历 `A` 的前 `nnz` 个元素

## 实现思路

一个二维线程负责一个输出元素 `(row,col)`，沿共享维度 `n=0..N-1` 扫描

```text
A index = row*N+n
B index = n*K+col
C index = row*K+col
```

读取到 `A[row,n]==0` 时跳过对应乘加，在不改变运行时语义的前提下利用稀疏性。全零矩阵会得到全零输出；`K=1` 时自然退化为矩阵向量乘法

`nnz` 仅用于描述输入，不参与寻址。参考实现显式忽略它，提交代码也保持这一行为，避免把逻辑建立在缺失的 CSR 元数据上

这份基线实现复杂度仍是 `O(MNK)`；进一步优化可使用 tiling 复用 `A` 与 `B` 数据，但必须保持对完整行主序 `A` 的读取

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void spmm_dense_layout(const float* A, const float* B,
                                  float* C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= K) return;

    float sum = 0.0f;
    for (int inner = 0; inner < N; ++inner) {
        float a = A[row * N + inner];
        if (a != 0.0f) sum += a * B[inner * K + col];
    }
    C[row * K + col] = sum;
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C,
                      int M, int N, int K, int nnz) {
    (void)nnz;
    dim3 threads(16, 16);
    dim3 blocks((K + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y);
    spmm_dense_layout<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```
