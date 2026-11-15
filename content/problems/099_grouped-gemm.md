---
id: grouped-gemm
leetgpuId: 99
titleEn: Grouped GEMM
titleZh: 分组矩阵乘法
description: 使用 CUDA tiled GEMM 为连续行组选择各自权重矩阵
problemUrl: https://leetgpu.com/challenges/grouped-gemm
updatedAt: 2026-11-12
---
## 题目定义

`A(M_total,K)` 按连续行划分为 G 组，边界由长度 `G+1` 的 INT32 数组 `group_offsets` 给出。第 g 组使用独立权重 `B[g](K,N)`：

```text
C[group_offsets[g]:group_offsets[g+1]]
  = A[group_offsets[g]:group_offsets[g+1]] × B[g]
```

空组的起止 offset 相同，因此不会写入任何行。这里使用的不是指针数组：A、B、C 都是连续的大数组，B 的布局为 `(G,K,N)`。

### 示例

```text
G=2，M_total=4，K=2，N=3，offsets=[0,2,4]
A=[[1,2],[3,4],[5,6],[7,8]]
B0=[[1,0,1],[0,1,0]]
B1=[[1,1,0],[1,1,1]]
C=[[1,2,1],[3,4,3],[11,11,6],[15,15,8]]
```

约束为 `1≤G≤64`、`1≤M_total≤16384`、`1≤K,N≤4096`，offset 非递减且首尾为 0、M_total。性能用例为 `G=8`、`M_total=8192`、`K=1024`、`N=2048`，容差 `1e-4`。

## 模板代码解析

starter 签名为：

```text
extern "C" void solve(const float* A, const float* B, const int* group_offsets,
                      float* C, int G, int M_total, int K, int N)
```

第 g 个 B 矩阵以 `g*K*N` 为基址。A、C 仍使用全局行编号，无需把各组数据复制到独立缓冲。

## 实现思路

每个组分别启动一个 `16×16` tiled GEMM kernel。host 端先复制规模很小的 offsets 数组，用来确定各组行数，并跳过空组。kernel 将局部行加上组起点，得到 A/C 的全局行；B 则使用组基址。

每个 Block 协作把 A 和 B 的 tile 加载到 Shared Memory，K 尾部和矩阵边缘补零。在最大性能规模下，大量 Block 可以并行计算，无需串行执行行列点积。G 最多 64，逐组 launch 的开销相对大矩阵乘法可控。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <vector>

__global__ void group_gemm_kernel(const float* A, const float* B, float* C,
                                  int row_start, int rows, int K, int N) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    int tx = threadIdx.x, ty = threadIdx.y;
    int local_row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;
    int global_row = row_start + local_row;
    float sum = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 16) {
        int ak = k0 + tx, bk = k0 + ty;
        As[ty][tx] = (local_row < rows && ak < K) ? A[global_row * K + ak] : 0.0f;
        Bs[ty][tx] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 16; ++i) sum += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }
    if (local_row < rows && col < N) C[global_row * N + col] = sum;
}

extern "C" void solve(const float* A, const float* B, const int* group_offsets,
                      float* C, int G, int M_total, int K, int N) {
    std::vector<int> offsets(G + 1);
    cudaMemcpy(offsets.data(), group_offsets, (G + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    dim3 threads(16, 16);
    for (int g = 0; g < G; ++g) {
        int start = offsets[g], rows = offsets[g + 1] - start;
        if (rows == 0) continue;
        dim3 blocks((N + 15) / 16, (rows + 15) / 16);
        group_gemm_kernel<<<blocks, threads>>>(A, B + static_cast<size_t>(g) * K * N,
                                               C, start, rows, K, N);
    }
    cudaDeviceSynchronize();
}
```
