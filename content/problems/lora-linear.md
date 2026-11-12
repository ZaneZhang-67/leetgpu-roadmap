---
id: lora-linear
leetgpuId: 85
titleEn: LoRA Linear
titleZh: LoRA 线性层
description: 使用 CUDA 计算基础线性变换与低秩 LoRA 残差分支
problemUrl: https://leetgpu.com/challenges/lora-linear
updatedAt: 2026-11-12
---
## 题目定义

给定 FP32 矩阵 `x(batch,d_in)`、基础权重 `W(d_out,d_in)`、下投影 `A(rank,d_in)` 和上投影 `B(d_out,rank)`，需要计算：

```text
output = xWᵀ + lora_scale × (xAᵀ)Bᵀ
```

`lora_scale` 已经是最终的缩放系数。接口没有单独的 alpha 参数，因此不需要再除以 rank。基础分支必须保留，LoRA 分支作为残差加到基础输出上。

### 示例

```text
batch=2，d_in=4，d_out=3，rank=2，lora_scale=0.5
x=[[1,0,-1,2],[0,1,1,-1]]
W=[[1,0,0,0],[0,1,0,0],[0,0,1,0]]
A=[[1,0,0,0],[0,1,0,0]]
B=[[1,0],[0,1],[0,0]]
output=[[1.5,0,-1],[0,1.5,1]]
```

约束为 `1≤batch≤1024`、`1≤d_in,d_out≤8192`、`1≤rank≤256`，并且 `rank<min(d_in,d_out)`。性能用例为 `batch=256`、`d_in=d_out=4096`、`rank=64`、`lora_scale=0.015625`，容差为 `1e-4`。

## 模板代码解析

CUDA starter 的签名为：

```text
extern "C" void solve(const float* x, const float* W, const float* A, const float* B,
                      float* output, int batch, int d_in, int d_out,
                      int rank, float lora_scale)
```

所有矩阵均按行主序保存。`W`、`A`、`B` 在公式中都经过转置后参与右乘，因此代码实际读取的是各自的一整行。输出形状为 `(batch,d_out)`。

## 实现思路

先计算形状为 `(batch,rank)` 的中间矩阵 `hidden=xAᵀ`。输出 kernel 使用 `16×16` Shared Memory tiled GEMM 计算基础分支：每个 Block 负责 `(batch,d_out)` 的一个子块，并沿 `d_in` 分块协作加载 16 列 `x` 和 W 对应行的 16 个元素。同一个输入值和权重值会被 Block 内的多个输出复用，并通过 FP32 累加得到 `base`。

完成所有 `d_in` tile 后，每个输出线程再沿 rank 累加 `hidden` 与 B 对应行的点积，得到 `delta`，最终写入 `base+lora_scale*delta`。边缘线程在加载阶段写入零值，并继续参与所有同步，只在最终写回时判断边界。

拆分为两个 kernel，可以避免为每个输出列重复计算 `xAᵀ`。临时空间为 `batch×rank` 个 FP32，输出 kernel 还会使用两个 `16×16` FP32 Shared Memory tile。这份实现可直接作为 tiled 性能基线提交，后续还可优化 hidden 分支以及 LoRA rank 维的数据复用。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void lora_hidden_kernel(const float* x, const float* A, float* hidden,
                                   int batch, int d_in, int rank) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (b >= batch || r >= rank) return;
    float sum = 0.0f;
    for (int k = 0; k < d_in; ++k) sum += x[b * d_in + k] * A[r * d_in + k];
    hidden[b * rank + r] = sum;
}

__global__ void lora_output_kernel(const float* x, const float* W, const float* B,
                                   const float* hidden, float* output,
                                   int batch, int d_in, int d_out, int rank, float scale) {
    __shared__ float X[16][16];
    __shared__ float Wtile[16][16];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int o = blockIdx.x * 16 + tx;
    int b = blockIdx.y * 16 + ty;
    float base = 0.0f;

    for (int k0 = 0; k0 < d_in; k0 += 16) {
        int x_k = k0 + tx;
        int w_k = k0 + ty;
        X[ty][tx] = (b < batch && x_k < d_in) ? x[b * d_in + x_k] : 0.0f;
        Wtile[ty][tx] = (o < d_out && w_k < d_in) ? W[o * d_in + w_k] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 16; ++i) base += X[ty][i] * Wtile[i][tx];
        __syncthreads();
    }

    if (b < batch && o < d_out) {
        float delta = 0.0f;
        for (int r = 0; r < rank; ++r) delta += hidden[b * rank + r] * B[o * rank + r];
        output[b * d_out + o] = base + scale * delta;
    }
}

// x, W, A, B, output are device pointers
extern "C" void solve(const float* x, const float* W, const float* A, const float* B,
                      float* output, int batch, int d_in, int d_out,
                      int rank, float lora_scale) {
    float* hidden = nullptr;
    cudaMalloc(&hidden, static_cast<size_t>(batch) * rank * sizeof(float));
    dim3 threads(16, 16);
    dim3 hidden_blocks((rank + 15) / 16, (batch + 15) / 16);
    dim3 output_blocks((d_out + 15) / 16, (batch + 15) / 16);
    lora_hidden_kernel<<<hidden_blocks, threads>>>(x, A, hidden, batch, d_in, rank);
    lora_output_kernel<<<output_blocks, threads>>>(x, W, B, hidden, output,
                                                   batch, d_in, d_out, rank, lora_scale);
    cudaDeviceSynchronize();
    cudaFree(hidden);
}
```
