---
id: token-embedding-layer
leetgpuId: 106
titleEn: Token Embedding Layer
titleZh: Token 嵌入层
description: 使用 CUDA 融合 token 与位置嵌入查表、相加和逐 token LayerNorm
problemUrl: https://leetgpu.com/challenges/token-embedding-layer
updatedAt: 2026-11-12
---
## 题目定义

对每个 `(b,t)`，分别根据 token ID 和 position ID 从两个行主序表中取出长度为 `D` 的行，再将两行相加：

```text
s = token_embeddings[token_ids[b,t]] + position_embeddings[position_ids[t]]
output[b,t,d] = gamma[d]*(s[d]-mean)/sqrt(var+eps)+beta[d]
```

均值和总体方差都只沿当前 token 的 `D` 维计算，方差除以 `D`，不使用 Bessel 校正。`position_ids` 只依赖时间位置，对 batch 共享

### 示例

```text
B=1,T=2,V=3,P=2,D=4, token_ids=[[2,0]], position_ids=[0,1]
output = [[[1.4142,0,-1.4142,0],
           [-0.5773,-0.5773,-0.5773,1.7320]]]
```

**约束与测试**

- `1≤B≤64`，`1≤T≤1024`，`1≤V≤50,000`，`1≤P≤4096`，`1≤D≤1024`
- token ID 满足 `0≤id<V`，position ID 满足 `0≤id<P`；运行时不测试越界 ID
- ID 为 INT32，其余张量为 FP32，`eps=1e-5`
- 测试覆盖单 token、全零嵌入、负嵌入、重复 ID 和非二次幂维度
- 性能测试 `B=32,T=512,V=30,000,P=2048,D=768`，容差 `atol=rtol=1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// token_ids, position_ids, token_embeddings, position_embeddings, gamma, beta, output are device
// pointers
extern "C" void solve(const int* token_ids, const int* position_ids, const float* token_embeddings,
                      const float* position_embeddings, const float* gamma, const float* beta,
                      float* output, int B, int T, int V, int P, int D, float eps) {}
```

官方 starter 中只有空入口。token 表的形状为 `(V,D)`，位置表为 `(P,D)`，输出为 `(B,T,D)`。`V`、`P` 用于描述合法 ID 范围；正式输入已保证范围正确

## 实现思路

每个 Block 处理一个 `(b,t)`。线程根据两个 ID 查找表行，并将对应维度相加：第一遍用 FP64 归约求均值，第二遍围绕 FP32 均值归约中心平方差，第三遍重新计算相加结果并完成逐维仿射变换。这种两遍统计可以避免 `E[s²]-E[s]²` 在高偏置、低方差向量上失稳，也省去了大小为 `B*T*D` 的中间缓冲区

网格共有 `B*T` 个 Block，共享内存复用一组 FP64 线程局部统计量。D 无需是二次幂；线程数固定为 256，未分配到元素的线程以零参与归约

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void embedding_kernel(const int* token_ids, const int* position_ids,
                                 const float* token_embeddings,
                                 const float* position_embeddings,
                                 const float* gamma, const float* beta,
                                 float* output, int T, int D, float eps) {
    int bt = blockIdx.x;
    int t = bt % T;
    int token = token_ids[bt];
    int position = position_ids[t];
    size_t tok_base = (size_t)token * D;
    size_t pos_base = (size_t)position * D;
    size_t out_base = (size_t)bt * D;
    extern __shared__ double shared[];
    __shared__ float mean_value;
    __shared__ float inv_value;
    int tid = static_cast<int>(threadIdx.x);
    int block_size = static_cast<int>(blockDim.x);

    double local_sum = 0.0;
    for (int d = tid; d < D; d += block_size) {
        float s = token_embeddings[tok_base + d] + position_embeddings[pos_base + d];
        local_sum += (double)s;
    }
    shared[tid] = local_sum;
    __syncthreads();
    for (int off = block_size / 2; off > 0; off >>= 1) {
        if (tid < off) shared[tid] += shared[tid + off];
        __syncthreads();
    }
    if (tid == 0) mean_value = (float)(shared[0] / (double)D);
    __syncthreads();

    double local_square = 0.0;
    for (int d = tid; d < D; d += block_size) {
        float s = token_embeddings[tok_base + d] + position_embeddings[pos_base + d];
        float difference = s - mean_value;
        local_square += (double)difference * difference;
    }
    shared[tid] = local_square;
    __syncthreads();
    for (int off = block_size / 2; off > 0; off >>= 1) {
        if (tid < off) shared[tid] += shared[tid + off];
        __syncthreads();
    }
    if (tid == 0) {
        float variance = (float)(shared[0] / (double)D);
        inv_value = rsqrtf(variance + eps);
    }
    __syncthreads();

    for (int d = tid; d < D; d += block_size) {
        float s = token_embeddings[tok_base + d] + position_embeddings[pos_base + d];
        output[out_base + d] = (s - mean_value) * inv_value * gamma[d] + beta[d];
    }
}

// token_ids, position_ids, token_embeddings, position_embeddings, gamma, beta, output are device
// pointers
extern "C" void solve(const int* token_ids, const int* position_ids, const float* token_embeddings,
                      const float* position_embeddings, const float* gamma, const float* beta,
                      float* output, int B, int T, int V, int P, int D, float eps) {
    (void)V;
    (void)P;
    int threads = 256;
    embedding_kernel<<<B * T, threads, threads * sizeof(double)>>>(
        token_ids, position_ids, token_embeddings, position_embeddings,
        gamma, beta, output, T, D, eps);
    cudaDeviceSynchronize();
}
```
