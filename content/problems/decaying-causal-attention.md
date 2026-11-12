---
id: decaying-causal-attention
leetgpuId: 92
titleEn: Decaying Causal Attention
titleZh: 衰减因果注意力
description: 使用 CUDA 计算带几何距离衰减且不经过 softmax 的因果注意力
problemUrl: https://leetgpu.com/challenges/decaying-causal-attention
updatedAt: 2026-11-12
---
## 题目定义

`Q`、`K`、`V` 均为形状 `(seq_len,d_model)` 的 FP32 行主序矩阵。位置 `n` 只能读取当前位置及过去位置 `m≤n`：

```text
output[n,d] = Σ gamma^(n-m) × (Q[n]·K[m]/sqrt(d_model)) × V[m,d]
              0≤m≤n
```

这里没有 softmax 或其他归一化。当前 token 的衰减因子为 1，距离每增加 1，因子再乘一次 `gamma`；未来位置完全屏蔽。

### 示例

官方示例取 `seq_len=2,d_model=4,gamma=0.5`。注意力分数、因果衰减矩阵及输出为：

```text
QK^T/sqrt(4) = [[0.5,0.5], [0.5,0.5]]
decay         = [[1.0,0.0], [0.5,1.0]]
weighted      = [[0.5,0.0], [0.25,0.5]]
output        = [[2,4,6,8], [3,6,9,12]]
```

**约束与测试**

- `1≤seq_len≤8,192`，`1≤d_model≤256`，`0<gamma≤1`
- 所有张量均为 GPU 上的 FP32
- 测试覆盖单 token、`gamma=1`、`gamma=0.1`、零 `Q/K`、全负值和非二次幂序列长度
- 性能测试为 `seq_len=4096,d_model=64`，绝对与相对误差容限均为 `1e-3`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int seq_len,
                      int d_model, float gamma) {}
```

starter 只提供空的 `solve`。四个矩阵均按行主序存储，`gamma` 按值传入；kernel 与启动逻辑需要自行实现。

## 实现思路

运行时先计算 `QKᵀ/sqrt(d_model)`，再乘衰减 mask，最后与 `V` 相乘。参考实现保留两次矩阵乘法的结合顺序，分两个 kernel 完成。

第一阶段由每个线程处理一个 `(n,m)`。当 `m≤n` 时，线程按 `k=0..d_model-1` 的顺序用 FP32 累加 `Q[n,k]×K[m,k]`，再乘 `1/sqrt(d_model)` 和 `gamma^(n-m)`，写入 `scores[n,m]`；未来位置写零。第二阶段由每个线程处理一个 `(n,d)`，按 `m=0..n` 的顺序用 FP32 累加 `scores[n,m]×V[m,d]`。这一实现与运行时先算 `QKᵀ`、再乘 `V` 的顺序一致，全程不使用 softmax。

该运算不能重关联为 `Q×Σ(K⊗V)`。一个合法的 FP32 反例是 `seq_len=2,d_model=2,gamma=1`：令第二行 `Q=[1,1]`，`K[0]=[33554432,-33554432]`、`K[1]=[1,1]`，两行 `V` 的第一维均为 1。运行时先计算两个点积，得到 0 和 2，缩放后输出 `sqrt(2)`。状态递推会在大数上丢失两个 `+1`，最终得到 0，误差远超 `1e-3`。

两阶段实现的时间复杂度为 `O(seq_len²×d_model)`；逐输出重算每个点积则需要 `O(seq_len²×d_model²)`，多出一维。正式性能规模约执行 10.7 亿次 FMA，但 `(n,m)` 和 `(n,d)` 都可以高度并行。`scores` 工作区包含 `seq_len²` 个 FP32：性能规模 `4096` 约占 64 MiB，题面上限 `8192` 约占 256 MiB。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void score_kernel(const float* Q, const float* K,
                             float* scores, size_t total,
                             int seq_len, int d_model, float gamma) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    float scale = rsqrtf((float)d_model);
    for (; index < total; index += stride) {
        int n = (int)(index / (size_t)seq_len);
        int m = (int)(index % (size_t)seq_len);
        if (m > n) {
            scores[index] = 0.0f;
            continue;
        }
        size_t q_base = (size_t)n * d_model;
        size_t k_base = (size_t)m * d_model;
        float dot = 0.0f;
        for (int k = 0; k < d_model; ++k)
            dot += Q[q_base + (size_t)k] * K[k_base + (size_t)k];
        scores[index] = dot * scale * powf(gamma, (float)(n - m));
    }
}

__global__ void output_kernel(const float* scores, const float* V,
                              float* output, size_t total,
                              int seq_len, int d_model) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (; index < total; index += stride) {
        int n = (int)(index / (size_t)d_model);
        int d = (int)(index % (size_t)d_model);
        size_t score_base = (size_t)n * seq_len;
        float result = 0.0f;
        for (int m = 0; m <= n; ++m) {
            size_t v_index = (size_t)m * d_model + (size_t)d;
            result += scores[score_base + (size_t)m] * V[v_index];
        }
        output[index] = result;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int seq_len,
                      int d_model, float gamma) {
    size_t score_count = (size_t)seq_len * seq_len;
    size_t output_count = (size_t)seq_len * d_model;
    float* scores;
    cudaMalloc(&scores, score_count * sizeof(float));

    int threads = 256;
    size_t score_needed = (score_count + threads - 1) / threads;
    size_t output_needed = (output_count + threads - 1) / threads;
    int score_blocks = score_needed > 4096 ? 4096 : (int)score_needed;
    int output_blocks = output_needed > 4096 ? 4096 : (int)output_needed;

    score_kernel<<<score_blocks, threads>>>(
        Q, K, scores, score_count, seq_len, d_model, gamma);
    output_kernel<<<output_blocks, threads>>>(
        scores, V, output, output_count, seq_len, d_model);
    cudaDeviceSynchronize();
    cudaFree(scores);
}
```
