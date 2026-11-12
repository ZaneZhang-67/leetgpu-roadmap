---
id: grouped-query-attention
leetgpuId: 80
titleEn: Grouped Query Attention
titleZh: 分组查询注意力
description: 让连续的 Query 头分组共享较少的 Key/Value 头，并计算无掩码缩放点积注意力
problemUrl: https://leetgpu.com/challenges/grouped-query-attention
updatedAt: 2026-11-12
---
## 题目定义

`Q` 的形状为 `[num_q_heads,seq_len,head_dim]`，`K`、`V` 的形状为 `[num_kv_heads,seq_len,head_dim]`。每组连续的 Query 头共享一个 KV 头：

```text
groups = num_q_heads / num_kv_heads
kv_head = q_head / groups
scores = Q[q_head] @ K[kv_head]ᵀ / sqrt(head_dim)
output[q_head] = softmax(scores) @ V[kv_head]
```

runtime reference 不使用 causal mask，因此每个 Query 位置都能关注对应 KV 头的全部 `seq_len` 个位置。

### 官方示例

示例使用 `num_q_heads=4`、`num_kv_heads=2`、`seq_len=3`、`head_dim=4`，因此 `Q0,Q1` 共享 `K0,V0`，`Q2,Q3` 共享 `K1,V1`。官方输出如下：

```text
output[0] = [[5.71,6.71,7.71,8.71],
             [5.71,6.71,7.71,8.71],
             [5.71,6.71,7.71,8.71]]
output[1] = [[6.07,7.07,8.07,9.07],
             [5.00,6.00,7.00,8.00],
             [5.71,6.71,7.71,8.71]]
output[2] = [[2.24,2.76,3.27,3.79],
             [3.96,4.70,5.44,6.17],
             [2.40,2.60,2.79,2.98]]
output[3] = [[0.76,0.58,0.40,0.22],
             [1.17,1.08,1.00,0.91],
             [2.84,3.37,3.91,4.44]]
```

### 完整约束

- `1 ≤ num_kv_heads ≤ num_q_heads ≤ 64`
- `num_q_heads` 能被 `num_kv_heads` 整除
- `1 ≤ seq_len ≤ 4096`
- 题面写 `8 ≤ head_dim ≤ 256` 且为 8 的倍数；runtime 示例与功能测试还会传入 `head_dim=4`，实现也必须覆盖
- 所有张量均为 FP32
- 性能测试为 `32` 个 Q 头、`8` 个 KV 头、`seq_len=1024`、`head_dim=128`
- 判定容差为 `1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int num_q_heads, int num_kv_heads,
                      int seq_len, int head_dim) {}
```

四个数组都按 `[head,position,feature]` 的行主序展开。平台只传入原始 KV 头，不会预先执行 `repeat_interleave`。

## 实现思路

每个 Block 负责一个 `(q_head,query_position)`。线程以 `blockDim.x` 为步长分摊 Key，各自计算对应的点积，并把缩放后的 score 写入动态 Shared Memory。同步后，线程 0 对最多 4096 个 score 执行减最大值 Softmax；再次同步后，各线程按输出 feature 并行累加 Value。

KV 映射必须使用整数分组 `q_head/(num_q_heads/num_kv_heads)`，而不是取模。例如，8 个 Q 头、2 个 KV 头时，Q 头 0–3 映射到 KV 头 0，Q 头 4–7 映射到 KV 头 1。

Shared Memory 大小为 `seq_len*sizeof(float)`，当 `seq_len=4096` 时最多占用 16 KiB。计算复杂度为 `O(num_q_heads*seq_len²*head_dim)`。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void gqa_kernel(const float* Q, const float* K, const float* V,
                           float* output, int num_q_heads, int num_kv_heads,
                           int seq_len, int head_dim) {
    extern __shared__ float weights[];
    int query = blockIdx.x;
    int q_head = blockIdx.y;
    int tid = threadIdx.x;
    int groups = num_q_heads / num_kv_heads;
    int kv_head = q_head / groups;
    float scale = rsqrtf(static_cast<float>(head_dim));

    for (int key = tid; key < seq_len; key += blockDim.x) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            int qi = (q_head * seq_len + query) * head_dim + d;
            int ki = (kv_head * seq_len + key) * head_dim + d;
            score += Q[qi] * K[ki];
        }
        weights[key] = score * scale;
    }
    __syncthreads();

    if (tid == 0) {
        float maximum = -CUDART_INF_F;
        for (int key = 0; key < seq_len; ++key) {
            maximum = fmaxf(maximum, weights[key]);
        }
        float denominator = 0.0f;
        for (int key = 0; key < seq_len; ++key) {
            weights[key] = expf(weights[key] - maximum);
            denominator += weights[key];
        }
        for (int key = 0; key < seq_len; ++key) weights[key] /= denominator;
    }
    __syncthreads();

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int key = 0; key < seq_len; ++key) {
            int vi = (kv_head * seq_len + key) * head_dim + d;
            sum += weights[key] * V[vi];
        }
        output[(q_head * seq_len + query) * head_dim + d] = sum;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int num_q_heads, int num_kv_heads,
                      int seq_len, int head_dim) {
    dim3 grid(seq_len, num_q_heads);
    gqa_kernel<<<grid, 256, static_cast<size_t>(seq_len) * sizeof(float)>>>(
        Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim);
    cudaDeviceSynchronize();
}
```
