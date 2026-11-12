---
id: beam-search-step
leetgpuId: 98
titleEn: Beam Search Step
titleZh: Beam Search 单步扩展
description: 使用 CUDA 并行扫描候选并选出每个 Batch 的 Top-K Beam
problemUrl: https://leetgpu.com/challenges/beam-search-step
updatedAt: 2026-11-12
---
## 题目定义

每个 Batch 包含 K 个活跃 Beam，词表大小为 V。候选 `(k,v)` 的分数为：

```text
candidate[b,k,v] = beam_scores[b,k] + token_logprobs[b,k,v]
```

从每个 Batch 的 `K×V` 个候选中选出分数最高的 K 个，按降序写入 `new_beam_scores`，同时写出来源 `parent_beam_indices` 和 Token 编号 `next_tokens`。官方测试保证同一 Batch 内的候选分数互不相同，因此无需另行定义 tie 规则。

### 示例

```text
beam_scores=[[-0.5,-1.0]]
token_logprobs=[[[-0.3,-1.2,-2.0,-2.5],[-0.4,-0.1,-1.6,-3.2]]]
new_beam_scores=[[-0.8,-1.1]]
parent_beam_indices=[[0,1]]
next_tokens=[[0,1]]
```

约束为 `1≤B≤64`、`1≤K≤32`、`1≤V≤100000`。性能用例为 `B=16`、`K=8`、`V=50000`。

## 模板代码解析

starter 签名为：

```text
extern "C" void solve(const float* beam_scores, const float* token_logprobs,
                      float* new_beam_scores, int* parent_beam_indices, int* next_tokens,
                      int B, int K, int V)
```

`token_logprobs` 按 `(B,K,V)` 行主序存放。扁平候选索引 `index=k*V+v`，可还原为 `k=index/V`、`v=index%V`。

## 实现思路

一个 Block 负责一个 Batch。每个线程以 Grid-stride 方式扫描候选并维护局部 Top-K，线程 0 再汇总所有结果。每个 Block 使用 128 个线程，合并后得到全局 Top-K。K 最大为 32，因此 Shared Memory 保存 `128×32` 个 `(score,index)`，共 32 KiB，未超过常见设备默认的单 Block Shared Memory 限额。

候选无需预先物化，读取时直接加上父 Beam 分数。局部插入和最终合并都保持分数降序。大词表下的候选计算可以并行完成，避免让单线程扫描 `K×V`。

## 参考实现

```cpp
#include <cfloat>
#include <cuda_runtime.h>

__global__ void beam_topk(const float* beam, const float* probs, float* out,
                          int* parents, int* tokens, int B, int K, int V) {
    extern __shared__ unsigned char raw[];
    float* scores = reinterpret_cast<float*>(raw);
    int block_size = static_cast<int>(blockDim.x);
    int* indices = reinterpret_cast<int*>(scores + block_size * K);
    int b = static_cast<int>(blockIdx.x);
    int tid = static_cast<int>(threadIdx.x);
    if (b >= B) return;
    float* local_scores = scores + tid * K;
    int* local_indices = indices + tid * K;
    for (int j = 0; j < K; ++j) { local_scores[j] = -FLT_MAX; local_indices[j] = -1; }

    int total = K * V;
    for (int index = tid; index < total; index += block_size) {
        int parent = index / V;
        float value = beam[b * K + parent] + probs[b * total + index];
        if (value > local_scores[K - 1]) {
            int pos = K - 1;
            while (pos > 0 && value > local_scores[pos - 1]) {
                local_scores[pos] = local_scores[pos - 1];
                local_indices[pos] = local_indices[pos - 1];
                --pos;
            }
            local_scores[pos] = value;
            local_indices[pos] = index;
        }
    }
    __syncthreads();

    if (tid == 0) {
        float best[32];
        int best_index[32];
        for (int j = 0; j < K; ++j) { best[j] = -FLT_MAX; best_index[j] = -1; }
        for (int t = 0; t < block_size; ++t) {
            for (int j = 0; j < K; ++j) {
                float value = scores[t * K + j];
                if (value <= best[K - 1]) continue;
                int pos = K - 1;
                while (pos > 0 && value > best[pos - 1]) {
                    best[pos] = best[pos - 1]; best_index[pos] = best_index[pos - 1]; --pos;
                }
                best[pos] = value; best_index[pos] = indices[t * K + j];
            }
        }
        for (int j = 0; j < K; ++j) {
            out[b * K + j] = best[j];
            parents[b * K + j] = best_index[j] / V;
            tokens[b * K + j] = best_index[j] % V;
        }
    }
}

extern "C" void solve(const float* beam_scores, const float* token_logprobs,
                      float* new_beam_scores, int* parent_beam_indices, int* next_tokens,
                      int B, int K, int V) {
    constexpr int threads = 128;
    size_t shared = static_cast<size_t>(threads) * K * (sizeof(float) + sizeof(int));
    beam_topk<<<B, threads, shared>>>(beam_scores, token_logprobs, new_beam_scores,
                                      parent_beam_indices, next_tokens, B, K, V);
    cudaDeviceSynchronize();
}
```
