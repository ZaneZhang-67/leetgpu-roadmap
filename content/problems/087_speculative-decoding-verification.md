---
id: speculative-decoding-verification
leetgpuId: 87
titleEn: Speculative Decoding Verification
titleZh: 推测解码验证
description: 使用 CUDA 完成草稿 token 的顺序接受、拒绝重采样与全接受奖励采样
problemUrl: https://leetgpu.com/challenges/speculative-decoding-verification
updatedAt: 2026-11-12
---
## 题目定义

对 `B` 条序列逐条验证 `T` 个草稿 token。位置 `i` 的草稿 token 为 `t`，其接受概率为：

```text
alpha = min(1, target_probs[b,i,t] / draft_probs[b,i,t])
```

若 `uniform_samples[b,i] < alpha`，就接受该 token 并继续。首次拒绝后立即停止验证，从 `max(0,target_probs-draft_probs)` 归一化后的分布中，用同一条序列的 `uniform_samples[b,T]` 做逆 CDF 采样。若差分全零，则回退到 `V` 个 token 的均匀分布。若 `T` 个 token 全部接受，则用同一个末尾随机数从 `target_probs[b,T-1,:]` 采样奖励 token。

输出形状为 `(B,T+1)`。接受或重采样位置之后均保持零。

### 示例

官方示例取 `B=1,T=3,V=4`、`draft_tokens=[1,2,0]`。位置 0 的 `alpha=0.50/0.60≈0.833`，随机数 `0.50` 使 token 1 被接受。位置 1 的 `alpha=0.20/0.50=0.40`，随机数 `0.70` 导致拒绝。差分归一化后的 CDF 为 `[0.667,0.667,0.667,1.0]`，末尾随机数 `0.90` 最终采得 token 3，因此：

```text
output_tokens = [1,3,0,0]
```

**约束与测试**

- `1≤B≤256`，`1≤T≤16`，`2≤V≤131,072`
- 概率张量为 FP32 且每行非负、和为 1；token 张量为 INT32
- 草稿 token 对应的草稿概率始终大于零；随机数位于 `[0,1)`
- 测试覆盖首位置拒绝、全部接受后的奖励采样、非二次幂词表和 `T=1`
- 性能测试为 `B=64,T=8,V=32,768`，绝对与相对误差容限均为 `1e-5`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens are device pointers
extern "C" void solve(const int* draft_tokens, const float* draft_probs, const float* target_probs,
                      const float* uniform_samples, int* output_tokens, int B, int T, int V) {}
```

概率数组按行主序存储，`draft_probs[b,i,v]` 的偏移为 `b*T*V+i*V+v`。starter 只提供空入口，初始化、验证和采样都需要自行实现。

## 实现思路

单条序列的接受过程存在严格的先后依赖，但 `B` 条序列彼此独立，因此每个 CUDA 线程处理一条序列。线程先将本行 `T+1` 个输出清零，再从左到右验证草稿 token。

接受概率需要复现 runtime 的 `.item()` 行为：先把 FP32 的 `p`、`q` 和随机数提升为双精度，再进行计算和比较。首次拒绝后，对整个词表扫描两遍。第一遍计算 `sum(max(q-p,0))`；第二遍逐项累加归一化后的 `max(q-p,0)/sum`，找到第一个满足 `cdf>=r` 的 token。这个顺序与 reference 的归一化、`cumsum` 和 `searchsorted` 一致。若差分总和为零，则按 FP32 的 `1/V` 逐项构造均匀 CDF。对于题目给定的有效概率分布，这只是为保留 reference 而设置的防御分支。

全部接受时，按顺序累加最后一个 target 分布，找到第一个满足 `cdf>=r` 的 token。两条采样路径都会把结果限制在 `[0,V-1]`。接受条件必须严格使用 `<`，逆 CDF 条件则使用 `>=`。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void verify_kernel(const int* draft_tokens, const float* draft_probs,
                              const float* target_probs, const float* uniform_samples,
                              int* output_tokens, int B, int T, int V) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    int out_base = b * (T + 1);
    for (int i = 0; i <= T; ++i) output_tokens[out_base + i] = 0;
    float sample = uniform_samples[out_base + T];
    for (int i = 0; i < T; ++i) {
        int tok = draft_tokens[b * T + i];
        size_t base = ((size_t)b * T + i) * V;
        double p = (double)draft_probs[base + tok];
        double q = (double)target_probs[base + tok];
        double alpha = q / p;
        if (alpha > 1.0) alpha = 1.0;
        if ((double)uniform_samples[out_base + i] < alpha) {
            output_tokens[out_base + i] = tok;
            continue;
        }
        float total = 0.0f;
        for (int v = 0; v < V; ++v) {
            float d = target_probs[base + v] - draft_probs[base + v];
            if (d > 0.0f) total += d;
        }
        int chosen = V - 1;
        if (total > 0.0f) {
            float cdf = 0.0f;
            for (int v = 0; v < V; ++v) {
                float d = target_probs[base + v] - draft_probs[base + v];
                if (d > 0.0f) cdf += d / total;
                if (cdf >= sample) { chosen = v; break; }
            }
        } else {
            float probability = 1.0f / V;
            float cdf = 0.0f;
            for (int v = 0; v < V; ++v) {
                cdf += probability;
                if (cdf >= sample) { chosen = v; break; }
            }
        }
        output_tokens[out_base + i] = chosen;
        return;
    }
    size_t base = ((size_t)b * T + (T - 1)) * V;
    float cdf = 0.0f;
    int chosen = V - 1;
    for (int v = 0; v < V; ++v) {
        cdf += target_probs[base + v];
        if (cdf >= sample) { chosen = v; break; }
    }
    output_tokens[out_base + T] = chosen;
}

// draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens are device pointers
extern "C" void solve(const int* draft_tokens, const float* draft_probs, const float* target_probs,
                      const float* uniform_samples, int* output_tokens, int B, int T, int V) {
    int threads = 128;
    verify_kernel<<<(B + threads - 1) / threads, threads>>>(
        draft_tokens, draft_probs, target_probs, uniform_samples,
        output_tokens, B, T, V);
    cudaDeviceSynchronize();
}
```
