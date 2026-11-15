---
id: parallel-reverse-scan-gae
leetgpuId: 110
titleEn: Parallel Reverse Scan (GAE)
titleZh: 并行反向扫描（GAE）
description: 沿时间维保持 FP32 递推顺序，并行处理批内序列以计算广义优势估计
problemUrl: https://leetgpu.com/challenges/parallel-reverse-scan-gae
updatedAt: 2026-11-12
---
## 题目定义

对 `rewards(B,S)`、`values(B,S)` 计算 GAE，序列末尾的 next value 取零：

```text
delta[t] = reward[t] + gamma*next_value[t] - value[t]
A[S]=0
A[t] = delta[t] + (gamma*lambda)*A[t+1]
```

每条序列都在 S 处结束。题目没有提供 done、padding 或 mask 输入，也不应自行加入终止掩码。

### 示例

```text
rewards=[1,2,3,4]，values=[0.5,1,1.5,2]
gamma=0.9，lambda=0.5，decay=0.45
delta=[1.4,2.35,3.3,2.0]
advantages=[3.308,4.240,4.200,2.000]
```

约束为 `1≤B≤256`、`1≤S≤65536`，gamma、lambda 位于 `[0,1]`，所有张量均为 FP32。性能用例为 `B=64`、`S=4096`，容差为 `1e-3`。

## 模板代码解析

starter 签名为：

```text
extern "C" void solve(const float* rewards, const float* values, float* advantages,
                      float gamma, float lam, int B, int S)
```

数组按 `(B,S)` 行主序保存。gamma 同时用于 TD bootstrap 和反向衰减，反向衰减系数为 `gamma*lam`。

## 实现思路

时间维上存在严格依赖：`A[t]` 只有在 `A[t+1]` 产生后才能计算。递推在实数代数中虽可改写为仿射扫描，但 FP32 加法不满足结合律；树形扫描会改变舍入顺序，并在大数抵消输入下产生不同结果。

参考实现因此让一个线程负责一条 Batch 序列，严格从 `S-1` 递推到 0，并在 Batch 维并行。`__fmul_rn`、`__fadd_rn` 与 `__fsub_rn` 明确复现 runtime 中“乘、加、减”分步执行的 FP32 舍入顺序。性能用例共有 `64×4096` 个时间步，64 条依赖链可在一个 Block 中并行执行，无需额外工作区或跨 kernel carry。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void gae_reverse(const float* rewards, const float* values,
                            float* advantages, float gamma, float lam,
                            int B, int S) {
    int batch = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (batch >= B) return;

    size_t base = static_cast<size_t>(batch) * S;
    float decay = __fmul_rn(gamma, lam);
    float last = 0.0f;
    for (int t = S - 1; t >= 0; --t) {
        float next = t + 1 < S ? values[base + t + 1] : 0.0f;
        float bootstrap = __fmul_rn(gamma, next);
        float delta = __fsub_rn(
            __fadd_rn(rewards[base + t], bootstrap), values[base + t]);
        last = __fadd_rn(delta, __fmul_rn(decay, last));
        advantages[base + t] = last;
    }
}

extern "C" void solve(const float* rewards, const float* values, float* advantages,
                      float gamma, float lam, int B, int S) {
    constexpr int threads = 256;
    gae_reverse<<<(B + threads - 1) / threads, threads>>>(
        rewards, values, advantages, gamma, lam, B, S);
    cudaDeviceSynchronize();
}
```
