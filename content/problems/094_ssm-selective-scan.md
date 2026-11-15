---
id: ssm-selective-scan
leetgpuId: 94
titleEn: SSM Selective Scan
titleZh: SSM 选择性扫描
description: 使用 CUDA 按序列顺序计算输入相关离散化、隐藏状态递推、状态投影和残差分支
problemUrl: https://leetgpu.com/challenges/ssm-selective-scan
updatedAt: 2026-11-12
---
## 题目定义

各张量的布局如下：`u,delta,y:[batch,seq_len,d_model]`，`A:[d_model,d_state]`，`B,C:[batch,seq_len,d_state]`，`skip:[d_model]`。隐藏状态 `h:[batch,d_model,d_state]` 从零开始，按时间顺序计算：

```text
A_bar = exp(delta[b,t,d] × A[d,n])
B_bar = delta[b,t,d] × B[b,t,n]
h[b,d,n] = A_bar × h[b,d,n] + B_bar × u[b,t,d]
y[b,t,d] = Σ_n C[b,t,n] × h[b,d,n] + skip[d] × u[b,t,d]
```

`B` 和 `C` 会随 batch 与时间变化，但不含 d_model 维；`delta` 则随每个模型通道变化。

### 实现要求

- 保持 `solve` 签名不变，只使用原生功能
- 时间维必须按 `t=0..seq_len-1` 递推
- 最终结果写入 `y`

### 示例

官方示例设置 `batch=1,seq_len=4,d_model=2,d_state=2`，其中 `delta` 全为 1，`A` 的两行为 `[-0.5,-1]`，同时给出随时间变化的 `B/C` 和全零的 `skip`。

### 约束

- `1 ≤ batch ≤ 16`
- `1 ≤ seq_len ≤ 8192`
- `1 ≤ d_model ≤ 2048`
- `1 ≤ d_state ≤ 64`
- 功能测试覆盖单 token、零输入、零 delta、非 2 的幂长度和典型 `d_state=16`
- 所有 `delta` 严格为正，所有 `A` 严格为负
- 性能测试为 `batch=4,seq_len=4096,d_model=512,d_state=16`
- 绝对误差和相对误差均为 `1e-3`

## 模板代码解析

```cpp
#include <cuda_runtime.h>
#include <math.h>

// u, delta, A, B, C, skip, y are device pointers
extern "C" void solve(
    const float* u, const float* delta, const float* A,
    const float* B, const float* C, const float* skip,
    float* y, int batch, int seq_len,
    int d_model, int d_state
) {}
```

starter 只给出入口，没有提供状态缓冲区或 kernel。所有输入都按上述行主序连续存储。

## 实现思路

一个 Warp 负责一个 `(batch,d_model)` 通道，32 个 lane 沿 `d_state` 并行。由于 `d_state≤64`，每个 lane 最多保存两个状态标量。时间维仍严格按顺序推进：每一步先更新本 lane 的状态，再用 Warp Shuffle 归约 C 投影，最后由 lane 0 加上 skip 残差并写回。

性能规模共有 2048 个通道，对应 2048 个 Warp、256 个 Block，避免退化成只有 8 个 Block 的低占用映射。状态更新显式使用 `__fmul_rn` 和 `__fadd_rn`，保留 runtime 中 `A_bar*h`、`B_bar*u` 先分别相乘、再相加的 FP32 舍入顺序；残差同样先乘后加。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void selective_scan_kernel(
    const float* u, const float* delta, const float* A,
    const float* B, const float* C, const float* skip,
    float* y, int batch, int seq_len,
    int d_model, int d_state
) {
    int global_thread = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int channel = global_thread >> 5;
    int lane = global_thread & 31;
    int total = batch * d_model;
    if (channel >= total) return;

    int b = channel / d_model;
    int d = channel % d_model;
    int n0 = lane;
    int n1 = lane + 32;
    float state0 = 0.0f;
    float state1 = 0.0f;

    for (int t = 0; t < seq_len; ++t) {
        int ud = (b * seq_len + t) * d_model + d;
        int sn = (b * seq_len + t) * d_state;
        float dt = delta[ud];
        float input = u[ud];
        float contribution = 0.0f;

        if (n0 < d_state) {
            float a_bar = expf(__fmul_rn(dt, A[d * d_state + n0]));
            float b_bar = __fmul_rn(dt, B[sn + n0]);
            state0 = __fadd_rn(__fmul_rn(a_bar, state0),
                               __fmul_rn(b_bar, input));
            contribution = __fmul_rn(C[sn + n0], state0);
        }
        if (n1 < d_state) {
            float a_bar = expf(__fmul_rn(dt, A[d * d_state + n1]));
            float b_bar = __fmul_rn(dt, B[sn + n1]);
            state1 = __fadd_rn(__fmul_rn(a_bar, state1),
                               __fmul_rn(b_bar, input));
            contribution = __fadd_rn(
                contribution, __fmul_rn(C[sn + n1], state1));
        }

        for (int offset = 16; offset > 0; offset >>= 1)
            contribution = __fadd_rn(
                contribution, __shfl_down_sync(0xffffffffu, contribution, offset));
        if (lane == 0) {
            float residual = __fmul_rn(skip[d], input);
            y[ud] = __fadd_rn(contribution, residual);
        }
    }
}

extern "C" void solve(
    const float* u, const float* delta, const float* A,
    const float* B, const float* C, const float* skip,
    float* y, int batch, int seq_len,
    int d_model, int d_state
) {
    constexpr int threads = 256;
    int warps = batch * d_model;
    int total_threads = warps * 32;
    selective_scan_kernel<<<(total_threads + threads - 1) / threads, threads>>>(
        u, delta, A, B, C, skip, y,
        batch, seq_len, d_model, d_state
    );
    cudaDeviceSynchronize();
}
```
