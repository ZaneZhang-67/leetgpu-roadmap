---
id: fused-adamw
leetgpuId: 127
titleEn: Fused AdamW
titleZh: 融合 AdamW
description: 使用单个 CUDA kernel 原地完成解耦权重衰减、矩估计和偏差修正更新
problemUrl: https://leetgpu.com/challenges/fused-adamw
updatedAt: 2026-11-12
---
## 题目定义

对长度为 `N` 的 FP32 参数、梯度以及一阶和二阶矩执行一次 AdamW 更新：

```text
params = params * (1-lr*weight_decay)
m = beta1*m + (1-beta1)*grad
v = beta2*v + (1-beta2)*grad²
m_hat = m/(1-beta1^t), v_hat = v/(1-beta2^t)
params = params - lr*m_hat/(sqrt(v_hat)+eps)
```

params、m、v 原地更新。权重衰减直接作用于更新前的 params，不并入 grad，并且发生在自适应步之前。

### 示例

```text
params=[1,-1], grad=[0.1,-0.1], m=v=[0,0]
lr=0.1,beta1=0.9,beta2=0.999,eps=1e-8,weight_decay=0,t=1
m_new=[0.01,-0.01], v_new=[1e-5,1e-5]
params=[0.9,-0.9]
```

**约束与测试**

- `1≤N≤16,777,216`，`1≤t≤100,000`
- params 与 grad 的值位于 `[-10,10]`，v 非负，m 可以为负
- `0<lr≤1`，`0≤beta1,beta2<1`，`1e-12≤eps≤1e-2`，`0≤weight_decay≤1`
- 测试覆盖零梯度、热状态、非默认 beta、大 eps 和大 t
- 性能测试 `N=16,777,216,t=1000`，容差 `atol=rtol=1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// params, grad, m, v are device pointers
extern "C" void solve(float* params, const float* grad, float* m, float* v, int N, float lr,
                      float beta1, float beta2, float eps, float weight_decay, int t) {}
```

官方 starter 只给出空入口。params、m、v 是读写缓冲区，grad 只读；没有独立输出。

## 实现思路

各元素完全独立，一个线程处理一个位置。偏差修正只依赖标量超参数，在 host 端用 `powf` 计算后传入 kernel，避免每个线程重复求幂。kernel 先保存旧 params，再按运行时顺序更新 m、v，最后用衰减后的 params 减去自适应步。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void adamw_kernel(float* params, const float* grad, float* m, float* v,
                             size_t N, float lr, float beta1, float beta2, float eps,
                             float weight_decay, float bc1, float bc2) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (; i < N; i += stride) {
        float g = grad[i];
        float mn = beta1 * m[i] + (1.0f - beta1) * g;
        float vn = beta2 * v[i] + (1.0f - beta2) * g * g;
        float p = params[i] * (1.0f - lr * weight_decay);
        m[i] = mn; v[i] = vn;
        params[i] = p - lr * (mn / bc1) / (sqrtf(vn / bc2) + eps);
    }
}

// params, grad, m, v are device pointers
extern "C" void solve(float* params, const float* grad, float* m, float* v, int N, float lr,
                      float beta1, float beta2, float eps, float weight_decay, int t) {
    float bc1 = 1.0f - powf(beta1, (float)t);
    float bc2 = 1.0f - powf(beta2, (float)t);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 4096) blocks = 4096;
    adamw_kernel<<<blocks, threads>>>(params, grad, m, v, (size_t)N, lr,
                                      beta1, beta2, eps, weight_decay, bc1, bc2);
    cudaDeviceSynchronize();
}
```
