---
id: shared-prefix-cascade-attention
leetgpuId: 88
titleEn: Shared-Prefix Cascade Attention
titleZh: 共享前缀级联注意力
description: 合并共享前缀与每序列因果后缀的 Softmax 统计量，计算数值稳定的级联注意力
problemUrl: https://leetgpu.com/challenges/shared-prefix-cascade-attention
updatedAt: 2026-11-12
---
## 题目定义

批内所有序列共享 `K_prefix,V_prefix`，每条序列另有自己的 `Q,K_suffix,V_suffix`。对于后缀 Query `i`，可见 Key 包括全部 `prefix_len` 个前缀和本序列后缀的 `0..i`；其他序列的后缀及未来位置均不可见。

```text
scores = q · [K_prefix; K_suffix[0..i]] / sqrt(head_dim)
output = softmax(scores) · [V_prefix; V_suffix[0..i]]
```

### 官方示例

```text
B=2, H=1, prefix_len=2, suffix_len=2, head_dim=2
K_prefix=[[1,0],[0,1]]
V_prefix=[[1,2],[3,4]]
output sequence 0=[[3,4],[4,5]]
output sequence 1=[[1,1.3333],[0.75,1]]
```

### 完整约束

- `1 ≤ batch_size,num_heads ≤ 64`
- `0 ≤ prefix_len ≤ 8192`
- `1 ≤ suffix_len ≤ 1024`
- `1 ≤ head_dim ≤ 128`
- 所有张量为 FP32，绝对与相对容差均为 `1e-4`
- prefix 可以为 0，此时退化为普通后缀 causal attention
- 性能测试为 `B=16,H=32,P=4096,L=256,D=128`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

extern "C" void solve(const float* Q, const float* K_prefix,
                      const float* V_prefix, const float* K_suffix,
                      const float* V_suffix, float* output,
                      int batch_size, int num_heads, int prefix_len,
                      int suffix_len, int head_dim) {}
```

共享前缀的形状为 `[H,P,D]`，其余张量的形状为 `[B,H,L,D]`。实现无需真正拼接 K/V，也无需构造 `[B,H,L,P+L]` 的 score 张量。

## 实现思路

每个 Block 负责一个 `(batch,head,query)`。动态 Shared Memory 保存当前 Query 可见的全部 score，最多容纳 `prefix_len+suffix_len=9216` 个 FP32，即 36 KiB。线程并行计算共享前缀和当前序列可见后缀的 score；同步后，线程 0 完成稳定 Softmax，各线程再按 feature 并行累加 Value。

两段 Softmax 可以通过 log-sum-exp 稳定合并。设前缀统计为 `(m_p,l_p)`，后缀统计为 `(m_s,l_s)`：

```text
m = max(m_p,m_s)
L = exp(m_p-m)*l_p + exp(m_s-m)*l_s
output = [exp(m_p-m)*weighted_prefix +
          exp(m_s-m)*weighted_suffix] / L
```

参考实现将两段 score 连续放入 Shared Memory，并使用共同的最大值 `m` 统一归一化，在代数上等价于上述合并公式。`prefix_len=0` 时，前缀部分直接跳过，最大值和分母只由后缀产生。每个 Query 至少能看见自身位置，因此分母始终非零。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void cascade_attention(const float* Q, const float* K_prefix,
                                  const float* V_prefix, const float* K_suffix,
                                  const float* V_suffix, float* output,
                                  int num_heads, int prefix_len,
                                  int suffix_len, int head_dim) {
    int query = blockIdx.x;
    int head = blockIdx.y;
    int batch = blockIdx.z;
    int tid = threadIdx.x;
    float scale = rsqrtf(static_cast<float>(head_dim));
    int qbase = ((batch * num_heads + head) * suffix_len + query) * head_dim;

    extern __shared__ float weights[];
    int visible = prefix_len + query + 1;
    for (int index = tid; index < visible; index += blockDim.x) {
        bool is_prefix = index < prefix_len;
        int key = is_prefix ? index : index - prefix_len;
        int kbase = is_prefix
            ? (head * prefix_len + key) * head_dim
            : ((batch * num_heads + head) * suffix_len + key) * head_dim;
        const float* keys = is_prefix ? K_prefix : K_suffix;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += Q[qbase + d] * keys[kbase + d];
        }
        weights[index] = score * scale;
    }
    __syncthreads();

    if (tid == 0) {
        float maximum = -CUDART_INF_F;
        for (int index = 0; index < visible; ++index) {
            maximum = fmaxf(maximum, weights[index]);
        }
        float denominator = 0.0f;
        for (int index = 0; index < visible; ++index) {
            weights[index] = expf(weights[index] - maximum);
            denominator += weights[index];
        }
        for (int index = 0; index < visible; ++index) {
            weights[index] /= denominator;
        }
    }
    __syncthreads();

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int key = 0; key < prefix_len; ++key) {
            int vbase = (head * prefix_len + key) * head_dim;
            sum += weights[key] * V_prefix[vbase + d];
        }
        for (int key = 0; key <= query; ++key) {
            int vbase = ((batch * num_heads + head) * suffix_len + key) * head_dim;
            sum += weights[prefix_len + key] * V_suffix[vbase + d];
        }
        output[qbase + d] = sum;
    }
}

extern "C" void solve(const float* Q, const float* K_prefix,
                      const float* V_prefix, const float* K_suffix,
                      const float* V_suffix, float* output,
                      int batch_size, int num_heads, int prefix_len,
                      int suffix_len, int head_dim) {
    dim3 grid(suffix_len, num_heads, batch_size);
    size_t shared = static_cast<size_t>(prefix_len + suffix_len) * sizeof(float);
    cascade_attention<<<grid, 128, shared>>>(
        Q, K_prefix, V_prefix, K_suffix, V_suffix, output,
        num_heads, prefix_len, suffix_len, head_dim);
    cudaDeviceSynchronize();
}
```
