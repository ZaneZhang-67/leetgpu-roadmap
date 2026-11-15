---
id: paged-kv-cache-attention
leetgpuId: 86
titleEn: Paged KV-Cache Attention
titleZh: 分页 KV Cache 注意力
description: 通过逻辑块表访问离散物理 KV Cache 页面，为每个序列与注意力头计算解码阶段注意力
problemUrl: https://leetgpu.com/challenges/paged-kv-cache-attention
updatedAt: 2026-11-12
---
## 题目定义

每个序列只有一个 Query，形状为 `[batch_size,num_heads,head_dim]`。KV Cache 的物理布局为 `[num_blocks,block_size,num_heads,head_dim]`，逻辑 token `t` 的物理地址由下式决定：

```text
logical_block = t / block_size
offset = t % block_size
physical_block = block_table[sequence,logical_block]
```

只处理满足 `t<context_lens[sequence]` 的 token。score 按 `1/sqrt(head_dim)` 缩放，在有效上下文上执行 Softmax 后，再对 V 加权求和。

### 官方示例

示例中，`batch=1, heads=1, head_dim=4, block_size=2`，`context_lens=[2]`，`block_table=[[1]]`。逻辑块映射到物理块 1，物理块 0 是无关数据：

```text
Q = [1,1,0,0]
K0 = [1,0,0,0], K1 = [0,1,0,0]
V0 = [2,0,0,0], V1 = [0,4,0,0]
scores = [0.5,0.5]
output = [1,2,0,0]
```

### 完整约束

- `1 ≤ batch_size ≤ 32`，`1 ≤ num_heads ≤ 64`
- `1 ≤ head_dim ≤ 256`，且是 4 的倍数
- `1 ≤ block_size ≤ 64`，且为 2 的幂
- 每个 `context_lens[s]` 在 `[1,8192]`
- `max_blocks_per_seq=ceil(max(context_lens)/block_size)`
- Q/K/V/output 为 FP32，块表与上下文长度为 int32
- 有效块表项是任意散布的物理块索引；无效尾部和不可达池块必须忽略
- 性能测试为 `batch=8,heads=32,head_dim=128,block_size=16,context=2048`
- 容差为 `1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

extern "C" void solve(const float* Q, const float* K_cache,
                      const float* V_cache, const int* block_table,
                      const int* context_lens, float* output,
                      int batch_size, int num_heads, int head_dim,
                      int block_size, int max_blocks_per_seq) {}
```

`num_blocks` 不单独传入，寻址只能通过有效的 `block_table` 项完成。Cache 不能按序列连续读取；测试会随机散布每个序列的页面，并混入无关块。

## 实现思路

每个 Block 负责一个 `(sequence,head)`。动态 Shared Memory 保存该序列所有有效 token 的权重，上下文长度达到最大值 8192 时占用 32 KiB。

线程以 `blockDim.x` 为步长分摊逻辑 token，通过 `block_table` 找到物理块，再按实际布局计算各自的 score：

```text
(((physical_block*block_size+offset)*num_heads+head)*head_dim+d)
```

完成 score 计算和稳定 Softmax 后，线程按 feature 并行遍历同一组逻辑到物理映射，累加 V。

实现只访问 `context_lens[sequence]` 个 token，因此不会读取部分填充尾块中的 padding，也不会读取未被块表引用的池块。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void paged_attention(const float* Q, const float* K_cache,
                                const float* V_cache, const int* block_table,
                                const int* context_lens, float* output,
                                int num_heads, int head_dim, int block_size,
                                int max_blocks_per_seq) {
    extern __shared__ float weights[];
    int sequence = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    int context = context_lens[sequence];
    float scale = rsqrtf(static_cast<float>(head_dim));

    for (int token = tid; token < context; token += blockDim.x) {
        int logical = token / block_size;
        int offset = token % block_size;
        int physical = block_table[(size_t)sequence * max_blocks_per_seq + logical];
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            size_t qi = ((size_t)sequence * num_heads + head) * head_dim + d;
            size_t ki = (((size_t)physical * block_size + offset) * num_heads + head)
                        * head_dim + d;
            score += Q[qi] * K_cache[ki];
        }
        weights[token] = score * scale;
    }
    __syncthreads();

    if (tid == 0) {
        float maximum = -CUDART_INF_F;
        for (int token = 0; token < context; ++token) {
            maximum = fmaxf(maximum, weights[token]);
        }
        float denominator = 0.0f;
        for (int token = 0; token < context; ++token) {
            weights[token] = expf(weights[token] - maximum);
            denominator += weights[token];
        }
        for (int token = 0; token < context; ++token)
            weights[token] /= denominator;
    }
    __syncthreads();

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int token = 0; token < context; ++token) {
            int logical = token / block_size;
            int offset = token % block_size;
            int physical = block_table[(size_t)sequence * max_blocks_per_seq + logical];
            size_t vi = (((size_t)physical * block_size + offset) * num_heads + head)
                        * head_dim + d;
            sum += weights[token] * V_cache[vi];
        }
        output[(sequence * num_heads + head) * head_dim + d] = sum;
    }
}

extern "C" void solve(const float* Q, const float* K_cache,
                      const float* V_cache, const int* block_table,
                      const int* context_lens, float* output,
                      int batch_size, int num_heads, int head_dim,
                      int block_size, int max_blocks_per_seq) {
    dim3 grid(batch_size, num_heads);
    size_t shared = static_cast<size_t>(max_blocks_per_seq) * block_size
                    * sizeof(float);
    paged_attention<<<grid, 256, shared>>>(
        Q, K_cache, V_cache, block_table, context_lens, output,
        num_heads, head_dim, block_size, max_blocks_per_seq);
    cudaDeviceSynchronize();
}
```
