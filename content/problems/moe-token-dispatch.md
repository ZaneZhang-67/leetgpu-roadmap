---
id: moe-token-dispatch
leetgpuId: 91
titleEn: MoE Token Dispatch
titleZh: MoE Token 分发
description: 使用 CUDA 按专家稳定分组 Token，并生成每个专家的 Token 数量
problemUrl: https://leetgpu.com/challenges/moe-token-dispatch
updatedAt: 2026-11-12
---
## 题目定义

给定 Token 特征 `x(T,D)` 和每个 Token 唯一的专家编号 `expert_idx(T)`，需要将 Token 分发到 `dispatched_x(E,capacity,D)`。每个专家内部必须保持原 Token 下标的升序，并将数量写入 `token_counts(E)`。

本题采用 top-1 路由：每个 Token 只有一个 `expert_idx`，不存在 top-k 重复项。官方运行时固定 `capacity=T`，因此任何专家都不会超出容量。每个专家都从 slot 0 开始紧凑写入，有效数量之后的槽位可以保持为零。

### 示例

```text
T=4，D=3，E=2，capacity=4
x=[[1,0,0],[0,1,0],[0,0,1],[1,1,0]]
expert_idx=[0,1,0,1]
token_counts=[2,2]
专家0有效区域=[[1,0,0],[0,0,1]]
专家1有效区域=[[0,1,0],[1,1,0]]
```

约束为 `1≤T≤65536`、`1≤D≤2048`、`2≤E≤64`，且 `expert_idx[i]∈[0,E)`。`x` 和 `dispatched_x` 的数据类型为 FP32，编号和计数为 INT32。性能用例为 `T=16384`、`D=512`、`E=8`。

## 模板代码解析

CUDA starter 的签名为：

```text
extern "C" void solve(const float* x, const int* expert_idx,
                      float* dispatched_x, int* token_counts,
                      int T, int D, int E, int capacity)
```

输出按 `((expert*capacity+slot)*D+feature)` 寻址。如果只用全局 `atomicAdd` 为并发 Token 分配 slot，写入顺序会取决于线程调度，无法保证专家内部的稳定顺序。

## 实现思路

一维 Grid 中的每个 Block 负责一个专家。Block 内的线程沿 Token 维分批扫描，每批都将“是否属于当前专家”写入 Shared Memory，再执行 inclusive scan。批内位置等于扫描值减一，当前批次之前的数量保存在 `base` 中，因此全局 slot 为 `base+local_position`。每批完成后更新 `base`，再继续处理下一批。

所有批次都按 Token 下标递增处理，批内扫描也遵循下标顺序，因此能够保持专家内部的稳定顺序。处理完所有批次后，写入 `token_counts[expert]=base`。特征复制由 Block 内的线程协作完成，目标布局为 `(E,capacity,D)`。复杂度为 `O(E×T+T×D)`；在官方 `E≤64` 的范围内，不需要额外的全局扫描缓冲。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void dispatch_kernel(const float* x, const int* expert_idx,
                                float* dispatched_x, int* token_counts,
                                int T, int D, int E, int capacity) {
    __shared__ int scan[256];
    __shared__ int base;
    __shared__ int batch_count;
    int expert = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = (int)blockDim.x;
    if (expert >= E) return;
    if (tid == 0) base = 0;
    __syncthreads();

    for (int start = 0; start < T; start += block_size) {
        int token = start + tid;
        int selected = token < T && expert_idx[token] == expert ? 1 : 0;
        scan[tid] = selected;
        __syncthreads();
        for (int offset = 1; offset < block_size; offset <<= 1) {
            int value = tid >= offset ? scan[tid - offset] : 0;
            __syncthreads();
            scan[tid] += value;
            __syncthreads();
        }
        if (tid == 0) {
            int valid = T - start;
            if (valid > block_size) valid = block_size;
            batch_count = scan[valid - 1];
        }
        __syncthreads();

        if (selected) {
            int slot = base + scan[tid] - 1;
            for (int feature = 0; feature < D; ++feature) {
                size_t output_index =
                    (static_cast<size_t>(expert) * capacity + slot) * D + feature;
                size_t input_index = (size_t)token * D + feature;
                dispatched_x[output_index] = x[input_index];
            }
        }
        __syncthreads();
        if (tid == 0) base += batch_count;
        __syncthreads();
    }
    if (tid == 0) token_counts[expert] = base;
}

// x, expert_idx, dispatched_x, token_counts are device pointers
extern "C" void solve(const float* x, const int* expert_idx,
                      float* dispatched_x, int* token_counts,
                      int T, int D, int E, int capacity) {
    dispatch_kernel<<<E, 256>>>(x, expert_idx, dispatched_x, token_counts,
                                T, D, E, capacity);
    cudaDeviceSynchronize();
}
```
