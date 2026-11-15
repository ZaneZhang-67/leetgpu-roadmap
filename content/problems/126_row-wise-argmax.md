---
id: row-wise-argmax
leetgpuId: 126
titleEn: Row-wise Argmax
titleZh: 逐行最大值索引
description: 使用 CUDA 对 FP32 矩阵逐行归约，并在并列最大值时返回最小列索引
problemUrl: https://leetgpu.com/challenges/row-wise-argmax
updatedAt: 2026-11-12
---
## 题目定义

对形状 `(B,V)` 的 FP32 logits 每行求最大值所在列，并把 INT32 索引写入 `output(B)`。若多个元素并列最大，返回最小列索引。

### 示例

```text
logits=[[1,3,2,-1],[-2,-0.5,-0.5,-3]]
output=[1,1]
```

第二行第 1、2 列并列，选择较小的索引 1。

**约束与测试**

- `1≤B≤1024`，`1≤V≤262,144`，不存在空行
- logits 全部是有限 FP32，官方契约和测试都不包含 NaN
- 测试覆盖 `V=1`、全零、全负、末列最大、重复最大值和 `V=32,000`
- 性能测试 `B=256,V=128,000`，输出要求完全一致，`atol=rtol=0`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// logits, output are device pointers
extern "C" void solve(const float* logits, int* output, int B, int V) {}
```

官方 starter 只提供空入口。logits 按行主序存储，output 是每行一个 INT32 索引。

## 实现思路

每个 Block 处理一行。每个线程从自己的第一列开始跨步扫描，维护局部 `(value,index)`。比较规则为：值更大者胜出；值相等时索引更小者胜出。共享内存归约同样使用该规则，因此与线程划分和归约树无关，始终得到首个最大值。

`V≥1` 且输入有限，可以用负无穷初始化局部最大值。官方不包含 NaN，无需定义 NaN 比较行为。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__device__ bool better(float va, int ia, float vb, int ib) {
    return va > vb || (va == vb && ia < ib);
}

__global__ void argmax_kernel(const float* logits, int* output, int B, int V) {
    int row = static_cast<int>(blockIdx.x);
    int tid = static_cast<int>(threadIdx.x);
    int block_size = static_cast<int>(blockDim.x);
    if (row >= B) return;
    extern __shared__ float values[];
    int* indices = reinterpret_cast<int*>(values + block_size);
    float best = -INFINITY;
    int best_index = V;
    size_t base = (size_t)row * V;
    for (int col = tid; col < V; col += block_size) {
        float value = logits[base + col];
        if (better(value, col, best, best_index)) {
            best = value; best_index = col;
        }
    }
    values[tid] = best; indices[tid] = best_index;
    __syncthreads();
    for (int off = block_size / 2; off > 0; off >>= 1) {
        if (tid < off && better(values[tid + off], indices[tid + off],
                                values[tid], indices[tid])) {
            values[tid] = values[tid + off];
            indices[tid] = indices[tid + off];
        }
        __syncthreads();
    }
    if (tid == 0) output[row] = indices[0];
}

// logits, output are device pointers
extern "C" void solve(const float* logits, int* output, int B, int V) {
    int threads = 256;
    size_t shared = threads * (sizeof(float) + sizeof(int));
    argmax_kernel<<<B, threads, shared>>>(logits, output, B, V);
    cudaDeviceSynchronize();
}
```
