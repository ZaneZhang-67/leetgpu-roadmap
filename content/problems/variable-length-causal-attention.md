---
id: variable-length-causal-attention
leetgpuId: 102
titleEn: Variable-Length Causal Attention
titleZh: 变长因果注意力
description: 根据累计序列长度在紧凑批次内执行序列隔离的因果缩放点积注意力
problemUrl: https://leetgpu.com/challenges/variable-length-causal-attention
updatedAt: 2026-11-12
---
## 题目定义

`Q`、`K`、`V` 以 `[T,d]` 布局紧凑存放所有序列，边界由 `cu_seqlens[S+1]` 给出。序列 `s` 占用半开区间 `[cu_seqlens[s],cu_seqlens[s+1])`，Query `t` 只能关注同一序列中不晚于自身的 Key：

```text
score(t,u) = dot(Q[t],K[u])/sqrt(d)
start <= u <= t
output[t] = softmax(score) @ V[start:t+1]
```

不同序列必须彼此隔离；即使另一个序列在 packed 数组中的全局下标更小，当前 Query 也不可见。

### 官方示例

```text
T=3,d=4,S=2,cu_seqlens=[0,2,3]
序列长度分别为2和1
output=[[1,2,3,4],
        [3.4898,4.4898,5.4898,6.4898],
        [9,10,11,12]]
```

### 完整约束

- `1 ≤ S ≤ 256`，每条序列长度为 `1..2048`
- `1 ≤ T ≤ 16384`，`1 ≤ d ≤ 256`
- `cu_seqlens[0]=0`、`cu_seqlens[S]=T`，且单调非递减
- Q/K/V/output 为 FP32，`cu_seqlens` 为 int32
- 性能测试为 `T=8192,d=64,S=32`，序列长度介于 128 和 384
- 判定容差为 `1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// Q, K, V, cu_seqlens, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V,
                      const int* cu_seqlens, float* output,
                      int T, int d, int S) {}
```

平台不会直接传入每个 token 的序列编号或局部位置，需要从 `cu_seqlens` 推导；输出仍按 packed token 顺序写入 `[T,d]`。

## 实现思路

先启动一个 kernel，为每个 packed token 二分查找所属序列的起点，并写入 `starts[T]`。随后由一个 Block 处理一个 Query token：线程 0 只遍历 `[start,t]`，计算稳定 Softmax 权重，其他线程则按输出维度并行累加 V。

最长序列为 2048，因此动态 Shared Memory 最大为 8 KiB。实现不会构造 `T×T` 分数矩阵，性能测试也只处理各序列自己的下三角区域。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void locate_starts(const int* cu, int* starts, int T, int S) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    int lo = 0, hi = S;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (cu[mid] <= t) lo = mid;
        else hi = mid;
    }
    starts[t] = cu[lo];
}

__global__ void varlen_attention(const float* Q, const float* K,
                                 const float* V, const int* starts,
                                 float* output, int d) {
    extern __shared__ float weights[];
    int t = blockIdx.x;
    int tid = threadIdx.x;
    int start = starts[t];
    int length = t - start + 1;
    float scale = rsqrtf(static_cast<float>(d));
    if (tid == 0) {
        float maximum = -CUDART_INF_F;
        for (int i = 0; i < length; ++i) {
            int u = start + i;
            float score = 0.0f;
            for (int j = 0; j < d; ++j)
                score += Q[t * d + j] * K[u * d + j];
            weights[i] = score * scale;
            maximum = fmaxf(maximum, weights[i]);
        }
        float denominator = 0.0f;
        for (int i = 0; i < length; ++i) {
            weights[i] = expf(weights[i] - maximum);
            denominator += weights[i];
        }
        for (int i = 0; i < length; ++i) weights[i] /= denominator;
    }
    __syncthreads();
    for (int j = tid; j < d; j += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < length; ++i)
            sum += weights[i] * V[(start + i) * d + j];
        output[t * d + j] = sum;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V,
                      const int* cu_seqlens, float* output,
                      int T, int d, int S) {
    int* starts = nullptr;
    cudaMalloc(&starts, static_cast<size_t>(T) * sizeof(int));
    locate_starts<<<(T + 255) / 256, 256>>>(cu_seqlens, starts, T, S);
    varlen_attention<<<T, 256, 2048 * sizeof(float)>>>(
        Q, K, V, starts, output, d);
    cudaFree(starts);
    cudaDeviceSynchronize();
}
```
