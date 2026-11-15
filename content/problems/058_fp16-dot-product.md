---
id: fp16-dot-product
leetgpuId: 58
titleEn: FP16 Dot Product
titleZh: FP16 向量点积
description: 使用 FP32 完成 FP16 向量乘积的确定性归约，并将唯一结果转换回 half
problemUrl: https://leetgpu.com/challenges/fp16-dot-product
updatedAt: 2026-11-12
---
## 题目定义

给定两个长度相同的 FP16 向量 `A` 和 `B`，计算：

```text
result = Σ A[i] × B[i], 0 ≤ i < N
```

输入元素和最终结果都是 `half`，中间乘法与累加使用 FP32

### 实现要求与约束

- 不允许使用外部库
- `solve` 函数签名保持不变
- `A` 和 `B` 长度相同
- `1 ≤ N ≤ 100,000,000`
- `result` 是长度为 1 的设备数组
- 性能测试使用 `N=100,000,000`
- 官方容差为 `atol=0.05`、`rtol=0.05`

### 示例

```text
A = [1,2,3,4]
B = [5,6,7,8]
result = 70
```

## 模板代码解析

CUDA starter 为：

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// A, B, result are device pointers
extern "C" void solve(const half* A, const half* B,
                      half* result, int N) {}
```

平台传入两个 FP16 设备数组和一个单元素输出作答需要完成并行乘积、跨线程归约以及最终 FP16 写回直接对 `half* result` 使用浮点原子加法既不符合 FP32 累加要求，也会让最终舍入发生多次

## 实现思路

参考实现采用两阶段归约每个线程以 Grid-Stride Loop 扫描输入，把 `half` 转成 `float` 后累加局部和；每个 Block 再在 Shared Memory 中按固定树形顺序归约，写出一个 FP32 部分和

第二个单 Block kernel 依次累加所有部分和，并只在最后把 FP32 总和转换为 `half`固定 Block 数量和固定归约顺序使同一输入的计算顺序稳定，也避免多个 Block 对同一结果进行非确定顺序的原子累加

首阶段最多使用 256 个 Block，因此第二阶段由一个线程串行合并的成本很小临时缓冲大小为 `blocks × sizeof(float)`，而不是随 `N` 线性增长

## 参考实现

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void partial_dot(const half* A, const half* B,
                            float* partial, int N) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;

    for (int i = index; i < N; i += stride) {
        sum += __half2float(A[i]) * __half2float(B[i]);
    }
    shared[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) shared[tid] += shared[tid + offset];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = shared[0];
}

__global__ void finish_dot(const float* partial, half* result, int count) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < count; ++i) sum += partial[i];
        result[0] = __float2half(sum);
    }
}

// A, B, result are device pointers
extern "C" void solve(const half* A, const half* B,
                      half* result, int N) {
    const int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 256) blocks = 256;

    float* partial = nullptr;
    cudaMalloc(&partial, blocks * sizeof(float));
    partial_dot<<<blocks, threads, threads * sizeof(float)>>>(A, B, partial, N);
    finish_dot<<<1, 1>>>(partial, result, blocks);
    cudaDeviceSynchronize();
    cudaFree(partial);
}
```
