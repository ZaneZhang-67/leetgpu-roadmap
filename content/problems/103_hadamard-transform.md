---
id: hadamard-transform
leetgpuId: 103
titleEn: Hadamard Transform
titleZh: 哈达玛变换
description: 使用 CUDA 蝶形运算对批量 FP32 行执行正交 Walsh–Hadamard 变换
problemUrl: https://leetgpu.com/challenges/hadamard-transform
updatedAt: 2026-11-12
---
## 题目定义

对 `(B,N)` 的每一行独立计算正交 Walsh–Hadamard 变换：

```text
y = H_N x / sqrt(N)
```

`N` 保证是二次幂。每个阶段 `h=1,2,...,N/2` 都把同一蝶形中的 `(a,b)` 替换为 `(a+b,a-b)`；全部阶段完成后，再乘 `1/sqrt(N)`

### 示例

```text
B=1, N=4
input  = [1,2,3,4]
h=1    = [3,-1,7,-1]
h=2    = [10,-2,-4,0]
output = [5,-1,-2,0]
```

**约束与测试**

- `1≤B≤16,384`，`1≤N≤4096` 且 `N` 是二次幂
- 输入输出均为 FP32，计算完全驻留 GPU
- 测试覆盖 `N=1/2`、全零、常量行、宽 batch 与 `N=2048`
- 性能测试 `B=8192,N=4096`，绝对与相对误差容限均为 `1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int B, int N) {}
```

官方 starter 中只有空入口。输入不可原地覆盖，最终结果写入形状相同的 `output`；kernel 和临时空间由作答者实现

## 实现思路

先将 input 复制到 output，再逐阶段原地执行蝶形。每个阶段共有 `B*N/2` 对元素，每个线程负责一对。设线程在当前行内的 pair 编号为 `p`，则：

```text
group = p / h
lane  = p % h
left  = group*(2h)+lane
right = left+h
```

同一阶段写入的元素互不重叠，但下一阶段依赖前一阶段，因此各阶段使用独立的 kernel launch 形成全局同步。全部阶段结束后，再单独乘 `1/sqrt(N)`。总工作量为 `O(BN logN)`，额外空间为零

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void copy_kernel(const float* input, float* output, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (; i < total; i += stride) output[i] = input[i];
}

__global__ void butterfly(float* output, size_t pairs, int N, int h) {
    size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (; p < pairs; p += stride) {
        size_t row = p / (N / 2);
        size_t local = p % (N / 2);
        size_t left = row * N + (local / h) * (2 * h) + local % h;
        size_t right = left + h;
        float a = output[left], b = output[right];
        output[left] = a + b;
        output[right] = a - b;
    }
}

__global__ void scale_kernel(float* output, size_t total, float scale) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (; i < total; i += stride) output[i] *= scale;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int B, int N) {
    size_t total = (size_t)B * N;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;
    copy_kernel<<<blocks, threads>>>(input, output, total);
    size_t pairs = total / 2;
    int pair_blocks = (int)((pairs + threads - 1) / threads);
    if (pair_blocks > 4096) pair_blocks = 4096;
    for (int h = 1; h < N; h *= 2)
        butterfly<<<pair_blocks, threads>>>(output, pairs, N, h);
    scale_kernel<<<blocks, threads>>>(output, total, rsqrtf((float)N));
    cudaDeviceSynchronize();
}
```
