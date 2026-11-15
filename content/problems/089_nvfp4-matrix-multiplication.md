---
id: nvfp4-matrix-multiplication
leetgpuId: 89
titleEn: NVFP4 Matrix Multiplication
titleZh: NVFP4 矩阵乘法
description: 解码打包的 E2M1 FP4 操作数和原始 E4M3 块缩放字节，计算带 FP32 alpha 的 FP16 矩阵乘法
problemUrl: https://leetgpu.com/challenges/nvfp4-matrix-multiplication
updatedAt: 2026-11-12
---
## 题目定义

计算：

```text
y = alpha × (x × wᵀ)
```

逻辑矩阵 `x` 的形状为 `[M,K]`，`w` 的形状为 `[N,K]`。输出 `y` 是按行主序存储的 `[M,N]` FP16 矩阵。输入采用 NVFP4 数据格式：

- `x_q` 的真实存储形状是 `[M,K/2]`，类型为 `uint8`
- `w_q` 的真实存储形状是 `[N,K/2]`，类型为 `uint8`
- 每个字节的高四位保存偶数 `k`，低四位保存奇数 `k`
- nibble 使用 OCP E2M1：`0, 0.5, 1, 1.5, 2, 3, 4, 6` 及对应负值
- 每连续 16 个 FP4 值共享一个 E4M3 块缩放值
- `x_scales` 和 `w_scales` 分别为 `[M,K/16]`、`[N,K/16]` 的原始 `uint8` 位模式，按 `float8_e4m3fn` 解码
- `alpha` 是独立的 FP32 标量，不是额外的每张量量化存储

解量化和乘法为：

```text
x[m,k] = E2M1(x_q[m,k]) × E4M3(x_scales[m,k/16])
w[n,k] = E2M1(w_q[n,k]) × E4M3(w_scales[n,k/16])
y[m,n] = FP16(alpha × Σ_k x[m,k] × w[n,k])
```

### 实现要求

- 只使用原生功能，不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果以 FP16 写入 `y`

### 示例

```text
M = 2，N = 2，K = 16，alpha = 1
x_q = [[0x22 × 8], [0x11 × 8]]
w_q = [[0x44 × 8], [0xAA × 8]]
x_scales = w_scales = [[0x38], [0x38]]
y = [[32, -16], [16, -8]]
```

其中 `0x2/0x1/0x4/0xA` 分别表示 `1/0.5/2/-1`，E4M3 位模式 `0x38` 表示 `1.0`。

### 约束

- `1 ≤ M,N ≤ 32,768`
- `16 ≤ K ≤ 32,768`，且 `K` 是 16 的倍数
- 所有输入张量按行主序存储
- 四个量化输入的实际 dtype 都是 `uint8`，输出为 FP16
- 功能测试包含非 2 的幂 `M,N`、零激活及 `alpha = 0.125`、`1/64`
- 性能测试使用 `M = 2048`、`N = 18432`、`K = 3072`、`alpha = 1/64`
- 绝对误差为 `0.1`，相对误差为 `0.05`

题面提到 Hopper 和 Blackwell，但官方 CUDA starter 只依赖 `cuda_fp16.h`、`cuda_runtime.h` 和 `stdint.h`，没有要求使用专用 FP4 类型或特定 `sm_` 编译目标。下面的实现按照官方真实的 `uint8` 位布局进行软件解码，不依赖 Blackwell FP4 指令。若改用硬件 FP4 Tensor Core，则需要匹配相应的架构、工具链和硬件布局。

## 模板代码解析

官方 CUDA starter 为：

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// x_q, x_scales, w_q, w_scales, y are device pointers
extern "C" void solve(
    const uint8_t* x_q,
    const uint8_t* x_scales,
    const uint8_t* w_q,
    const uint8_t* w_scales,
    float alpha,
    __half* y,
    int M,
    int N,
    int K
) {}
```

`x_q` 和 `w_q` 并非每个元素占一个字节，而是每个字节打包两个 FP4。scale 指针虽然是 `uint8_t*`，其中的字节却采用 E4M3 浮点编码，不能按 0 到 255 的整数缩放值解释。

## 实现思路

输出按 `16×16` Tile 分块，每个 CUDA Block 启动 256 个线程，由线程 `(ty,tx)` 负责一个 `y[row,column]`。归约维同样按 16 分块。每轮由全部线程协作解码一个 `16×16` 的 `x` Tile 和一个 `16×16` 的 `w` Tile，并写入共享内存；随后，每个线程复用这两块数据，累加 16 项。

第 `k` 个 FP4 位于 `packed[k/2]`：`k` 为偶数时取高 nibble，为奇数时取低 nibble。E2M1 通过 16 项常量表精确解码。由于 `K` 是 16 的倍数，每个归约 Tile 恰好对应一个 E4M3 scale block，同一输入行或权重行只需读取对应的一个 scale。

E4M3FN 的字节布局包含 1 位符号、4 位指数和 3 位尾数，bias 为 7。指数非零时，值为 `(-1)^s × 2^(e-7) × (1+m/8)`；指数为零时，则按次正规数 `(-1)^s × 2^-6 × (m/8)` 解码。特殊编码 `exp=15,m=7` 表示 NaN。官方公开 scale 是将从 `0.5` 到 `2.0` 的正数转换为 `torch.float8_e4m3fn`，不会生成 NaN；示例中的 `0x38` 解码为 1。

累加使用 FP32，循环结束后再乘 `alpha`，并转换为 `__half`。边缘 Block 会在加载和写回时分别检查 `row < M`、`column < N`，越界的共享内存槽填零。与为每个输出重复解码相比，这种写法让每个解量化值在 Tile 内供 16 个输出复用，适合官方的大矩阵性能形状。使用硬件 FP4 Tensor Core 还可进一步提升吞吐。

## 参考实现

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ float decode_e2m1(uint8_t bits) {
    const float table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f,
        2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f,
        -2.0f, -3.0f, -4.0f, -6.0f
    };
    return table[bits & 0x0f];
}

__device__ __forceinline__ float decode_e4m3fn(uint8_t bits) {
    int sign = bits >> 7;
    int exponent = (bits >> 3) & 0x0f;
    int mantissa = bits & 0x07;
    if (exponent == 15 && mantissa == 7) return nanf("");
    float value;
    if (exponent == 0) {
        value = ldexpf(static_cast<float>(mantissa) / 8.0f, -6);
    } else {
        value = ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f,
                       exponent - 7);
    }
    return sign ? -value : value;
}

__device__ __forceinline__ float load_fp4(
    const uint8_t* row, int k
) {
    uint8_t byte = row[k >> 1];
    uint8_t nibble = (k & 1) ? (byte & 0x0f) : (byte >> 4);
    return decode_e2m1(nibble);
}

__global__ void nvfp4_matmul_kernel(
    const uint8_t* x_q,
    const uint8_t* x_scales,
    const uint8_t* w_q,
    const uint8_t* w_scales,
    float alpha,
    __half* y,
    int M,
    int N,
    int K
) {
    __shared__ float xTile[16][16];
    __shared__ float wTile[16][16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty;
    int column = blockIdx.x * 16 + tx;
    int packedStride = K / 2;
    int scaleStride = K / 16;
    float sum = 0.0f;

    for (int kBase = 0; kBase < K; kBase += 16) {
        int k = kBase + tx;
        if (row < M) {
            const uint8_t* xRow = x_q + row * packedStride;
            float scale = decode_e4m3fn(
                x_scales[row * scaleStride + kBase / 16]
            );
            xTile[ty][tx] = load_fp4(xRow, k) * scale;
        } else {
            xTile[ty][tx] = 0.0f;
        }

        int weightColumn = blockIdx.x * 16 + ty;
        int weightK = kBase + tx;
        if (weightColumn < N) {
            const uint8_t* wRow = w_q + weightColumn * packedStride;
            float scale = decode_e4m3fn(
                w_scales[weightColumn * scaleStride + kBase / 16]
            );
            wTile[ty][tx] = load_fp4(wRow, weightK) * scale;
        } else {
            wTile[ty][tx] = 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int j = 0; j < 16; ++j) {
            sum += xTile[ty][j] * wTile[tx][j];
        }
        __syncthreads();
    }

    if (row < M && column < N) {
        y[row * N + column] = __float2half_rn(alpha * sum);
    }
}

// x_q, x_scales, w_q, w_scales, y are device pointers
extern "C" void solve(
    const uint8_t* x_q,
    const uint8_t* x_scales,
    const uint8_t* w_q,
    const uint8_t* w_scales,
    float alpha,
    __half* y,
    int M,
    int N,
    int K
) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    nvfp4_matmul_kernel<<<blocks, threads>>>(
        x_q, x_scales, w_q, w_scales,
        alpha, y, M, N, K
    );
    cudaDeviceSynchronize();
}
```
