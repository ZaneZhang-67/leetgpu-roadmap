---
id: swiglu-mlp-block
leetgpuId: 84
titleEn: SwiGLU MLP Block
titleZh: SwiGLU MLP 模块
description: 使用 CUDA 实现门控投影、上投影、SiLU 门控和下投影组成的完整 SwiGLU MLP
problemUrl: https://leetgpu.com/challenges/swiglu-mlp-block
updatedAt: 2026-11-12
---
## 题目定义

输入 `x` 的形状为 `[M, d_model]`，三个 FP32 权重矩阵采用行主序：

- `W_gate`：`[d_model, d_ffn]`
- `W_up`：`[d_model, d_ffn]`
- `W_down`：`[d_ffn, d_model]`

计算：

```text
gate = x × W_gate
up = x × W_up
hidden = SiLU(gate) ⊙ up
output = hidden × W_down
SiLU(z) = z × sigmoid(z)
```

输出形状为 `[M, d_model]`。

### 实现要求

- 不使用框架之外的外部库
- `solve` 函数签名必须保持不变
- 最终结果写入 `output`

### 示例

```text
M = 2，d_model = 2，d_ffn = 4
x = [[1, 0], [0, 1]]
W_gate = W_up = [[1, 0, 0, 0], [0, 1, 0, 0]]
W_down = [[1, 0], [0, 1], [0, 0], [0, 0]]
output ≈ [[0.7311, 0], [0, 0.7311]]
```

### 约束

- `1 ≤ M ≤ 65,536`
- `1 ≤ d_model ≤ 8,192`
- `1 ≤ d_ffn ≤ 32,768`
- 所有张量均为设备端 FP32，输入值范围为 `[-10,10]`
- 功能测试包含零输入和非 2 的幂维度
- 性能测试使用 `M = 512`、`d_model = 4096`、`d_ffn = 14336`
- 绝对误差和相对误差均为 `1e-4`

## 模板代码解析

CUDA starter 只提供固定入口：

```cpp
#include <cuda_runtime.h>

// x, W_gate, W_up, W_down, output are device pointers
extern "C" void solve(
    const float* x,
    const float* W_gate,
    const float* W_up,
    const float* W_down,
    float* output,
    int M,
    int d_model,
    int d_ffn
) {}
```

所有指针均指向设备内存。starter 没有提供 kernel、隐藏层缓冲区或启动配置。权重无需转置，矩阵元素分别按 `W_gate[p*d_ffn+j]`、`W_up[p*d_ffn+j]` 和 `W_down[j*d_model+n]` 读取。

## 实现思路

实现包含两个采用 `16×16` Shared Memory Tile 的 GEMM kernel。每个 Block 启动 256 个线程，由线程 `(ty,tx)` 负责一个输出位置。

第一个 kernel 负责计算 `hidden` 的 `[M,d_ffn]` Tile。每轮沿 `d_model` 加载三个共享内存 Tile：`xTile`、`gateTile` 和 `upTile`。同一份 `xTile` 同时用于门控投影和上投影，各线程分别在寄存器中累计：

```text
gate = Σ_p x[m,p] × W_gate[p,j]
up = Σ_p x[m,p] × W_up[p,j]
hidden[m,j] = gate × sigmoid(gate) × up
```

第二个 kernel 负责计算 `[M,d_model]` 输出 Tile。每轮沿 `d_ffn` 协作加载 `hiddenTile` 和 `downTile`，随后累计：

```text
output[m,n] = Σ_j hidden[m,j] × W_down[j,n]
```

遇到任意维度不足 16 的边缘 Tile 时，加载阶段会为越界位置写零。所有线程都参与两次 `__syncthreads()`，仅在最终写回时检查输出的行列边界，以免边缘线程提前返回并造成同步死锁。两个 kernel 在默认流中依次启动，确保下投影读取到完整的 `hidden`。临时缓冲区包含 `M × d_ffn` 个 FP32 元素；经过分块后，每个加载到共享内存的值都能被同一 Tile 的 16 个输出复用。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

constexpr int TILE = 16;

__global__ void project_and_gate(
    const float* x,
    const float* W_gate,
    const float* W_up,
    float* hidden,
    int M,
    int d_model,
    int d_ffn
) {
    __shared__ float xTile[TILE][TILE];
    __shared__ float gateTile[TILE][TILE];
    __shared__ float upTile[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int column = blockIdx.x * TILE + tx;
    float gate = 0.0f;
    float up = 0.0f;

    for (int base = 0; base < d_model; base += TILE) {
        int xColumn = base + tx;
        int weightRow = base + ty;
        xTile[ty][tx] =
            (row < M && xColumn < d_model)
            ? x[row * d_model + xColumn]
            : 0.0f;
        gateTile[ty][tx] =
            (weightRow < d_model && column < d_ffn)
            ? W_gate[weightRow * d_ffn + column]
            : 0.0f;
        upTile[ty][tx] =
            (weightRow < d_model && column < d_ffn)
            ? W_up[weightRow * d_ffn + column]
            : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float value = xTile[ty][k];
            gate += value * gateTile[k][tx];
            up += value * upTile[k][tx];
        }
        __syncthreads();
    }

    if (row < M && column < d_ffn) {
        float sigmoid = 1.0f / (1.0f + expf(-gate));
        hidden[row * d_ffn + column] = gate * sigmoid * up;
    }
}

__global__ void down_projection(
    const float* hidden,
    const float* W_down,
    float* output,
    int M,
    int d_model,
    int d_ffn
) {
    __shared__ float hiddenTile[TILE][TILE];
    __shared__ float downTile[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int column = blockIdx.x * TILE + tx;
    float sum = 0.0f;

    for (int base = 0; base < d_ffn; base += TILE) {
        int hiddenColumn = base + tx;
        int weightRow = base + ty;
        hiddenTile[ty][tx] =
            (row < M && hiddenColumn < d_ffn)
            ? hidden[row * d_ffn + hiddenColumn]
            : 0.0f;
        downTile[ty][tx] =
            (weightRow < d_ffn && column < d_model)
            ? W_down[weightRow * d_model + column]
            : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += hiddenTile[ty][k] * downTile[k][tx];
        }
        __syncthreads();
    }

    if (row < M && column < d_model) {
        output[row * d_model + column] = sum;
    }
}

// x, W_gate, W_up, W_down, output are device pointers
extern "C" void solve(
    const float* x,
    const float* W_gate,
    const float* W_up,
    const float* W_down,
    float* output,
    int M,
    int d_model,
    int d_ffn
) {
    float* hidden = nullptr;
    cudaMalloc(
        &hidden,
        static_cast<size_t>(M) * d_ffn * sizeof(float)
    );

    dim3 threads(TILE, TILE);
    dim3 hiddenBlocks(
        (d_ffn + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );
    dim3 outputBlocks(
        (d_model + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );
    project_and_gate<<<hiddenBlocks, threads>>>(
        x, W_gate, W_up, hidden, M, d_model, d_ffn
    );
    down_projection<<<outputBlocks, threads>>>(
        hidden, W_down, output, M, d_model, d_ffn
    );

    cudaDeviceSynchronize();
    cudaFree(hidden);
}
```
