---
id: fused-gate-up-projection-with-silu
leetgpuId: 146
titleEn: Fused Gate-Up Projection with SiLU
titleZh: 融合 Gate-Up 投影与 SiLU
description: 使用共享内存分块同时计算 Gate 与 Up 投影并融合 SiLU 乘法
problemUrl: https://leetgpu.com/challenges/fused-gate-up-projection-with-silu
updatedAt: 2026-11-12
---
## 题目定义

`x(M,H)` 乘 packed 权重 `W_gate_up(H,2I)`。前 I 列是 Gate，后 I 列是 Up：

```text
g[m,i] = Σh x[m,h] W[h,i]
u[m,i] = Σh x[m,h] W[h,I+i]
output[m,i] = g*sigmoid(g)*u
```

权重是横向 `[gate|up]`，不是 `(2I,H)`。乘积必须以 FP32 或更高精度累加。

### 示例

`M=H=I=2`，x 为单位阵，packed 权重两半均为单位阵，输出为 `[[0.731059,0],[0,0.731059]]`。

约束为 `M,H≤8192`、`I≤16384`，Gate 可达约 ±20。性能用例 `4096×4096×11008`。

## 模板代码解析

签名为：

```text
extern "C" void solve(const float* x, const float* W_gate_up,
                      float* output, int M, int H, int I)
```

输出仅有 I 列；无需把 `M×2I` 投影写回全局内存。

## 实现思路

一个 `16×16` Block 计算输出 tile。共享内存中的 x tile 被 Gate、Up 两次点积共同复用，同时加载权重两半。每个线程维护两个 FP32 累加器，K 维扫描结束后直接执行 SiLU 和逐元素乘法，融合 epilogue 并省去巨大的中间矩阵。

所有边缘维度都带判断，支持奇数 I 和非 16 倍数。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <cmath>

__global__ void gate_up(const float* x,const float* W,float* out,int M,int H,int I){
 __shared__ float X[16][16],G[16][16],U[16][16];
 int tx=threadIdx.x,ty=threadIdx.y,m=blockIdx.y*16+ty,i=blockIdx.x*16+tx;
 float g=0,u=0;
 for(int k0=0;k0<H;k0+=16){int kx=k0+tx,kw=k0+ty;
  X[ty][tx]=(m<M&&kx<H)?x[(long long)m*H+kx]:0;
  G[ty][tx]=(kw<H&&i<I)?W[(long long)kw*(2*I)+i]:0;
  U[ty][tx]=(kw<H&&i<I)?W[(long long)kw*(2*I)+I+i]:0;
  __syncthreads();
  #pragma unroll
  for(int k=0;k<16;++k){g+=X[ty][k]*G[k][tx];u+=X[ty][k]*U[k][tx];}
  __syncthreads();
 }
 if(m<M&&i<I)out[(long long)m*I+i]=(g/(1.0f+expf(-g)))*u;
}
extern "C" void solve(const float*x,const float*W_gate_up,float*output,int M,int H,int I){
 dim3 t(16,16),b((I+15)/16,(M+15)/16);gate_up<<<b,t>>>(x,W_gate_up,output,M,H,I);cudaDeviceSynchronize();
}
```
