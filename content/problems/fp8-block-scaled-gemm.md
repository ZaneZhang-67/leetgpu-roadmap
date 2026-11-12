---
id: fp8-block-scaled-gemm
leetgpuId: 140
titleEn: FP8 Block-Scaled GEMM
titleZh: FP8 分块缩放矩阵乘法
description: 按 E4M3FN 位级格式解码矩阵元素，应用 reduction 维的 1×128 分块缩放并输出 FP16
problemUrl: https://leetgpu.com/challenges/fp8-block-scaled-gemm
updatedAt: 2026-11-12
---
## 题目定义

`x_q[M,K]` 与 `w_q[N,K]` 的每个 uint8 都是 PyTorch `float8_e4m3fn` 的位模式，而不是无符号整数值。每 128 个连续 K 元素共享一个 FP32 scale：

```text
x[m,k]=decode_e4m3fn(x_q[m,k])*x_scales[m,k/128]
w[n,k]=decode_e4m3fn(w_q[n,k])*w_scales[n,k/128]
y[m,n]=Σk x[m,k]*w[n,k]
```

输出为 FP16 `[M,N]`。题目没有额外 alpha 参数；两个 block scale 的乘积直接乘入每项。

### 官方示例

```text
0x38=1.0, 0x30=0.5, 0x40=2.0, 0xB8=-1.0
M=N=2,K=128, scales=1
y=[[256,-128],[128,-64]] (float16)
```

### 完整约束

- `1 ≤ M,N ≤ 16384`
- `128 ≤ K ≤ 16384` 且 K 能被 128 整除
- FP8 payload 有限，无 NaN；题面给出的 scale 范围是 `[0.01,4]`，runtime 的 tiny-scale 用例还会生成约 `[0.0025,0.015]` 的正 FP32 scale
- x/w 为 uint8 位模式，y 为 FP16
- 性能测试为 `M=2048,N=4096,K=4096`
- 容差为 `atol=0.1,rtol=0.05`

## 模板代码解析

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
extern "C" void solve(const uint8_t* x_q,const float* x_scales,
 const uint8_t* w_q,const float* w_scales,__half* y,int M,int N,int K) {}
```

scale 布局分别为 `[M,K/128]` 和 `[N,K/128]`。参考代码手工按 E4M3FN 解码，覆盖零、subnormal 和 finite normal；测试排除了 NaN 位型。

## 实现思路

一个线程计算一个输出元素，从 FP8 解码和 scale 相乘开始就使用 double，并按 K 累加 double。E4M3FN 位字段为 sign 1 位、exponent 4 位、mantissa 3 位，bias 为 7：exp=0 时值为 `mantissa*2^-9`；其余有限值为 `(1+mantissa/8)*2^(exp-7)`，再应用符号。scale 每 128 项加载一次。最终只在写回时用 `__double2half` 转成 FP16。这样与 runtime 的 float64 reference 一致，也覆盖基础采样再乘 `0.01`、实际下探至约 `0.0025` 的 tiny-scale 测试。二维 Grid 覆盖全部 M×N。

## 参考实现

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
__device__ float decode(uint8_t u){int sign=u>>7,exp=(u>>3)&15,man=u&7;float v;if(exp==0)v=ldexpf((float)man,-9);else v=ldexpf(1.0f+(float)man*0.125f,exp-7);return sign?-v:v;}
__global__ void gemm(const uint8_t*x,const float*xs,const uint8_t*w,const float*ws,__half*y,int M,int N,int K){int n=blockIdx.x*blockDim.x+threadIdx.x,m=blockIdx.y*blockDim.y+threadIdx.y;if(m<M&&n<N){int NB=K/128;double sum=0;for(int k=0;k<K;++k){int b=k/128;double xv=(double)decode(x[m*K+k])*(double)xs[m*NB+b];double wv=(double)decode(w[n*K+k])*(double)ws[n*NB+b];sum+=xv*wv;}y[m*N+n]=__double2half(sum);}}
extern "C" void solve(const uint8_t*x,const float*xs,const uint8_t*w,const float*ws,__half*y,int M,int N,int K){dim3 block(16,16),grid((N+15)/16,(M+15)/16);gemm<<<grid,block>>>(x,xs,w,ws,y,M,N,K);cudaDeviceSynchronize();}
```
