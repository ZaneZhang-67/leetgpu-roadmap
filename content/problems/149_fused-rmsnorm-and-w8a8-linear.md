---
id: fused-rmsnorm-and-w8a8-linear
leetgpuId: 149
titleEn: Fused RMSNorm and W8A8 Linear
titleZh: 融合 RMSNorm 与 W8A8 线性层
description: 对每个 token 执行 RMSNorm 和对称 INT8 动态量化，再与逐输出通道量化权重相乘
problemUrl: https://leetgpu.com/challenges/fused-rmsnorm-and-w8a8-linear
updatedAt: 2026-11-12
---
## 题目定义

先对 `x[M,K]` 每行做带 gamma 的 RMSNorm，epsilon 为 `1e-6`。再按行求独立 activation scale，将归一化结果对称量化到有符号 `[-127,127]`：

```text
y[i,k]=x[i,k]*rsqrt(mean(x[i,:]^2)+1e-6)*gamma[k]
x_scale[i]=max(abs(y[i,:]))/127
q[i,k]=clamp(round_to_nearest_even(y/x_scale),-127,127)
output[i,j]=x_scale[i]*w_scale[j]*Σk q[i,k]*W[j,k]
```

零行的 scale 为 0，所有 q 为 0。W 是 `[N,K]` 的 int8，每个输出通道有一个 `w_scale[N]`；累加器必须至少为 int32，输出为 FP32。

### 官方示例

```text
M=2,N=3,K=4
gamma=[1,1,1,1], w_scale=[0.5,0.25,0.01]
q0=[127,-42,0,42], q1=[127,127,127,127]
output=[[0.9045,-0.2991,0.1809],[0.5,0,0.4]]
```

### 完整约束

- `1 ≤ M,K ≤ 8192`，`1 ≤ N ≤ 16384`
- x/gamma/w_scale/output 为 FP32；W 为 int8，值域 `[-127,127]`
- gamma 在 `[0.5,1.5]`；w_scale 约为 `1/(127*sqrt(K))`
- x 可含约 60 倍 outlier，也可能有全零行；测试还覆盖较大 token/通道规模和非二次幂尾部
- 性能测试为 `M=N=K=4096`
- 容差为 `1e-2`

## 模板代码解析

```cpp
#include <cuda_runtime.h>
extern "C" void solve(const float* x,const float* gamma,const int8_t* W,
 const float* w_scale,float* output,int M,int N,int K) {}
```

权重方向是输出通道行主序 `W[j*K+k]`，不是 `[K,N]`。activation scale 按 token 行布局，weight scale 按输出通道布局。

## 实现思路

第一阶段一个 Block 处理一行，用 256 线程归约平方和与归一化后绝对最大值，然后用 `__float2int_rn` 做 ties-to-even 舍入并写 int8 q 与行 scale。第二阶段二维 Grid 一线程计算一个 `(i,j)`，以 int32 精确累加 `q*W`，最后乘两种 scale。

最大单项为 `127²`，K≤8192 时和小于 2³¹，int32 不溢出。性能规模两个阶段均并行，没有单线程处理整张矩阵。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
__global__ void quantize(const float*x,const float*g,int8_t*q,float*scale,int K){extern __shared__ float s[];int r=blockIdx.x,t=threadIdx.x;float sum=0;for(int k=t;k<K;k+=blockDim.x){float v=x[r*K+k];sum+=v*v;}s[t]=sum;__syncthreads();for(int o=blockDim.x/2;o;o>>=1){if(t<o)s[t]+=s[t+o];__syncthreads();}float inv=rsqrtf(s[0]/K+1e-6f);__syncthreads();float mx=0;for(int k=t;k<K;k+=blockDim.x)mx=fmaxf(mx,fabsf(x[r*K+k]*inv*g[k]));s[t]=mx;__syncthreads();for(int o=blockDim.x/2;o;o>>=1){if(t<o)s[t]=fmaxf(s[t],s[t+o]);__syncthreads();}float sc=s[0]/127.f;if(t==0)scale[r]=sc;for(int k=t;k<K;k+=blockDim.x){int z=sc>0?__float2int_rn(x[r*K+k]*inv*g[k]/sc):0;z=z>127?127:(z<-127?-127:z);q[r*K+k]=(int8_t)z;}}
__global__ void linear(const int8_t*q,const int8_t*W,const float*xs,const float*ws,float*out,int M,int N,int K){int j=blockIdx.x*blockDim.x+threadIdx.x,i=blockIdx.y*blockDim.y+threadIdx.y;if(i<M&&j<N){int acc=0;for(int k=0;k<K;++k)acc+=(int)q[i*K+k]*(int)W[j*K+k];out[i*N+j]=(float)acc*xs[i]*ws[j];}}
extern "C" void solve(const float*x,const float*g,const int8_t*W,const float*ws,float*out,int M,int N,int K){int8_t*q;float*xs;cudaMalloc(&q,(size_t)M*K);cudaMalloc(&xs,(size_t)M*4);quantize<<<M,256,256*sizeof(float)>>>(x,g,q,xs,K);dim3 b(16,16),grid((N+15)/16,(M+15)/16);linear<<<grid,b>>>(q,W,xs,ws,out,M,N,K);cudaFree(q);cudaFree(xs);cudaDeviceSynchronize();}
```
