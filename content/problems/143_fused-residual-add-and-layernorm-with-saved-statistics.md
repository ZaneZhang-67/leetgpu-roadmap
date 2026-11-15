---
id: fused-residual-add-and-layernorm-with-saved-statistics
leetgpuId: 143
titleEn: Fused Residual Add and LayerNorm with Saved Statistics
titleZh: 融合残差加法与保存统计量的 LayerNorm
description: 使用 CUDA 融合残差加法、稳定 LayerNorm，并保存均值与倒标准差
problemUrl: https://leetgpu.com/challenges/fused-residual-add-and-layernorm-with-saved-statistics
updatedAt: 2026-11-12
---
## 题目定义

先计算 `h=x+residual`，再逐行做 LayerNorm：

```text
mean=mean(h), rstd=rsqrt(mean((h-mean)²)+eps)
y=(h-mean)*rstd*gamma+beta
```

必须同时完整写出 `y(N,C)`、`residual_out=h`、`mean(N)` 和 `rstd(N)`。统计基于已经舍入为 FP32 的 h。

### 示例

```text
h=[[2,2,2,4],[1,1,1,1]]
mean=[2.5,1], rstd≈[1.1547,316.2278]
y≈[[-0.5773,-1.1547,0.4227,0.8660],[0,0,1,0]]
```

**约束与测试**

- `1≤N≤65,536`，`1≤C≤16,384`，全部张量为 FP32
- x/residual `[-1000,1000]`，gamma `[-2,2]`，beta `[-0.5,0.5]`
- `1e-6≤eps≤1`；测试含常量行、大 DC offset 和非二次幂宽度
- 性能测试 `N=16,384,C=4096`，容差 `atol=rtol=1e-3`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// x, residual, gamma, beta, y, residual_out, mean, rstd are device pointers
extern "C" void solve(const float* x, const float* residual, const float* gamma, const float* beta,
                      float* y, float* residual_out, float* mean, float* rstd,
                      int N, int C, float eps) {}
```

官方 starter 只提供空入口。四个输出都预填垃圾值，不能遗漏任何元素。

## 实现思路

每个 Block 负责一行。第一遍计算 FP32 的 `h=x+residual` 并写 residual_out，同时用 Welford 算法形成线程局部 `(count,mean,M2)`；共享内存合并后得到稳定总体方差。该方法避免大 offset 测试中 `E[h²]-E[h]²` 的灾难性抵消。

线程 0 写 mean 与 rstd，同步后各线程读取 residual_out，完成 gamma/beta 仿射并写 y。每个输入只在第一遍读取一次。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__device__ void merge(int& na,float& ma,float& va,int nb,float mb,float vb){if(!nb)return;if(!na){na=nb;ma=mb;va=vb;return;}float d=mb-ma;int n=na+nb;ma+=d*((float)nb/n);va+=vb+d*d*((float)na*nb/n);na=n;}
__global__ void fused_kernel(const float*x,const float*r,const float*g,const float*b,float*y,float*h,float*means,float*rstds,int N,int C,float eps){
 int row=(int)blockIdx.x,tid=(int)threadIdx.x,bs=(int)blockDim.x;if(row>=N)return;extern __shared__ unsigned char sm[];float*ms=(float*)sm;float*vs=ms+bs;int*ns=(int*)(vs+bs);size_t base=(size_t)row*C;int n=0;float m=0,v=0;
 for(int j=tid;j<C;j+=bs){float z=x[base+j]+r[base+j];h[base+j]=z;++n;float d=z-m;m+=d/n;v+=d*(z-m);}ns[tid]=n;ms[tid]=m;vs[tid]=v;__syncthreads();
 for(int o=bs/2;o;o>>=1){if(tid<o){int nn=ns[tid];float mm=ms[tid],vv=vs[tid];merge(nn,mm,vv,ns[tid+o],ms[tid+o],vs[tid+o]);ns[tid]=nn;ms[tid]=mm;vs[tid]=vv;}__syncthreads();}
 if(tid==0){means[row]=ms[0];rstds[row]=rsqrtf(vs[0]/C+eps);}__syncthreads();
 for(int j=tid;j<C;j+=bs)y[base+j]=(h[base+j]-means[row])*rstds[row]*g[j]+b[j];}

// x, residual, gamma, beta, y, residual_out, mean, rstd are device pointers
extern "C" void solve(const float* x, const float* residual, const float* gamma, const float* beta,
                      float* y, float* residual_out, float* mean, float* rstd,
                      int N, int C, float eps) {
 int threads=256;size_t sm=threads*(2*sizeof(float)+sizeof(int));fused_kernel<<<N,threads,sm>>>(x,residual,gamma,beta,y,residual_out,mean,rstd,N,C,eps);cudaDeviceSynchronize();
}
```
