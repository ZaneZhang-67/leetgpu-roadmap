---
id: layer-norm-backward
leetgpuId: 144
titleEn: Layer Norm Backward
titleZh: Layer Norm 反向传播
description: 使用 CUDA 和保存的逐行统计量计算输入、缩放与偏置梯度
problemUrl: https://leetgpu.com/challenges/layer-norm-backward
updatedAt: 2026-11-12
---
## 题目定义

前向保存了 mean(N) 与 rstd(N)。反向定义：

```text
xhat=(x-mean)*rstd, g=dy*gamma
dx=rstd*(g-mean(g)-xhat*mean(g*xhat))
dgamma[j]=Σ dy[i,j]*xhat[i,j]
dbeta[j]=Σ dy[i,j]
```

三个输出均未初始化，必须完整覆盖。

### 示例

```text
x=[[1,2,3,4],[-1,0,0,1]], gamma=[1,1,1,1]
mean=[2.5,0], rstd=[0.8944,1.4142]
dy=[[1,1,1,1],[1,0,0,1]]
dx=[[0,0,0,0],[0.7071,-0.7071,-0.7071,0.7071]]
dgamma=[-2.7559,-0.4472,0.4472,2.7559], dbeta=[2,1,1,2]
```

**约束与测试**

- `1≤N≤65,536`，`1≤C≤16,384`，所有张量为 FP32
- x `[-250,250]`，dy `[-10,10]`，gamma `[-2,2]`
- 测试覆盖 C=1、常量行、零 dy、负 gamma、非二次幂和奇数 N
- 性能测试 `N=16,384,C=4096`，容差 `atol=rtol=1e-3`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// x, gamma, mean, rstd, dy, dx, dgamma, dbeta are device pointers
extern "C" void solve(const float* x, const float* gamma, const float* mean, const float* rstd,
                      const float* dy, float* dx, float* dgamma, float* dbeta, int N, int C) {}
```

mean 和 rstd 直接来自前向，不重新统计 x。dx 需要逐行归约，dgamma 与 dbeta 需要逐列归约。

## 实现思路

dx kernel 每个 Block 处理一行，同时归约 `sum(g)` 和 `sum(g*xhat)`，再写该行 dx。参数梯度 kernel 每线程处理一列并按行号递增的固定顺序累加 dgamma/dbeta，避免原子加法的非确定顺序，也直接覆盖预填的输出。两个阶段均为 `O(NC)`。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void dx_kernel(const float*x,const float*gamma,const float*mean,const float*rstd,const float*dy,float*dx,int N,int C){
 int row=(int)blockIdx.x,tid=(int)threadIdx.x,bs=(int)blockDim.x;if(row>=N)return;extern __shared__ float s[];float*sg=s;float*sp=s+bs;size_t b=(size_t)row*C;float a=0,p=0;
 for(int j=tid;j<C;j+=bs){float xh=(x[b+j]-mean[row])*rstd[row];float g=dy[b+j]*gamma[j];a+=g;p+=g*xh;}sg[tid]=a;sp[tid]=p;__syncthreads();
 for(int o=bs/2;o;o>>=1){if(tid<o){sg[tid]+=sg[tid+o];sp[tid]+=sp[tid+o];}__syncthreads();}
 float mg=sg[0]/C,mp=sp[0]/C;for(int j=tid;j<C;j+=bs){float xh=(x[b+j]-mean[row])*rstd[row];float g=dy[b+j]*gamma[j];dx[b+j]=rstd[row]*(g-mg-xh*mp);}}
__global__ void param_kernel(const float*x,const float*mean,const float*rstd,const float*dy,float*dg,float*db,int N,int C){
 int j=(int)blockIdx.x*(int)blockDim.x+(int)threadIdx.x;if(j>=C)return;float a=0,beta=0;for(int i=0;i<N;++i){size_t k=(size_t)i*C+j;float d=dy[k];a+=d*(x[k]-mean[i])*rstd[i];beta+=d;}dg[j]=a;db[j]=beta;}

// x, gamma, mean, rstd, dy, dx, dgamma, dbeta are device pointers
extern "C" void solve(const float* x, const float* gamma, const float* mean, const float* rstd,
                      const float* dy, float* dx, float* dgamma, float* dbeta, int N, int C) {
 int threads=256;dx_kernel<<<N,threads,2*threads*sizeof(float)>>>(x,gamma,mean,rstd,dy,dx,N,C);param_kernel<<<(C+threads-1)/threads,threads>>>(x,mean,rstd,dy,dgamma,dbeta,N,C);cudaDeviceSynchronize();
}
```
