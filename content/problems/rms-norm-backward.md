---
id: rms-norm-backward
leetgpuId: 142
titleEn: RMS Norm Backward
titleZh: RMS Norm 反向传播
description: 使用 CUDA 计算 RMSNorm 输入梯度与跨行权重梯度
problemUrl: https://leetgpu.com/challenges/rms-norm-backward
updatedAt: 2026-11-12
---
## 题目定义

对 `x(N,C)`、weight(C) 和上游梯度 dy(N,C)，重新计算前向统计并输出 dx 与 dweight：

```text
r = rsqrt(mean(x²)+eps), xhat=x*r, g=dy*weight
proj=mean(g*xhat)
dx=r*(g-xhat*proj)
dweight[j]=Σ dy[i,j]*xhat[i,j]
```

两个输出初始含未定义数据，必须完整覆盖。

### 示例

```text
N=2,C=4,eps=1e-5
x=[[1,-1,1,-1],[2,0,0,0]], weight=[1,2,0.5,1]
dy=[[1,1,1,1],[1,0,0,1]]
dx≈[[1.375,1.625,0.875,0.625],[0,0,0,1]]
dweight≈[3,-1,1,-1]
```

**约束与测试**

- `1≤N≤65,536`，`1≤C≤16,384`，全部为 FP32
- `x∈[-250,250]`，`dy∈[-10,10]`，`weight∈[-2,2]`，`1e-6≤eps≤1`
- 测试覆盖零 x、零 dy、相关梯度、非二次幂和奇数行数
- 性能测试 `N=16,384,C=4096`，容差 `atol=rtol=1e-3`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// x, weight, dy, dx, dweight are device pointers
extern "C" void solve(const float* x, const float* weight, const float* dy, float* dx,
                      float* dweight, int N, int C, float eps) {}
```

官方 starter 只提供空入口。rstd 未保存，必须从 x 重新计算；dx 按行归约，dweight 按列归约。

## 实现思路

第一个 kernel 每个 Block 处理一行，用共享内存归约 `sum(x²)` 得到 rstd，将它同时保存到共享标量和临时数组 `row_rstd[N]`；随后归约 `sum(dy*weight*xhat)` 得到 proj，最后写 dx。独立的共享标量可避免第二次归约覆盖 `s[0]` 后误读 rstd。

第二个 kernel 每个线程负责一列 j，按 `i=0..N-1` 的固定顺序读取 `row_rstd[i]` 并累加 dweight。这样不使用原子操作，也不重复计算每行平方和，总复杂度为 `O(NC)`。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void dx_kernel(const float* x, const float* weight, const float* dy,
                          float* dx, float* row_rstd, int N, int C, float eps) {
    int row = (int)blockIdx.x, tid = (int)threadIdx.x, bs = (int)blockDim.x;
    if (row >= N) return;
    extern __shared__ float s[];
    __shared__ float saved_rstd;
    size_t base = (size_t)row * C;
    float sum = 0.0f;
    for (int j = tid; j < C; j += bs) sum += x[base+j]*x[base+j];
    s[tid] = sum; __syncthreads();
    for (int o=bs/2;o;o>>=1){if(tid<o)s[tid]+=s[tid+o];__syncthreads();}
    if(tid==0){saved_rstd=rsqrtf(s[0]/C+eps);row_rstd[row]=saved_rstd;}__syncthreads();
    float r=saved_rstd, proj=0.0f;
    for(int j=tid;j<C;j+=bs){float xh=x[base+j]*r;proj+=dy[base+j]*weight[j]*xh;}
    s[tid]=proj;__syncthreads();
    for(int o=bs/2;o;o>>=1){if(tid<o)s[tid]+=s[tid+o];__syncthreads();}
    proj=s[0]/C;
    for(int j=tid;j<C;j+=bs){float xh=x[base+j]*r;float g=dy[base+j]*weight[j];dx[base+j]=r*(g-xh*proj);}
}

__global__ void dw_kernel(const float* x,const float* dy,const float* row_rstd,float* dw,int N,int C){
    int j=(int)blockIdx.x*(int)blockDim.x+(int)threadIdx.x;if(j>=C)return;float out=0.0f;
    for(int i=0;i<N;++i){size_t k=(size_t)i*C+j;out+=dy[k]*x[k]*row_rstd[i];}dw[j]=out;
}

// x, weight, dy, dx, dweight are device pointers
extern "C" void solve(const float* x, const float* weight, const float* dy, float* dx,
                      float* dweight, int N, int C, float eps) {
    float* row_rstd;cudaMalloc(&row_rstd,(size_t)N*sizeof(float));int threads=256;
    dx_kernel<<<N,threads,threads*sizeof(float)>>>(x,weight,dy,dx,row_rstd,N,C,eps);
    dw_kernel<<<(C+threads-1)/threads,threads>>>(x,dy,row_rstd,dweight,N,C);
    cudaFree(row_rstd);cudaDeviceSynchronize();
}
```
