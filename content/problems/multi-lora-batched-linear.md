---
id: multi-lora-batched-linear
leetgpuId: 133
titleEn: Multi-LoRA Batched Linear
titleZh: 多 LoRA 批量线性层
description: 为批内每个 token 按 adapter id 选择独立的低秩增量并与共享基础投影融合
problemUrl: https://leetgpu.com/challenges/multi-lora-batched-linear
updatedAt: 2026-11-12
---
## 题目定义

`x[T,K]` 共享基础权重 `W[K,N]`。token `t` 的 adapter id 为 `a` 时：

```text
y[t] = x[t] @ W + scale * (x[t] @ A[a]) @ B[a]
```

其中 `A[L,K,R]`、`B[L,R,N]`。`adapter_ids[t]=-1` 表示只使用基础投影。批内 token 没有按 adapter 排序，每行必须独立读取自己的 id。

### 官方示例

```text
T=3,K=2,N=2,R=1,L=2,scale=0.5
adapter_ids=[0,1,-1]
y=[[1.5,2.5],[5.0,2.0],[1.0,1.0]]
```

### 完整约束

- `1 ≤ T ≤ 4096`，`1 ≤ K,N ≤ 8192`
- `1 ≤ R,L ≤ 64`
- `-1 ≤ adapter_ids[t] < L`
- 数据张量为 FP32，adapter id 为 int32
- 性能测试为 `T=1024,K=N=4096,R=16,L=8`
- 容差为 `1e-3`

## 模板代码解析

```cpp
#include <cuda_runtime.h>
extern "C" void solve(const float* x,const float* W,const float* A,const float* B,
 const int* adapter_ids,float* y,int T,int K,int N,int R,int L,float scale) {}
```

A 的索引是 `((adapter*K+k)*R+r)`，B 是 `((adapter*R+r)*N+n)`。`L` 限定 adapter id，不是矩阵维度。

## 实现思路

先并行计算 base GEMM，每线程负责 `(t,n)`；生产级实现可进一步用 K 维 tiling 在 Shared Memory复用 x/W。随后计算 shrink `[T,R]`：仅当 id 非负时读取对应 A；最后并行 expand 并乘 `scale` 加入 y。这样不会为 `-1` 访问 adapter 0，也支持 adapter id 交错。性能规模的三个阶段都有二维 Grid，不由单线程承担矩阵乘法。

## 参考实现

```cpp
#include <cuda_runtime.h>
__global__ void base(const float*x,const float*W,float*y,int K,int N){int n=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y;if(n<N){float s=0;for(int k=0;k<K;++k)s+=x[t*K+k]*W[k*N+n];y[t*N+n]=s;}}
__global__ void shrink(const float*x,const float*A,const int*ids,float*z,int K,int R){int r=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y;if(r<R){int a=ids[t];float s=0;if(a>=0)for(int k=0;k<K;++k)s+=x[t*K+k]*A[(a*K+k)*R+r];z[t*R+r]=s;}}
__global__ void expand(const float*z,const float*B,const int*ids,float*y,int N,int R,float scale){int n=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y;if(n<N){int a=ids[t];if(a>=0){float s=0;for(int r=0;r<R;++r)s+=z[t*R+r]*B[(a*R+r)*N+n];y[t*N+n]+=scale*s;}}}
extern "C" void solve(const float*x,const float*W,const float*A,const float*B,const int*ids,float*y,int T,int K,int N,int R,int L,float scale){(void)L;float*z;cudaMalloc(&z,(size_t)T*R*sizeof(float));base<<<dim3((N+255)/256,T),256>>>(x,W,y,K,N);shrink<<<dim3((R+255)/256,T),256>>>(x,A,ids,z,K,R);expand<<<dim3((N+255)/256,T),256>>>(z,B,ids,y,N,R,scale);cudaFree(z);cudaDeviceSynchronize();}
```
