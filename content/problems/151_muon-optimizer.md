---
id: muon-optimizer
leetgpuId: 151
titleEn: Muon Optimizer
titleZh: Muon 优化器
description: 使用 CUDA 实现动量更新方向和五步 Newton–Schulz 正交化
problemUrl: https://leetgpu.com/challenges/muon-optimizer
updatedAt: 2026-11-12
---
## 题目定义

输入 G、Buf 均为只读 `(M,N)` FP32 矩阵。先计算 `V=momentum*Buf+G`，再令：

```text
X=V/(||V||F+1e-7)
重复5次：A=X*Xᵀ，B=-4.775*A+2.0315*A²，X=3.4445*X+B*X
output=X*sqrt(max(M,N))
```

当 `M>N` 时，先转置 X，使 Gram 矩阵使用较小维度。题目只计算更新方向：不修改 Buf，不更新参数，也不包含权重衰减或 Nesterov。

### 示例

```text
M=2,N=3,momentum=0.9
G=[[0.5,-1,2],[1,0,-0.5]], Buf=[[0.1,0.2,0],[-0.3,0.4,0.1]]
V=[[0.59,-0.82,2],[0.73,0.36,-0.41]]
output≈[[0.5427371,-0.4092258,1.1156158],
        [1.7146817,0.5666907,-0.3777507]]
```

**约束与测试**

- `1≤M,N≤1024`，`0≤momentum<1`，全部张量为 FP32
- 必须恰好执行五步指定多项式；不是精确 SVD 极分解
- 测试覆盖宽矩阵、高矩阵、低秩、零输入、`1e-8` 小输入、单行和单列
- 性能测试 `M=N=1024,momentum=0.9`，容差 `atol=rtol=1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// G, Buf, output are device pointers
extern "C" void solve(const float* G, const float* Buf, float* output, int M, int N,
                      float momentum) {}
```

官方 starter 只给出空入口。G 与 Buf 只读，output 预填垃圾值，必须覆盖全部元素。

## 实现思路

参考实现保留纯 CUDA FP32 累加顺序。先构造 V，并用每行分块归约、随后单 Block 汇总行和，贴合运行时“两级 FP32 求和”定义的 Frobenius 范数。若 `M>N`，工作矩阵按转置布局保存；设其形状为 `R×C`，始终满足 `R≤C`。

每轮用朴素 tiled-free kernel 依次计算 `A=X*Xᵀ`、`A²`、`B=-4.775A+2.0315A²` 和 `X=3.4445X+BX`。它执行指定的 15 次矩阵乘法等价链，内存规模可覆盖 1024×1024。生产性能通常还会采用共享内存分块，但不能启用 TF32 或半精度运算。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void init(const float*G,const float*Buf,float*X,float*rows,int M,int N,float mu,bool tr){int r=(int)blockIdx.x,t=(int)threadIdx.x,bs=(int)blockDim.x;float s=0;for(int j=t;j<N;j+=bs){float v=mu*Buf[(size_t)r*N+j]+G[(size_t)r*N+j];if(tr)X[(size_t)j*M+r]=v;else X[(size_t)r*N+j]=v;s+=v*v;}extern __shared__ float q[];q[t]=s;__syncthreads();for(int o=bs/2;o;o>>=1){if(t<o)q[t]+=q[t+o];__syncthreads();}if(t==0)rows[r]=q[0];}
__global__ void norm(float*X,const float*rows,int M,int N){extern __shared__ float q[];int t=(int)threadIdx.x,bs=(int)blockDim.x;float s=0;for(int i=t;i<M;i+=bs)s+=rows[i];q[t]=s;__syncthreads();for(int o=bs/2;o;o>>=1){if(t<o)q[t]+=q[t+o];__syncthreads();}float d=sqrtf(q[0])+1e-7f;for(size_t i=(size_t)t;i<(size_t)M*N;i+=(size_t)bs)X[i]/=d;}
__global__ void gram(const float*X,float*A,int R,int C){int i=blockIdx.y*blockDim.y+threadIdx.y,j=blockIdx.x*blockDim.x+threadIdx.x;if(i<R&&j<R){float s=0;for(int k=0;k<C;++k)s+=X[(size_t)i*C+k]*X[(size_t)j*C+k];A[(size_t)i*R+j]=s;}}
__global__ void square(const float*A,float*A2,int R){int i=blockIdx.y*blockDim.y+threadIdx.y,j=blockIdx.x*blockDim.x+threadIdx.x;if(i<R&&j<R){float s=0;for(int k=0;k<R;++k)s+=A[(size_t)i*R+k]*A[(size_t)k*R+j];A2[(size_t)i*R+j]=s;}}
__global__ void poly(const float*A,const float*A2,float*B,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)B[i]=-4.7750f*A[i]+2.0315f*A2[i];}
__global__ void update(const float*X,const float*B,float*Y,int R,int C){int i=blockIdx.y*blockDim.y+threadIdx.y,j=blockIdx.x*blockDim.x+threadIdx.x;if(i<R&&j<C){float s=0;for(int k=0;k<R;++k)s+=B[(size_t)i*R+k]*X[(size_t)k*C+j];Y[(size_t)i*C+j]=3.4445f*X[(size_t)i*C+j]+s;}}
__global__ void finish(const float*X,float*out,int M,int N,bool tr,float scale){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,total=(size_t)M*N;for(;i<total;i+=(size_t)blockDim.x*gridDim.x){int r=i/N,c=i%N;out[i]=(tr?X[(size_t)c*M+r]:X[i])*scale;}}

// G, Buf, output are device pointers
extern "C" void solve(const float* G, const float* Buf, float* output, int M, int N,
                      float momentum) {
 bool tr=M>N;int R=tr?N:M,C=tr?M:N;size_t rc=(size_t)M*N*sizeof(float),rr=(size_t)R*R*sizeof(float);float *X,*Y,*A,*A2,*B,*rows;cudaMalloc(&X,rc);cudaMalloc(&Y,rc);cudaMalloc(&A,rr);cudaMalloc(&A2,rr);cudaMalloc(&B,rr);cudaMalloc(&rows,(size_t)M*sizeof(float));int th=256;init<<<M,th,th*sizeof(float)>>>(G,Buf,X,rows,M,N,momentum,tr);norm<<<1,th,th*sizeof(float)>>>(X,rows,M,N);
 dim3 block(16,16),gA((R+15)/16,(R+15)/16),gX((C+15)/16,(R+15)/16);for(int it=0;it<5;++it){gram<<<gA,block>>>(X,A,R,C);square<<<gA,block>>>(A,A2,R);poly<<<(R*R+th-1)/th,th>>>(A,A2,B,R*R);update<<<gX,block>>>(X,B,Y,R,C);float*tmp=X;X=Y;Y=tmp;}int blocks=(M*N+th-1)/th;if(blocks>4096)blocks=4096;finish<<<blocks,th>>>(X,output,M,N,tr,sqrtf((float)(M>N?M:N)));cudaFree(X);cudaFree(Y);cudaFree(A);cudaFree(A2);cudaFree(B);cudaFree(rows);cudaDeviceSynchronize();
}
```
