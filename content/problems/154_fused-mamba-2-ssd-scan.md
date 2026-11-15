---
id: fused-mamba-2-ssd-scan
leetgpuId: 154
titleEn: Fused Mamba-2 SSD Scan
titleZh: 融合 Mamba-2 SSD 扫描
description: 使用 CUDA 分块状态递推计算 Mamba-2 选择性状态空间层
problemUrl: https://leetgpu.com/challenges/fused-mamba-2-ssd-scan
updatedAt: 2026-11-12
---
## 题目定义

每个 `(batch,head)` 维护 `S(head_dim,state_dim)`，初值为零。Head h 使用连续分组 `g=h/(num_heads/num_groups)`：

```text
delta=softplus(dt+dt_bias)
decay=exp(delta*A[h])
S=decay*S + delta*x[:,None]*B[None,:]
y=(S*C + D[h]*x)*silu(z)
```

状态先衰减再加入当前 Token，当前更新参与当前输出。Batch 之间互不共享状态。

### 示例

官方示例 `delta=ln(2)`、`A=-1`，故每步 decay=0.5。给定三步 x、B、C 后，输出为 `[[0.8722605,1.7445210],[0.2533656,-0.3655293],[1.7445210,-0.2533656]]`。

约束：全 FP32；`batch≤4`、`seq_len≤4096`、`groups≤heads≤32` 且整除、`head_dim,state_dim≤128`、`-16≤A<0`、`dt+dt_bias∈[-30,30]`。性能用例为 `2×4096×32×64`、4 groups、state_dim=128，容差 `1e-4`。

## 模板代码解析

签名为：

```text
extern "C" void solve(const float* x, const float* dt, const float* dt_bias,
 const float* A, const float* B, const float* C, const float* D,
 const float* z, float* y, int batch, int seq_len, int num_heads,
 int head_dim, int num_groups, int state_dim)
```

x/z/y 是 `(b,L,H,P)`；dt 是 `(b,L,H)`；B/C 是 `(b,L,G,N)`。题目没有 initial/final state 指针，每条序列总从零开始，也不输出最终状态。

## 实现思路

参考实现以 64 Token 为 Chunk。第一阶段让每个 `(batch,head,chunk,p)` Block 顺序扫描该 Chunk，线程并行 state_dim，生成“零初态下的局部输出”、Chunk 最终状态和总 decay。第二阶段每个 `(batch,head,p)` Block 仅扫描至多 64 个 Chunk，记录每个 Chunk 的输入 carry。第三阶段并行修正各 Token：carry 在 Chunk 内乘截至当前 Token 的 decay，再与 C 点积，加到局部 SSM 输出，最后融合 D skip 与 SiLU gate。

这样不会串行全部输出：Chunk 内不同 p、head、batch 并行，跨 Chunk 只递推最多 64 项。尾 Chunk 严格使用 `min(64,L-start)`，不让 padding 更新状态；Chunk carry 先乘当前 Chunk 总 decay再加局部末状态，与逐 Token recurrence 一致。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <cmath>
__device__ float sp(float v){return v>20?v:log1pf(expf(v));}
__global__ void local(const float*x,const float*dt,const float*db,const float*A,const float*B,const float*C,float*base,float*end,float*cd,int L,int H,int P,int G,int N,int NC){
 int p=blockIdx.x,h=blockIdx.y%H,b=blockIdx.y/H,ch=blockIdx.z,t=threadIdx.x,g=h/(H/G),start=ch*64,len=min(64,L-start);extern __shared__ float S[];for(int n=t;n<N;n+=blockDim.x)S[n]=0;__shared__ float pref; if(t==0)pref=1;__syncthreads();
 for(int j=0;j<len;++j){int q=start+j;float delta=sp(dt[((size_t)b*L+q)*H+h]+db[h]),dec=expf(delta*A[h]),xv=x[(((size_t)b*L+q)*H+h)*P+p];for(int n=t;n<N;n+=blockDim.x)S[n]=dec*S[n]+delta*xv*B[(((size_t)b*L+q)*G+g)*N+n];if(t==0)pref*=dec;__syncthreads();float sum=0;for(int n=t;n<N;n+=blockDim.x)sum+=S[n]*C[(((size_t)b*L+q)*G+g)*N+n];__shared__ float r[128];r[t]=sum;__syncthreads();for(int o=64;o;o>>=1){if(t<o)r[t]+=r[t+o];__syncthreads();}if(t==0)base[(((size_t)b*L+q)*H+h)*P+p]=r[0];__syncthreads();}
 for(int n=t;n<N;n+=blockDim.x)end[((((size_t)b*H+h)*NC+ch)*P+p)*N+n]=S[n];if(t==0&&p==0)cd[((size_t)b*H+h)*NC+ch]=pref;
}
__global__ void carries(const float*end,const float*cd,float*carry,int H,int P,int N,int NC){int p=blockIdx.x,h=blockIdx.y%H,b=blockIdx.y/H,t=threadIdx.x;extern __shared__ float S[];for(int n=t;n<N;n+=blockDim.x)S[n]=0;__syncthreads();for(int ch=0;ch<NC;++ch){for(int n=t;n<N;n+=blockDim.x){size_t o=((((size_t)b*H+h)*NC+ch)*P+p)*N+n;carry[o]=S[n];S[n]=cd[((size_t)b*H+h)*NC+ch]*S[n]+end[o];}__syncthreads();}}
__global__ void finish(const float*x,const float*dt,const float*db,const float*A,const float*C,const float*D,const float*z,const float*base,const float*carry,float*y,int batch,int L,int H,int P,int G,int N,int NC){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,total=(size_t)batch*L*H*P;if(i>=total)return;int p=i%P,h=(i/P)%H,q=(i/(P*H))%L,b=i/(P*H*L),g=h/(H/G),ch=q/64,start=ch*64;float decay=1;for(int s=start;s<=q;++s)decay*=expf(sp(dt[((size_t)b*L+s)*H+h]+db[h])*A[h]);float off=0;for(int n=0;n<N;++n)off+=carry[((((size_t)b*H+h)*NC+ch)*P+p)*N+n]*C[(((size_t)b*L+q)*G+g)*N+n];float xv=x[i],gate=z[i]/(1+expf(-z[i]));y[i]=(base[i]+decay*off+D[h]*xv)*gate;}
extern "C" void solve(const float*x,const float*dt,const float*db,const float*A,const float*B,const float*C,const float*D,const float*z,float*y,int batch,int L,int H,int P,int G,int N){int NC=(L+63)/64;size_t sn=(size_t)batch*H*NC*P*N;float *base,*end,*carry,*cd;cudaMalloc(&base,(size_t)batch*L*H*P*4);cudaMalloc(&end,sn*4);cudaMalloc(&carry,sn*4);cudaMalloc(&cd,(size_t)batch*H*NC*4);local<<<dim3(P,batch*H,NC),128,N*4>>>(x,dt,db,A,B,C,base,end,cd,L,H,P,G,N,NC);carries<<<dim3(P,batch*H),128,N*4>>>(end,cd,carry,H,P,N,NC);size_t total=(size_t)batch*L*H*P;finish<<<(total+255)/256,256>>>(x,dt,db,A,C,D,z,base,carry,y,batch,L,H,P,G,N,NC);cudaDeviceSynchronize();cudaFree(cd);cudaFree(carry);cudaFree(end);cudaFree(base);}
```
