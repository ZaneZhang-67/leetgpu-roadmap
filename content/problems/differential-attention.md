---
id: differential-attention
leetgpuId: 148
titleEn: Differential Attention
titleZh: 差分注意力
description: 计算两张因果注意力图的加权差，对完整双倍 Head 特征执行 RMS 归一化
problemUrl: https://leetgpu.com/challenges/differential-attention
updatedAt: 2026-11-12
---
## 题目定义

Q/K/V/output 形状均为 `[S,H,2D]`。Q、K 沿最后一维连续拆成前 D 维和后 D 维，分别计算两张 causal Softmax 图：

```text
A1=softmax(Q[:D]@K[:D]ᵀ/sqrt(D))
A2=softmax(Q[D:]@K[D:]ᵀ/sqrt(D))
O=(A1-lambdas[h]*A2)@V
output=(1-lambda_init)*O/sqrt(mean(O² over 2D)+1e-5)
```

λ 是直接传入的每 Head 标量，不由向量指数公式重新计算。RMSNorm 覆盖完整 `2D`，V 也保留完整 `2D`。

### 官方示例

```text
S=2,H=1,D=1,lambdas=[0.5],lambda_init=0.2
output=[[1.1314,0],[0.0445,1.1305]]
```

### 完整约束

- `1 ≤ S ≤ 8192`，`1 ≤ H ≤ 64`，`1 ≤ D ≤ 128`
- `-1 ≤ lambdas[h] ≤ 1`，`0 ≤ lambda_init < 1`
- 所有张量为 FP32；Q/K/V 来自标准正态分布，Q 可能被放大，使 score 约落在 `[-30,30]`；score 和累加使用 FP32 或更高
- O 全零时输出全零
- 性能测试为 `S=4096,H=32,D=64`
- 容差为 `1e-3`

## 模板代码解析

```cpp
#include <cuda_runtime.h>
extern "C" void solve(const float* Q,const float* K,const float* V,
 const float* lambdas,float* output,int seq_len,int num_heads,
 int head_dim,float lambda_init) {}
```

布局是 `[sequence,head,feature]`，不是 `[head,sequence,feature]`。每张图只看 Key `0..query`。

## 实现思路

`attn_part` 每个 `(query,head)` 使用一个 Block，动态 Shared Memory只保存这一行最多 S 个 score。线程协作计算 score，归约最大值和指数和，再把 Shared Memory原地改为 Softmax 权重；输出 feature 线程直接复用这组权重累加 V，不按 feature 重算 dot 或 exp。S=8192 时动态 Shared Memory为 32 KiB，加上固定归约槽仍低于 48 KiB。

同一默认流先对 Q1/K1 启动一次 `attn_part`，覆盖写入 `A1@V`；再对 Q2/K2 启动一次，以 `-lambdas[h]` 累加 `A2@V`。因此无需 `[H,S,S]` 全局临时权重。最后独立 RMS kernel 对 `2D` 输出归约平方和，按 `mean+1e-5` 归一化并乘 `1-lambda_init`。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>
__global__ void attn_part(const float*Q,const float*K,const float*V,const float*lam,float*out,int H,int D,int half,int add){extern __shared__ float score[];__shared__ float red[256],maximum,den;int q=blockIdx.x,h=blockIdx.y,tid=threadIdx.x,F=2*D,off=half*D;float scale=rsqrtf((float)D);for(int t=tid;t<=q;t+=blockDim.x){float z=0;for(int j=0;j<D;++j)z+=Q[((size_t)q*H+h)*F+off+j]*K[((size_t)t*H+h)*F+off+j];score[t]=z*scale;}__syncthreads();float v=-CUDART_INF_F;for(int t=tid;t<=q;t+=blockDim.x)v=fmaxf(v,score[t]);red[tid]=v;__syncthreads();for(int o=128;o;o>>=1){if(tid<o)red[tid]=fmaxf(red[tid],red[tid+o]);__syncthreads();}if(tid==0)maximum=red[0];__syncthreads();float sum=0;for(int t=tid;t<=q;t+=blockDim.x){float e=expf(score[t]-maximum);score[t]=e;sum+=e;}red[tid]=sum;__syncthreads();for(int o=128;o;o>>=1){if(tid<o)red[tid]+=red[tid+o];__syncthreads();}if(tid==0)den=red[0];__syncthreads();float coeff=half?-lam[h]:1.0f;size_t base=((size_t)q*H+h)*F;for(int j=tid;j<F;j+=blockDim.x){float z=0;for(int t=0;t<=q;++t)z+=score[t]*V[((size_t)t*H+h)*F+j];float value=coeff*z/den;if(add)out[base+j]+=value;else out[base+j]=value;}}
__global__ void rms_out(float*out,int H,int D,float init){__shared__ float red[256],factor;int q=blockIdx.x,h=blockIdx.y,tid=threadIdx.x,F=2*D;size_t base=((size_t)q*H+h)*F;float s=0;for(int j=tid;j<F;j+=blockDim.x)s+=out[base+j]*out[base+j];red[tid]=s;__syncthreads();for(int o=128;o;o>>=1){if(tid<o)red[tid]+=red[tid+o];__syncthreads();}if(tid==0)factor=rsqrtf(red[0]/F+1e-5f)*(1-init);__syncthreads();for(int j=tid;j<F;j+=blockDim.x)out[base+j]*=factor;}
extern "C" void solve(const float*Q,const float*K,const float*V,const float*l,float*out,int S,int H,int D,float init){size_t sh=(size_t)S*sizeof(float);attn_part<<<dim3(S,H),256,sh>>>(Q,K,V,l,out,H,D,0,0);attn_part<<<dim3(S,H),256,sh>>>(Q,K,V,l,out,H,D,1,1);rms_out<<<dim3(S,H),256>>>(out,H,D,init);cudaDeviceSynchronize();}
```
