---
id: distillation-kl-divergence
leetgpuId: 150
titleEn: Distillation KL Divergence
titleZh: 蒸馏 KL 散度
description: 使用 CUDA 稳定计算教师与学生温度软化分布的逐行前向 KL 散度
problemUrl: https://leetgpu.com/challenges/distillation-kl-divergence
updatedAt: 2026-11-12
---
## 题目定义

输入是两组 `(N,V)` FP32 logits，而不是概率或对数概率。令 `a=teacher_logits/T`、`b=student_logits/T`，逐行计算：

```text
p=softmax(a), q=softmax(b)
kl[i]=Σ p[i,j]*(log(p[i,j])-log(q[i,j]))
```

方向是 `KL(teacher || student)`。输出不做 batch 归约，也不乘常见蒸馏损失中的 `T²`。

### 示例

```text
teacher=[[1,2,3],[0,0,0]], student=[[3,2,1],[0,0,0]]
T=1 → kl=[1.1504,0]
T=2 → kl=[0.3202,0]
```

**约束与测试**

- `1≤N≤65,536`，`1≤V≤262,144`，`0.5≤T≤10`
- logits 为 `[-500,500]` 内的 FP32
- 测试覆盖 `V=1`、相同分布的行偏移、反向 KL 区分、大 logits、小温度和超长行
- 性能测试 `N=2048,V=128256,T=2`，容差 `atol=rtol=1e-3`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// teacher_logits, student_logits, kl are device pointers
extern "C" void solve(const float* teacher_logits, const float* student_logits, float* kl, int N,
                      int V, float temperature) {}
```

官方 starter 只给出空入口。kl 预填了垃圾值，每一行都要直接写出结果。

## 实现思路

利用恒等式：

```text
KL = Σ p*(a-b) - LSE(a) + LSE(b)
```

每个 Block 处理一行，分三遍扫描：先分别求 a、b 的最大值，再求两个平移后的指数和，最后求教师概率加权的 `a-b`。最大值平移可防止 `exp` 溢出；整个过程只使用固定大小共享内存，不会把最长可达 262144 的整行放入片上存储。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void kl_kernel(const float* t,const float* u,float* out,int N,int V,float invT){
 int row=(int)blockIdx.x,tid=(int)threadIdx.x,bs=(int)blockDim.x;if(row>=N)return;extern __shared__ float s[];float*sa=s;float*sb=s+bs;size_t base=(size_t)row*V;
 float ma=-INFINITY,mb=-INFINITY;for(int j=tid;j<V;j+=bs){ma=fmaxf(ma,t[base+j]*invT);mb=fmaxf(mb,u[base+j]*invT);}sa[tid]=ma;sb[tid]=mb;__syncthreads();
 for(int o=bs/2;o;o>>=1){if(tid<o){sa[tid]=fmaxf(sa[tid],sa[tid+o]);sb[tid]=fmaxf(sb[tid],sb[tid+o]);}__syncthreads();}ma=sa[0];mb=sb[0];
 float za=0,zb=0;for(int j=tid;j<V;j+=bs){za+=expf(t[base+j]*invT-ma);zb+=expf(u[base+j]*invT-mb);}sa[tid]=za;sb[tid]=zb;__syncthreads();
 for(int o=bs/2;o;o>>=1){if(tid<o){sa[tid]+=sa[tid+o];sb[tid]+=sb[tid+o];}__syncthreads();}za=sa[0];zb=sb[0];
 float w=0;for(int j=tid;j<V;j+=bs){float a=t[base+j]*invT;float b=u[base+j]*invT;w+=expf(a-ma)*(a-b);}sa[tid]=w;__syncthreads();
 for(int o=bs/2;o;o>>=1){if(tid<o)sa[tid]+=sa[tid+o];__syncthreads();}
 if(tid==0)out[row]=sa[0]/za-(ma+logf(za))+(mb+logf(zb));
}

// teacher_logits, student_logits, kl are device pointers
extern "C" void solve(const float* teacher_logits, const float* student_logits, float* kl, int N,
                      int V, float temperature) {
 int threads=256;kl_kernel<<<N,threads,2*threads*sizeof(float)>>>(teacher_logits,student_logits,kl,N,V,1.0f/temperature);cudaDeviceSynchronize();
}
```
