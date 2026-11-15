---
id: moe-expert-weight-gradient
leetgpuId: 155
titleEn: MoE Expert Weight Gradient
titleZh: MoE 专家权重梯度
description: 按已排序的专家路由区间，对每个专家确定性归约激活与上游梯度的外积
problemUrl: https://leetgpu.com/challenges/moe-expert-weight-gradient
updatedAt: 2026-11-12
---
## 题目定义

路由后的 token 已按专家连续排列。专家 `g` 占用 `[group_offsets[g],group_offsets[g+1])`，其梯度布局为 `[G,K,N]` 行主序：

```text
dW[(g*K+k)*N+n] = Σm X[m,k]*dY[m,n]
```

归约范围仅是该专家的行。Router top-k 权重已经折入 dY，不应再次乘 gate weight。空专家必须把完整 `[K,N]` 梯度写为 0，因为输入 dW 预填了非零垃圾值。

### 官方示例

```text
G=3, offsets=[0,2,2,3]
dW[0]=[[7,3],[0,-1]]
dW[1]=[[0,0],[0,0]]
dW[2]=[[0,0],[-4,12]]
```

### 完整约束

- `1 ≤ G ≤ 128`，`1 ≤ M_total ≤ 65536`
- `1 ≤ K,N ≤ 4096`
- offsets 长度 G+1，首项 0、末项 M_total，单调非递减，允许空组
- X/dY/dW 为 FP32，offsets 为 int32；X 与 dY 元素在 `[-1,1]`
- 使用 FP32 或更高精度，不把操作数降为 TF32/FP16/BF16
- 性能测试为 `G=64,M_total=32768,K=1024,N=512`，含两个空专家
- 容差为 `1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>
extern "C" void solve(const float* X,const float* dY,const int* group_offsets,
 float* dW,int G,int M_total,int K,int N) {}
```

`M_total` 是所有 top-k 路由副本合计的行数，不是原 token 数。接口没有 route weights 或 expert ids；分组和权重信息已分别体现在 offsets 与 dY 中。

## 实现思路

二维输出 tile 与 expert 组成三维 Grid。每个线程负责一个 `(g,k,n)`，从 `start` 到 `end` 按固定行序进行 double 累加，最后转为 FP32 覆盖 dW。这样结果确定，不依赖 atomic 的调度顺序；空专家循环零次，自然写出 0。

性能用例产生约 3355 万个输出线程，每线程只处理对应专家的路由行。偏斜负载只改变单个线程的归约长度，不改变布局或权重语义。

## 参考实现

```cpp
#include <cuda_runtime.h>
__global__ void expert_grad(const float*X,const float*dY,const int*off,float*dW,int K,int N){int n=blockIdx.x*blockDim.x+threadIdx.x,k=blockIdx.y*blockDim.y+threadIdx.y,g=blockIdx.z;if(k<K&&n<N){int begin=off[g],end=off[g+1];double sum=0.0;for(int m=begin;m<end;++m)sum+=(double)X[m*K+k]*(double)dY[m*N+n];dW[((size_t)g*K+k)*N+n]=(float)sum;}}
extern "C" void solve(const float*X,const float*dY,const int*off,float*dW,int G,int M_total,int K,int N){(void)M_total;dim3 block(16,16),grid((N+15)/16,(K+15)/16,G);expert_grad<<<grid,block>>>(X,dY,off,dW,K,N);cudaDeviceSynchronize();}
```
