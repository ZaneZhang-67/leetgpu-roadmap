---
id: fused-moe-swiglu-layer
leetgpuId: 130
titleEn: Fused MoE SwiGLU Layer
titleZh: 融合 MoE SwiGLU 层
description: 按 Top-K 路由计算专家 SwiGLU 并使用原始路由权重合并输出
problemUrl: https://leetgpu.com/challenges/fused-moe-swiglu-layer
updatedAt: 2026-11-12
---
## 题目定义

每个 Token 路由至 K 个互异专家。专家权重布局为 `w_gate(E,H,I)`、`w_up(E,H,I)`、`w_down(E,I,H)`：

```text
hidden=SiLU(x W_gate[e])⊙(x W_up[e])
output[t]=Σk topk_weights[t,k]*(hidden W_down[e]), e=topk_ids[t,k]
```

路由权重按原值使用，可能未归一化、为负数或零；空专家不产生工作。题面没有 capacity 或丢弃规则。

### 示例

官方 `T=H=I=E=2,K=1` 示例中，Token 0 经专家 0、权重 1，Token 1 经专家 1、权重 0.5，输出 `[[0.7311,0],[0,0.7311]]`。

约束 `T≤8192`、`H,I≤4096`、`K≤E≤64`。性能用例 `T=4096,H=1024,I=1408,E=16,K=2`。

## 模板代码解析

签名为：

```text
extern "C" void solve(const float* x, const int* topk_ids,
 const float* topk_weights, const float* w_gate, const float* w_up,
 const float* w_down, float* output, int T, int H, int I, int E, int K)
```

矩阵是 `(H,I)` 与 `(I,H)`，不是常见的 `(out,in)` 方向。

## 实现思路

先把每个 `(token,route)` 展开为 R=T×K 行。每个 Block 固定处理一个 route，16×16 个线程分别覆盖 16 个输出通道和 K 维分片；这样整个 Block 只读取该 route 对应专家的权重，不会让同一 Shared Memory tile 混入多个专家。Gate 与 Up 共用 x 分片并融合 SiLU，Down 阶段同样保持一个 Block 一个 route，再以路由权重 `atomicAdd` 到已清零的 Token 输出。

中间缓冲大小为 `T×K×I`，性能用例约 44 MiB。Grid 的 x 维完整覆盖 R，包括最后一个 route；y 维覆盖 I 或 H 的通道 Tile。x 维上限足以容纳最大 `R=8192×64=524288`，每个 route 在输出通道上有多个 Block 并行。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <cmath>
__global__ void up(const float*x,const int*ids,const float*wg,const float*wu,float*h,int K,int H,int I){
 __shared__ float part_g[16][16],part_u[16][16];int lane=threadIdx.x,kpart=threadIdx.y,r=blockIdx.x,i=blockIdx.y*16+lane,t=r/K,e=ids[r];float g=0,u=0;
 for(int k=kpart;k<H;k+=16){float xv=x[(size_t)t*H+k];if(i<I){g+=xv*wg[((size_t)e*H+k)*I+i];u+=xv*wu[((size_t)e*H+k)*I+i];}}
 part_g[kpart][lane]=g;part_u[kpart][lane]=u;__syncthreads();for(int off=8;off;off>>=1){if(kpart<off){part_g[kpart][lane]+=part_g[kpart+off][lane];part_u[kpart][lane]+=part_u[kpart+off][lane];}__syncthreads();}
 if(kpart==0&&i<I){g=part_g[0][lane];h[(size_t)r*I+i]=g/(1+expf(-g))*part_u[0][lane];}}
__global__ void down(const float*h,const int*ids,const float*coef,const float*wd,float*out,int K,int H,int I){
 __shared__ float part[16][16];int lane=threadIdx.x,kpart=threadIdx.y,r=blockIdx.x,c=blockIdx.y*16+lane,e=ids[r];float s=0;for(int i=kpart;i<I;i+=16)if(c<H)s+=h[(size_t)r*I+i]*wd[((size_t)e*I+i)*H+c];part[kpart][lane]=s;__syncthreads();for(int off=8;off;off>>=1){if(kpart<off)part[kpart][lane]+=part[kpart+off][lane];__syncthreads();}if(kpart==0&&c<H)atomicAdd(out+(size_t)(r/K)*H+c,coef[r]*part[0][lane]);}
extern "C" void solve(const float*x,const int*topk_ids,const float*topk_weights,const float*w_gate,const float*w_up,const float*w_down,float*output,int T,int H,int I,int E,int K){(void)E;int R=T*K;float*h;cudaMalloc(&h,(size_t)R*I*4);cudaMemset(output,0,(size_t)T*H*4);dim3 th(16,16);up<<<dim3(R,(I+15)/16),th>>>(x,topk_ids,w_gate,w_up,h,K,H,I);down<<<dim3(R,(H+15)/16),th>>>(h,topk_ids,topk_weights,w_down,output,K,H,I);cudaDeviceSynchronize();cudaFree(h);}
```
