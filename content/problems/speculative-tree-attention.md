---
id: speculative-tree-attention
leetgpuId: 132
titleEn: Speculative Tree Attention
titleZh: 推测树注意力
description: 让每个候选节点只关注共享前缀及自己的祖先路径，并计算稳定缩放点积注意力
problemUrl: https://leetgpu.com/challenges/speculative-tree-attention
updatedAt: 2026-11-12
---
## 题目定义

`P` 个缓存 token 由所有树节点共享。候选节点 `i` 只能关注全部缓存，以及从根到 `i` 的祖先和自身；它不能看到兄弟、其他根或后代。`parents[i]<i`，根节点的 parent 为 `-1`。

每个 Head 在允许集合上计算 `Q·K/sqrt(D)`、稳定 Softmax 和 V 加权，输出形状 `[T,H,D]`。

### 官方示例

```text
P=1,T=3,H=1,D=2,parents=[-1,0,0]
node0 sees {cache,0}
node1 sees {cache,0,1}
node2 sees {cache,0,2}
output=[[1.6605,1.3395],[1.4011,1.5989],[2.0000,1.1978]]
```

### 完整约束

- `1 ≤ P ≤ 16384`，`1 ≤ T ≤ 128`
- `1 ≤ H ≤ 64`，`1 ≤ D ≤ 256`
- `-1 ≤ parents[i] < i`，输入值范围 `[-6,6]`
- 除 parents 为 int32 外，所有张量为 FP32
- 性能测试为 `P=8192,T=64,H=32,D=128`
- 容差为 `1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>
extern "C" void solve(const float* Q,const float* K_cache,const float* V_cache,
 const float* K_tree,const float* V_tree,const int* parents,float* output,
 int P,int T,int H,int D) {}
```

缓存布局为 `[P,H,D]`，树张量为 `[T,H,D]`。树 mask 由 parents 动态决定，不能用普通下三角 causal mask 代替。

## 实现思路

第一阶段每个 `(node,head)` 使用一个 Block。线程先沿 parent 链构造长度至多 128 的祖先标记，再协作计算每个允许位置的 score，直接写入全局临时数组。

第二阶段以相同 Grid 并行归约 score 最大值与指数和，并把未归一化权重写回；随后线程按输出 feature 并行累加 V。Block 只使用 256 个 FP32 归约槽和 128 字节祖先标记，不依赖超过默认上限的动态 Shared Memory。全局权重为 `T*H*(P+T)`，性能规模约 64.5 MiB（67.6 MB）；score 计算不再额外乘输出维 D。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>
__global__ void tree_scores(const float*Q,const float*Kc,const float*Kt,const int*par,float*w,int P,int T,int H,int D){__shared__ unsigned char allowed[128];int node=(int)blockIdx.x,h=(int)blockIdx.y,tid=(int)threadIdx.x,total=P+T;if(tid<T)allowed[tid]=0;__syncthreads();if(tid==0)for(int p=node;p>=0;p=par[p])allowed[p]=1;__syncthreads();float scale=rsqrtf((float)D);for(int s=tid;s<total;s+=(int)blockDim.x){size_t at=((size_t)node*H+h)*total+s;if(s>=P&&!allowed[s-P]){w[at]=-CUDART_INF_F;continue;}const float*k=s<P?Kc+((size_t)s*H+h)*D:Kt+((size_t)(s-P)*H+h)*D;float z=0;for(int d=0;d<D;++d)z+=Q[((size_t)node*H+h)*D+d]*k[d];w[at]=z*scale;}}
__global__ void tree_finish(const float*Vc,const float*Vt,float*w,float*out,int P,int T,int H,int D){__shared__ float red[256],den;int node=(int)blockIdx.x,h=(int)blockIdx.y,tid=(int)threadIdx.x,total=P+T;size_t base=((size_t)node*H+h)*total;float local=-CUDART_INF_F;for(int s=tid;s<total;s+=(int)blockDim.x)local=fmaxf(local,w[base+s]);red[tid]=local;__syncthreads();for(int off=128;off;off>>=1){if(tid<off)red[tid]=fmaxf(red[tid],red[tid+off]);__syncthreads();}float maximum=red[0],sum=0;__syncthreads();for(int s=tid;s<total;s+=(int)blockDim.x){float x=expf(w[base+s]-maximum);w[base+s]=x;sum+=x;}red[tid]=sum;__syncthreads();for(int off=128;off;off>>=1){if(tid<off)red[tid]+=red[tid+off];__syncthreads();}if(tid==0)den=red[0];__syncthreads();for(int d=tid;d<D;d+=(int)blockDim.x){float z=0;for(int s=0;s<total;++s){const float*v=s<P?Vc+((size_t)s*H+h)*D:Vt+((size_t)(s-P)*H+h)*D;z+=w[base+s]*v[d];}out[((size_t)node*H+h)*D+d]=z/den;}}
extern "C" void solve(const float*Q,const float*Kc,const float*Vc,const float*Kt,const float*Vt,const int*p,float*o,int P,int T,int H,int D){int total=P+T;float*w;cudaMalloc(&w,(size_t)T*H*total*sizeof(float));tree_scores<<<dim3(T,H),256>>>(Q,Kc,Kt,p,w,P,T,H,D);tree_finish<<<dim3(T,H),256>>>(Vc,Vt,w,o,P,T,H,D);cudaDeviceSynchronize();cudaFree(w);}
```
