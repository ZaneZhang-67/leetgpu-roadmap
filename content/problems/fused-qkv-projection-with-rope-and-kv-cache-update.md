---
id: fused-qkv-projection-with-rope-and-kv-cache-update
leetgpuId: 115
titleEn: Fused QKV Projection with RoPE and KV Cache Update
titleZh: 融合 QKV 投影、RoPE 与 KV Cache 更新
description: 使用 CUDA 完成 packed QKV 投影、半区式 RoPE，并更新指定解码位置的 KV Cache
problemUrl: https://leetgpu.com/challenges/fused-qkv-projection-with-rope-and-kv-cache-update
updatedAt: 2026-11-12
---
## 题目定义

`x:[B,d_model]`，`W_qkv:[d_model,(H_q+2H_kv)D]`。投影输出按连续列打包为 Q、K、V：Q 占前 `H_q D`，K 接着占 `H_kv D`，V 占最后 `H_kv D`。

RoPE cache `[S_max,D]` 的前 `D/2` 是 cos，后 `D/2` 是 sin。向量前后半区配对：

```text
out[first] = first*cos-second*sin
out[second] = second*cos+first*sin
```

Q 写入 `[B,H_q,D]`；旋转后的 K 和未旋转的 V 写入 cache 的 `[b,h,positions[b],:]`，其他 cache 位置保持原值。

### 实现要求与约束

保持 starter 签名，并满足：

- `1≤B≤1024`，`1≤d_model≤8192`
- `1≤H_kv≤H_q≤128`，且 `H_q` 是 `H_kv` 的倍数
- `D` 为偶数，`2≤D≤128`
- `1≤S_max≤4096`，且 `0≤positions[b]<S_max`
- `positions` 为 int32，其余张量为 FP32
- x、W_qkv、K_cache、V_cache 的值位于 `[-10,10]`
- 功能测试含 GQA、MHA、零输入和非 2 的幂维度
- 性能测试 `B=64,d_model=4096,H_q=32,H_kv=8,D=128,S_max=2048`
- 绝对误差和相对误差均为 `1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>
extern "C" void solve(
 const float* x,const float* W_qkv,const float* cos_sin_cache,const int* positions,
 float* K_cache,float* V_cache,float* Q_out,int B,int d_model,int H_q,int H_kv,int D,int S_max
) {}
```

## 实现思路

先用 `16×16` Shared Memory tiled GEMM 计算 packed qkv，Shared Memory 输入保持 float，跨 K tile 的 sum 使用 double 累加并在写入 qkv 时转回 float，避免每个输出独立反复读取 x/W。第二个 kernel 以线程映射 Q/KV 元素：Q 和 K 按半区配对读取临时投影及 cos/sin，V 直接写入指定 cache slot。每个输出位置只有一个 writer，任意维度边缘加载置零。

## 参考实现

```cpp
#include <cuda_runtime.h>

constexpr int TILE=16;
__global__ void project(const float* x,const float* W,float* qkv,int B,int K,int C){
 __shared__ float a[TILE][TILE],w[TILE][TILE];int tx=threadIdx.x,ty=threadIdx.y,r=blockIdx.y*TILE+ty,c=blockIdx.x*TILE+tx;double sum=0;
 for(int base=0;base<K;base+=TILE){int ak=base+tx,wk=base+ty;a[ty][tx]=(r<B&&ak<K)?x[r*K+ak]:0;w[ty][tx]=(wk<K&&c<C)?W[wk*C+c]:0;__syncthreads();for(int k=0;k<TILE;++k)sum+=a[ty][k]*w[k][tx];__syncthreads();}if(r<B&&c<C)qkv[r*C+c]=static_cast<float>(sum);
}
__global__ void write_qkv(const float* qkv,const float* cs,const int* pos,float* Kc,float* Vc,float* Qo,int B,int Hq,int Hkv,int D,int S){
 int i=blockIdx.x*blockDim.x+threadIdx.x,total=B*(Hq+2*Hkv)*D;if(i>=total)return;int C=(Hq+2*Hkv)*D,b=i/C,col=i%C,half=D/2,p=pos[b];
 if(col<Hq*D){int h=col/D,k=col%D,pair=k<half?k+half:k-half;float first=qkv[b*C+h*D+(k<half?k:pair)],second=qkv[b*C+h*D+(k<half?pair:k)];float co=cs[p*D+(k<half?k:pair)],si=cs[p*D+half+(k<half?k:pair)];Qo[(b*Hq+h)*D+k]=k<half?first*co-second*si:second*co+first*si;}
 else if(col<(Hq+Hkv)*D){int q=col-Hq*D,h=q/D,k=q%D,pair=k<half?k+half:k-half,base=Hq*D+h*D;float first=qkv[b*C+base+(k<half?k:pair)],second=qkv[b*C+base+(k<half?pair:k)];float co=cs[p*D+(k<half?k:pair)],si=cs[p*D+half+(k<half?k:pair)];Kc[(size_t(b*Hkv+h)*S+p)*D+k]=k<half?first*co-second*si:second*co+first*si;}
 else{int q=col-(Hq+Hkv)*D,h=q/D,k=q%D;Vc[(size_t(b*Hkv+h)*S+p)*D+k]=qkv[b*C+col];}
}
extern "C" void solve(const float* x,const float* W_qkv,const float* cos_sin_cache,const int* positions,float* K_cache,float* V_cache,float* Q_out,int B,int d_model,int H_q,int H_kv,int D,int S_max){
 int C=(H_q+2*H_kv)*D;float* qkv;cudaMalloc(&qkv,size_t(B)*C*4);dim3 t(TILE,TILE),g((C+15)/16,(B+15)/16);project<<<g,t>>>(x,W_qkv,qkv,B,d_model,C);int n=B*C;write_qkv<<<(n+255)/256,256>>>(qkv,cos_sin_cache,positions,K_cache,V_cache,Q_out,B,H_q,H_kv,D,S_max);cudaDeviceSynchronize();cudaFree(qkv);
}
```
