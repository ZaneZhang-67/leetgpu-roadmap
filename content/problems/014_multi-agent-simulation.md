---
id: multi-agent-simulation
leetgpuId: 14
titleEn: Multi-Agent Simulation
titleZh: 多智能体模拟
description: 使用 CUDA 并行执行一次 Boids 状态更新，严格处理邻居半径、自身排除、无邻居回退以及速度和位置的更新顺序
problemUrl: https://leetgpu.com/challenges/multi-agent-simulation
updatedAt: 2026-11-12
---
## 题目定义

给定 `N` 个智能体，每个智能体使用连续四个 32 位浮点数表示：

```text
[x, y, vx, vy]
```

其中 `(x, y)` 是位置，`(vx, vy)` 是速度第 `i` 个智能体位于 `agents[4*i ... 4*i+3]`

一次状态更新按以下规则执行：

1. 邻居是除自身外、欧氏距离严格小于 `r = 5.0` 的智能体
2. 若存在邻居，计算邻居速度的平均值；否则平均速度取自身当前速度
3. 使用 `alpha = 0.05` 更新速度
4. 使用新速度更新位置

```text
v_new = v + alpha × (v_avg - v)
p_new = p + v_new
```

所有智能体都从同一份旧状态 `agents` 读取，并将新状态写入 `agents_next`

### 实现要求

- 只使用原生功能，不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写入 `agents_next`

### 示例

```text
N = 2
agents = [
  0.0, 0.0, 1.0,  0.0,
  3.0, 4.0, 0.0, -1.0
]
```

两者距离恰好为 `5.0`，而邻居条件是严格小于 `5.0`，因此它们互不影响：

```text
agents_next = [
  1.0,  0.0, 1.0,  0.0,
  3.0,  3.0, 0.0, -1.0
]
```

### 约束

- 每个智能体占用四个浮点数，输入和输出长度均为 `4 × N`
- 固定邻居半径 `r = 5.0`
- 固定速度融合系数 `alpha = 0.05`
- 功能测试包含 `N = 1`、邻居交互、边界距离和 `N = 1024`
- 性能测试使用 `N = 10,000`

## 模板代码解析

平台提供固定的 CUDA 入口：

```cpp
#include <cuda_runtime.h>

// agents, agents_next are device pointers
extern "C" void solve(
    const float* agents,
    float* agents_next,
    int N
) {}
```

- `agents` 是 GPU 上的旧状态，只读
- `agents_next` 是 GPU 上的新状态，长度同样为 `4 × N`
- `N` 是智能体数量

模板没有提供 kernel 或启动配置

## 实现思路

一个线程负责一个智能体线程读取自己的位置和速度，再遍历全部智能体寻找邻居

### 1. 使用平方距离判断邻居

无需计算平方根，只需比较：

```text
dx² + dy² < 25
```

判断中必须同时满足 `j != i`测试参考实现会显式排除对角线，因此智能体自身不计入邻居平均值，距离恰好等于 5 的智能体也不计入实现分别对两个平方做 FP32 舍入后再相加，避免编译器融合乘加改变严格边界判断

### 2. 计算邻居平均速度

线程累加所有邻居的 `vx`、`vy` 并记录数量若数量为零，平均速度使用自身速度，这会使新速度保持不变

### 3. 保持状态更新顺序

速度先更新：

```cpp
newVx = vx + 0.05f * (avgVx - vx);
newVy = vy + 0.05f * (avgVy - vy);
```

位置再使用新速度更新：

```cpp
newX = x + newVx;
newY = y + newVy;
```

若错误地使用旧速度更新位置，会与 reference 语义不符

### 4. 分离输入输出

所有线程只读取 `agents`，只写自己的 `agents_next[4*i ... 4*i+3]`，因此计算顺序不会影响邻居状态，也不需要原子操作

该实现时间复杂度为 `O(N²)`，额外空间为 `O(1)`，覆盖 reference 的完整状态更新规则

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void simulate_kernel(
    const float* agents,
    float* agents_next,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x = agents[4 * i];
    float y = agents[4 * i + 1];
    float vx = agents[4 * i + 2];
    float vy = agents[4 * i + 3];

    float sumVx = 0.0f;
    float sumVy = 0.0f;
    int count = 0;

    for (int j = 0; j < N; ++j) {
        if (j == i) continue;

        float dx = x - agents[4 * j];
        float dy = y - agents[4 * j + 1];
        float dxSquared = __fmul_rn(dx, dx);
        float dySquared = __fmul_rn(dy, dy);
        float distanceSquared = __fadd_rn(dxSquared, dySquared);

        if (distanceSquared < 25.0f) {
            sumVx += agents[4 * j + 2];
            sumVy += agents[4 * j + 3];
            ++count;
        }
    }

    float avgVx = vx;
    float avgVy = vy;
    if (count > 0) {
        float inverseCount = 1.0f / static_cast<float>(count);
        avgVx = sumVx * inverseCount;
        avgVy = sumVy * inverseCount;
    }

    constexpr float alpha = 0.05f;
    float newVx = vx + alpha * (avgVx - vx);
    float newVy = vy + alpha * (avgVy - vy);

    agents_next[4 * i] = x + newVx;
    agents_next[4 * i + 1] = y + newVy;
    agents_next[4 * i + 2] = newVx;
    agents_next[4 * i + 3] = newVy;
}

// agents, agents_next are device pointers
extern "C" void solve(
    const float* agents,
    float* agents_next,
    int N
) {
    constexpr int threadsPerBlock = 256;
    int blocksPerGrid =
        (N + threadsPerBlock - 1) / threadsPerBlock;

    simulate_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        agents, agents_next, N
    );
    cudaDeviceSynchronize();
}
```
