export const sections = [
  {
    id: "start",
    title: "学习路线",
    pages: [
      { id: "overview", title: "LeetGPU 学习总览" },
      { id: "how-to-use", title: "如何使用这套教程" },
      { id: "environment", title: "平台与 CUDA 环境" },
    ],
  },
  {
    id: "fundamentals",
    title: "CUDA 基础",
    pages: [
      { id: "cuda-model", title: "GPU 与 CUDA 编程模型" },
      { id: "grid-block-thread", title: "Grid、Block 与 Thread" },
      { id: "kernel-launch", title: "Kernel 启动与边界判断" },
      { id: "memory-model", title: "GPU 内存层级" },
      { id: "coalescing", title: "合并访存" },
    ],
  },
  {
    id: "elementwise",
    title: "基础算子",
    pages: [
      { id: "vector-addition", title: "向量加法" },
      { id: "value-clipping", title: "数值裁剪" },
      { id: "matrix-addition", title: "矩阵加法" },
      { id: "interleave-arrays", title: "数组交错" },
      { id: "rgb-grayscale", title: "RGB 转灰度" },
    ],
  },
  {
    id: "memory",
    title: "访存与布局",
    pages: [
      { id: "matrix-transpose", title: "矩阵转置" },
      { id: "shared-memory", title: "Shared Memory" },
      { id: "bank-conflict", title: "Bank Conflict" },
      { id: "vectorized-access", title: "向量化读写" },
    ],
  },
  {
    id: "reduction",
    title: "归约与融合",
    pages: [
      { id: "reduction-sum", title: "并行归约" },
      { id: "warp-shuffle", title: "Warp Shuffle" },
      { id: "softmax", title: "Softmax" },
      { id: "rmsnorm", title: "RMSNorm" },
    ],
  },
  {
    id: "gemm",
    title: "GEMM 优化",
    pages: [
      { id: "gemm-intro", title: "GEMM 问题定义" },
      { id: "naive-gemm", title: "朴素 GEMM" },
      { id: "tiled-gemm", title: "Shared Memory 分块" },
      { id: "register-tiling", title: "寄存器分块" },
      { id: "tensor-core", title: "Tensor Core 入门" },
    ],
  },
  {
    id: "quant",
    title: "量化算子",
    pages: [
      { id: "quantization", title: "INT8 量化基础" },
      { id: "w8a8-gemm", title: "W8A8 GEMM" },
      { id: "quant-fusion", title: "Scale 与后处理融合" },
    ],
  },
  {
    id: "practice",
    title: "刷题记录",
    pages: [
      { id: "checklist", title: "完整题单" },
      { id: "activity", title: "学习热力图" },
      { id: "interview", title: "面试复习清单" },
    ],
  },
];

const code = String.raw;

export const pages = {
  overview: {
    title: "LeetGPU 学习路线",
    description: "从第一个 CUDA 线程开始，系统学习访存、归约、GEMM、Tensor Core 与 W8A8。",
    body: `
<p>这里是整套 LeetGPU 教程的总入口。路线不是按题目难度简单排序，而是按 GPU 编程中能力的依赖关系组织：先理解线程怎样找到数据，再理解数据怎样高效移动，最后进入计算密集型算子的分块和流水线。</p>
<div class="callout"><strong>学习目标</strong><p>完成路线后，你应该能够从空白实现常见 CUDA 算子，解释线程映射与内存访问，并使用基准测试判断一次优化是否真实有效。</p></div>
<h2 id="order">推荐学习顺序</h2>
<ol class="route-list">
<li><a href="#/cuda-model"><strong>CUDA 基础</strong></a>：GPU 执行模型、Grid/Block/Thread、kernel launch 与边界判断。</li>
<li><a href="#/vector-addition"><strong>基础算子</strong></a>：通过向量加法、裁剪和矩阵加法建立 elementwise 模板。</li>
<li><a href="#/matrix-transpose"><strong>访存与布局</strong></a>：学习合并访存、shared memory 和 bank conflict。</li>
<li><a href="#/reduction-sum"><strong>归约与融合</strong></a>：从线程独立计算进入线程协作，完成 Softmax 和 RMSNorm。</li>
<li><a href="#/gemm-intro"><strong>GEMM 优化</strong></a>：从朴素矩阵乘推进到 shared-memory tiling、register tiling 和 Tensor Core。</li>
<li><a href="#/quantization"><strong>量化算子</strong></a>：理解 INT8 表示、INT32 累加、scale 与 W8A8 GEMM。</li>
</ol>
<h2 id="method">每一题怎么学</h2>
<p>每道题固定走四遍：先写 CPU 数学定义；再独立完成正确版本；然后画出线程与地址映射；最后只改变一个因素做性能实验。代码通过测试之后，学习才刚完成一半。</p>
<table><thead><tr><th>阶段</th><th>你要回答的问题</th><th>产物</th></tr></thead><tbody>
<tr><td>定义</td><td>输入、输出和公式是什么？</td><td>CPU 伪代码</td></tr>
<tr><td>正确性</td><td>每个线程负责什么？边界在哪里？</td><td>基础 kernel</td></tr>
<tr><td>解释</td><td>线程访问哪些地址？是否同步？</td><td>索引图与笔记</td></tr>
<tr><td>优化</td><td>瓶颈是计算、带宽还是延迟？</td><td>基准结果与结论</td></tr>
</tbody></table>
<h2 id="start">现在开始</h2>
<p>如果这是你第一次写 CUDA，从 <a href="#/cuda-model">GPU 与 CUDA 编程模型</a> 开始；如果已经理解 kernel launch，可以直接进入 <a href="#/vector-addition">向量加法</a>。</p>`,
  },
  "how-to-use": {
    title: "如何使用这套教程",
    description: "阅读、编码、提交、复盘的完整闭环。",
    body: `
<p>左侧目录就是建议的学习顺序。每篇文章会解释当前题目新增的知识，而不是一次把所有优化技巧堆给你。</p>
<h2 id="before">做题前</h2><ul><li>自己写下输入输出和公式。</li><li>估算需要启动多少个线程。</li><li>明确一个线程处理一个元素、一个输出，还是一小块数据。</li></ul>
<h2 id="during">编码时</h2><p>先保留 LeetGPU 的函数签名和启动框架，只补题目要求的 kernel。发生错误时记录完整编译信息或失败用例，不用“多改几个地方再试”掩盖根因。</p>
<h2 id="after">通过后</h2><ul><li>关掉答案，从空白重写。</li><li>用一个很小的输入手算线程下标。</li><li>记录当前版本的 GPU、shape、数据类型和时间。</li><li>点击页面右上角“标记完成”，加入独立热力图。</li></ul>`,
  },
  environment: {
    title: "平台与 CUDA 环境",
    description: "LeetGPU 在线环境、本地环境和性能测试分别适合什么。",
    body: `
<p>LeetGPU 可以在线编辑、编译和评测 CUDA kernel。入门阶段使用网页即可，本机不需要先配置 CUDA。</p>
<h2 id="online">在线刷题</h2><p>Challenges 用于按题目提交；Playground 用于验证独立代码片段。正确性测试与排行榜解决的问题不同：前者确认结果，后者才讨论在特定 GPU 和 shape 上的性能。</p>
<h2 id="local">什么时候需要本地或云 GPU</h2><p>进入 GEMM、Tensor Core 和 W8A8 后，需要稳定硬件、重复基准和 Nsight Compute 指标。这时再建立本地 CUDA 工程或按小时使用云 GPU。</p>
<div class="warning"><strong>基准规则</strong><p>比较两个版本时保持 GPU、shape、dtype、预热次数和计时方式一致。</p></div>`,
  },
  "cuda-model": {
    title: "GPU 与 CUDA 编程模型",
    description: "理解为什么同一个 kernel 会被大量线程同时执行。",
    body: `
<p>CPU 擅长执行少量复杂线程，GPU 擅长让大量线程执行相同结构的计算。CUDA kernel 是运行在 GPU 上的函数；CPU 端负责配置并启动它。</p>
<h2 id="host-device">Host 与 Device</h2><pre><code>CPU / Host                         GPU / Device
执行 solve()                       执行 kernel()
准备参数和启动配置        →         大量线程并行计算
等待或继续提交任务         ←         把结果写入显存</code></pre>
<p><code>__global__</code> 表示函数由 CPU 发起、在 GPU 上运行。kernel 一般不使用返回值，而是通过输出指针写入结果。</p>
<h2 id="same-code">相同代码，不同数据</h2><p>所有线程执行相同 kernel，但每个线程看到不同的 <code>blockIdx</code> 和 <code>threadIdx</code>。这些内置变量让线程找到自己负责的数据。</p>
${code`<pre><code>__global__ void kernel(float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = ...;
}</code></pre>`}
<p>下一篇：<a href="#/grid-block-thread">Grid、Block 与 Thread</a>。</p>`,
  },
  "grid-block-thread": {
    title: "Grid、Block 与 Thread",
    description: "CUDA 线程组织结构和全局索引推导。",
    body: `
<p>一次 kernel launch 创建一个 Grid。Grid 包含多个 Block，每个 Block 包含多个 Thread。</p>
<div class="diagram"><div>Grid<div class="blocks"><span>Block 0<br><small>Thread 0 … 255</small></span><span>Block 1<br><small>Thread 0 … 255</small></span><span>Block 2<br><small>Thread 0 … 255</small></span></div></div></div>
<h2 id="index">一维全局索引</h2><pre><code>int i = blockIdx.x * blockDim.x + threadIdx.x;</code></pre>
<p>假设每个 block 有 256 个线程，Block 2 内的 Thread 10 对应全局索引 <code>2 × 256 + 10 = 522</code>。</p>
<h2 id="why-block">为什么还要 Block</h2><p>Block 不只是分组。处在同一 Block 的线程可以通过 shared memory 交换数据，也可以使用 <code>__syncthreads()</code> 同步。这会成为转置、归约和 GEMM 的基础。</p>`,
  },
  "kernel-launch": {
    title: "Kernel 启动与边界判断",
    description: "读懂 <<<grid, block>>> 和整数向上取整。",
    body: `
<pre><code>int threads = 256;
int blocks = (N + threads - 1) / threads;
kernel&lt;&lt;&lt;blocks, threads&gt;&gt;&gt;(input, output, N);</code></pre>
<p><code>&lt;&lt;&lt;blocks, threads&gt;&gt;&gt;</code> 配置 block 数量和每个 block 的线程数。公式执行整数向上取整，确保线程总数至少覆盖 N。</p>
<h2 id="boundary">边界为何存在</h2><p>当 N=513 时，需要 3 个 block，共启动 768 个线程。合法下标只有 0 到 512，所以编号 513 到 767 的线程必须停止数组访问。</p>
<pre><code>if (i &lt; N) {
    output[i] = input[i];
}</code></pre>
<div class="warning"><strong>越界不是稳定报错</strong><p>它可能表现为错误结果、破坏相邻显存或 illegal memory access，也可能暂时看起来正常。</p></div>`,
  },
  "memory-model": {
    title: "GPU 内存层级",
    description: "从全局内存、共享内存到寄存器。",
    body: `
<p>算子性能往往受数据移动限制。不同内存层级在容量、作用域、延迟和带宽上差异很大。</p>
<table><thead><tr><th>层级</th><th>作用域</th><th>特点</th></tr></thead><tbody><tr><td>Global Memory</td><td>整个 GPU</td><td>容量大、延迟高，输入输出通常位于这里</td></tr><tr><td>Shared Memory</td><td>一个 Block</td><td>片上、可由线程协作复用</td></tr><tr><td>Register</td><td>一个 Thread</td><td>最快，但数量有限</td></tr></tbody></table>
<p>优化的核心不是“全部搬进最快内存”，而是让每次昂贵的数据读取产生更多有效计算。</p>`,
  },
  coalescing: {
    title: "合并访存",
    description: "为什么相邻线程最好访问相邻地址。",
    body: `
<p>一个 warp 中的线程访问连续、对齐的地址时，硬件可以把请求合并成较少的内存事务。</p>
<pre><code>// 良好的连续访问
int i = blockIdx.x * blockDim.x + threadIdx.x;
float x = input[i];</code></pre>
<p>若线程 0、1、2 分别访问相距很远的位置，会浪费传输带宽。分析访存时应从“一个 warp 的 32 个线程分别访问哪里”出发。</p>`,
  },
};

function problemPage({ title, titleEn, description, formula, url, concepts, prompts, explanation, codeText }) {
  return {
    title,
    description,
    problemId: titleEn.toLowerCase().replace(/[^a-z0-9]+/g, "-"),
    body: `
<div class="problem-meta"><span>LeetGPU</span><span>CUDA</span><span>基础实现</span></div>
<p>${description}</p>
<p><a class="external-link" href="${url}" target="_blank" rel="noreferrer">打开 LeetGPU 题目 ↗</a></p>
<h2 id="definition">题目定义</h2><div class="formula-block">${formula}</div>
<h2 id="thinking">写代码前先回答</h2><ol>${prompts.map((x) => `<li>${x}</li>`).join("")}</ol>
<h2 id="mapping">线程映射</h2><p>${explanation}</p>
<h2 id="template">参考实现</h2><pre><code>${codeText.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")}</code></pre>
<h2 id="review">复盘清单</h2><ul>${concepts.map((x) => `<li>${x}</li>`).join("")}</ul>
<div class="callout"><strong>练习要求</strong><p>先根据提示独立完成；提交通过后，再对照实现并从空白重写一次。</p></div>`,
  };
}

pages["vector-addition"] = problemPage({
  title: "向量加法",
  titleEn: "Vector Addition",
  description: "这是整套路线的第一道题。目标不是加法本身，而是建立 kernel、线程索引、边界判断和显存指针的完整模型。",
  formula: "C[i] = A[i] + B[i]，0 ≤ i < N",
  url: "https://leetgpu.com/challenges/vector-addition",
  concepts: ["能解释 blockIdx.x、blockDim.x 和 threadIdx.x。", "能推导全局索引 i。", "能解释为什么向上取整后必须检查 i < N。", "能说明相邻线程访问相邻元素。"],
  prompts: ["向量共有多少个元素？", "如何给每个线程分配唯一的 i？", "当 N 不是 256 的倍数时会发生什么？"],
  explanation: "采用一维 Grid。每个线程处理一个元素；总线程数向上取整覆盖 N，最后一个 block 中多余线程通过边界条件退出。",
  codeText: `#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}`,
});

pages["value-clipping"] = problemPage({
  title: "数值裁剪",
  titleEn: "Value Clipping",
  description: "复用向量加法的线程框架，把单元素数学规则翻译成 elementwise kernel。",
  formula: "output[i] = min(max(input[i], lo), hi)",
  url: "https://leetgpu.com/challenges/value-clipping",
  concepts: ["线程映射与 Vector Addition 相同。", "能覆盖小于 lo、大于 hi 和区间内三种情况。", "知道分支表达和 min/max 表达的对应关系。"],
  prompts: ["输入与输出长度是否相同？", "每个线程需要读取和写入几个元素？", "三个取值区间分别输出什么？"],
  explanation: "仍然让线程 i 处理 input[i]。新的知识只在单元素计算规则，不需要改变 Grid 配置。",
  codeText: `__global__ void clip_kernel(const float* input, float* output,
                            float lo, float hi, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = input[i];
        output[i] = x < lo ? lo : (x > hi ? hi : x);
    }
}`,
});

pages["matrix-addition"] = problemPage({
  title: "矩阵加法",
  titleEn: "Matrix Addition",
  description: "理解二维矩阵在显存中仍是一段连续数组。逐元素算子不一定需要显式计算行列。",
  formula: "C[r,c] = A[r,c] + B[r,c]",
  url: "https://leetgpu.com/challenges/matrix-addition",
  concepts: ["N×N 矩阵的元素总数是 N²。", "合法线性下标范围是 [0, N²)。", "逐元素操作可以直接线性遍历。"],
  prompts: ["矩阵在内存中怎样展平？", "总线程数应覆盖 N 还是 N×N？", "什么时候才必须计算 row 和 col？"],
  explanation: "把矩阵视为长度 N×N 的连续数组。线程 i 直接处理 A[i]、B[i] 和 C[i]。",
  codeText: `__global__ void matrix_add(const float* A, const float* B,
                           float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (i < total) C[i] = A[i] + B[i];
}`,
});

const generic = {
  "interleave-arrays": ["数组交错", "由输入索引推导多个输出地址。"],
  "rgb-grayscale": ["RGB 转灰度", "练习多通道图像的数据布局和地址计算。"],
  "matrix-transpose": ["矩阵转置", "从朴素实现进入二维 Grid、共享内存与合并访存。"],
  "shared-memory": ["Shared Memory", "理解 Block 内线程怎样协作搬运和复用数据。"],
  "bank-conflict": ["Bank Conflict", "理解共享内存地址布局为何影响吞吐。"],
  "vectorized-access": ["向量化读写", "使用更宽的内存操作减少指令数量。"],
  "reduction-sum": ["并行归约", "用树形结构合并线程局部结果。"],
  "warp-shuffle": ["Warp Shuffle", "在 warp 内通过寄存器交换数据。"],
  softmax: ["Softmax", "组合 Max、Exp、Sum 与 Normalize，并保持数值稳定。"],
  rmsnorm: ["RMSNorm", "通过平方和归约与融合归一化理解 LLM 常见算子。"],
  "gemm-intro": ["GEMM 问题定义", "明确 M、N、K、矩阵布局和性能指标。"],
  "naive-gemm": ["朴素 GEMM", "一个线程计算一个输出元素，建立正确性基线。"],
  "tiled-gemm": ["Shared Memory 分块", "通过 tile 提升输入数据复用率。"],
  "register-tiling": ["寄存器分块", "让每个线程计算多个输出，提升算术强度。"],
  "tensor-core": ["Tensor Core 入门", "理解 WMMA/MMA 的 tile、数据类型与布局约束。"],
  quantization: ["INT8 量化基础", "理解量化值、scale、zero point 和误差来源。"],
  "w8a8-gemm": ["W8A8 GEMM", "实现 INT8×INT8→INT32，再融合 scale 与输出转换。"],
  "quant-fusion": ["Scale 与后处理融合", "减少中间张量和全局内存往返。"],
  checklist: ["完整题单", "按阶段查看并更新所有题目的学习状态。"],
  activity: ["学习热力图", "用独立绿点记录每天完成的 LeetGPU 题目。"],
  interview: ["面试复习清单", "按索引、访存、归约、GEMM 和量化整理口述问题。"],
};

Object.entries(generic).forEach(([id, [title, description]]) => {
  if (pages[id]) return;
  pages[id] = {
    title,
    description,
    body: `<p>${description}</p><div class="callout"><strong>章节建设中</strong><p>这部分已经纳入完整路线。当前先按左侧顺序完成前置章节，后续内容会随着刷题进度补充实现、图解与性能实验。</p></div><h2 id="goals">学习目标</h2><ul><li>写出正确的数学定义和 CPU 参考实现。</li><li>画出线程到数据的映射。</li><li>解释全局内存、共享内存与寄存器之间的数据流。</li><li>用固定测试条件验证优化效果。</li></ul>`,
  };
});

export function flatPages() {
  return sections.flatMap((section) => section.pages.map((page) => ({ ...page, section: section.title })));
}
