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

export const pages = {
  overview: {
    title: "LeetGPU 学习路线",
    description: "从第一个 CUDA 线程开始，系统学习访存、归约、GEMM、Tensor Core 与 W8A8。",
    body: `
<p>这里是整套 LeetGPU 教程的总入口。路线不是按题目难度简单排序，而是按 GPU 编程中能力的依赖关系组织：先理解线程怎样找到数据，再理解数据怎样高效移动，最后进入计算密集型算子的分块和流水线。</p>
<div class="callout"><strong>学习目标</strong><p>完成路线后，你应该能够从空白实现常见 CUDA 算子，解释线程映射与内存访问，并使用基准测试判断一次优化是否真实有效。</p></div>
<h2 id="order">推荐学习顺序</h2>
<ol class="route-list">
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
<p>从 <a href="#/vector-addition">向量加法</a> 开始，在具体题目中逐步理解线程索引、边界判断和 kernel launch。</p>`,
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
};

function problemPage({ title, titleEn, description, formula, originalSpec, url, concepts, prompts, explanation, codeText }) {
  const thinkingSection = prompts.length
    ? `<h2 id="thinking">写代码前先回答</h2><ol>${prompts.map((x) => `<li>${x}</li>`).join("")}</ol>`
    : "";
  const mappingSection = explanation
    ? `<h2 id="mapping">线程映射</h2><p>${explanation}</p>`
    : "";

  return {
    title,
    description,
    problemId: titleEn.toLowerCase().replace(/[^a-z0-9]+/g, "-"),
    body: `
<p><a class="external-link" href="${url}" target="_blank" rel="noreferrer">打开 LeetGPU 题目 ↗</a></p>
<h2 id="definition">题目定义</h2><p>${description}</p>${originalSpec || `<div class="formula-block">${formula}</div>`}
${thinkingSection}
${mappingSection}
<h2 id="template">参考实现</h2><pre><code>${codeText.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")}</code></pre>
<h2 id="review">复盘清单</h2><ul>${concepts.map((x) => `<li>${x}</li>`).join("")}</ul>
<div class="callout"><strong>练习要求</strong><p>先根据提示独立完成；提交通过后，再对照实现并从空白重写一次。</p></div>`,
  };
}

pages["vector-addition"] = problemPage({
  title: "向量加法",
  titleEn: "Vector Addition",
  description: "编写一个 GPU 程序，对两个包含 32 位浮点数的向量执行逐元素加法。程序接收两个长度相同的输入向量，并生成一个包含它们之和的输出向量。",
  formula: "C[i] = A[i] + B[i]，0 ≤ i < N",
  originalSpec: `
<h3>实现要求</h3>
<ul>
  <li>不允许使用外部库。</li>
  <li><code>solve</code> 函数签名必须保持不变。</li>
  <li>最终结果必须存储在向量 <code>C</code> 中。</li>
</ul>
<h3>示例 1</h3>
<pre><code>输入：A = [1.0, 2.0, 3.0, 4.0]
      B = [5.0, 6.0, 7.0, 8.0]
输出：C = [6.0, 8.0, 10.0, 12.0]</code></pre>
<h3>示例 2</h3>
<pre><code>输入：A = [1.5, 1.5, 1.5]
      B = [2.3, 2.3, 2.3]
输出：C = [3.8, 3.8, 3.8]</code></pre>
<h3>约束</h3>
<ul>
  <li>输入向量 <code>A</code> 和 <code>B</code> 的长度相同。</li>
  <li><code>1 ≤ N ≤ 100,000,000</code>。</li>
  <li>性能测试使用 <code>N = 25,000,000</code>。</li>
</ul>`,
  url: "https://leetgpu.com/challenges/vector-addition",
  concepts: ["能解释 blockIdx.x、blockDim.x 和 threadIdx.x。", "能推导全局索引 i。", "能解释为什么向上取整后必须检查 i < N。", "能说明相邻线程访问相邻元素。"],
  prompts: [],
  explanation: "",
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
