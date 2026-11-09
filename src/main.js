import "./styles.css";
import { problems, stages, starterProgress } from "./data/problems.js";
import { principles, weeklyPlan } from "./data/guide.js";
import {
  activityCounts,
  exportProgress,
  importProgress,
  loadProgress,
  saveProgress,
} from "./utils/progress.js";
import { renderHeatmap } from "./utils/heatmap.js";

const app = document.querySelector("#app");
let progress = loadProgress(starterProgress);
let selectedProblem = problems[0].id;
let searchText = "";
let statusFilter = "all";
let stageFilter = "all";

const icon = {
  arrow: `<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10h11M11 5l5 5-5 5"/></svg>`,
  check: `<svg viewBox="0 0 20 20" aria-hidden="true"><path d="m4 10 4 4 8-9"/></svg>`,
  github: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2a10 10 0 0 0-3.16 19.49c.5.1.68-.22.68-.48v-1.7c-2.78.6-3.37-1.18-3.37-1.18-.46-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.61.07-.61 1 .07 1.53 1.03 1.53 1.03.9 1.53 2.34 1.09 2.91.83.09-.65.35-1.09.64-1.34-2.22-.25-4.55-1.11-4.55-4.94 0-1.09.39-1.98 1.03-2.68-.1-.25-.45-1.27.1-2.64 0 0 .84-.27 2.75 1.02A9.6 9.6 0 0 1 12 6.99a9.6 9.6 0 0 1 2.5.34c1.91-1.29 2.75-1.02 2.75-1.02.55 1.37.2 2.39.1 2.64.64.7 1.03 1.59 1.03 2.68 0 3.84-2.34 4.68-4.57 4.93.36.31.68.92.68 1.86V21c0 .27.18.59.69.48A10 10 0 0 0 12 2Z"/></svg>`,
};

function statusOf(id) {
  return progress[id]?.status || "todo";
}

function stats() {
  const done = problems.filter((item) => statusOf(item.id) === "done").length;
  const doing = problems.filter((item) => statusOf(item.id) === "doing").length;
  return { done, doing, total: problems.length, percent: Math.round((done / problems.length) * 100) };
}

function shell() {
  const currentStats = stats();
  app.innerHTML = `
    <header class="site-header">
      <a class="brand" href="#top" aria-label="Kernel Path 首页">
        <span class="brand-mark"><span></span><span></span><span></span></span>
        <span><strong>Kernel Path</strong><small>LEETGPU ROADMAP</small></span>
      </a>
      <nav class="main-nav" aria-label="主导航">
        <a href="#roadmap">路线</a><a href="#problems">题单</a><a href="#journal">进度</a><a href="#method">方法</a>
      </nav>
      <div class="header-actions">
        <button class="text-button" id="export-progress">导出进度</button>
        <label class="text-button file-button">导入进度<input id="import-progress" type="file" accept="application/json" /></label>
        <a class="github-button" href="https://github.com/" target="_blank" rel="noreferrer">${icon.github}<span>GitHub</span></a>
      </div>
    </header>

    <main id="top">
      <section class="hero section-shell">
        <div class="hero-copy">
          <div class="eyebrow"><span></span>从第一个线程，到 W8A8 GEMM</div>
          <h1>把 GPU 算子<br><em>真正写进脑子里</em></h1>
          <p>一条为面试与实战设计的 LeetGPU 学习路线。每道题都有目标、思考提示和复盘位置；不收集答案，只记录你如何得到答案。</p>
          <div class="hero-actions">
            <a class="primary-button" href="#problems">继续刷题 ${icon.arrow}</a>
            <a class="secondary-button" href="#roadmap">查看完整路线</a>
          </div>
          <div class="hero-meta"><span>${currentStats.done}/${currentStats.total} 已完成</span><i></i><span>当前阶段：线程与索引</span></div>
        </div>
        <div class="hero-panel">
          <div class="terminal-head"><span></span><span></span><span></span><small>vector_add.cu</small></div>
          <pre><code><b>__global__ void</b> vector_add(
  <i>const float</i>* A,
  <i>const float</i>* B,
  <i>float</i>* C,
  <i>int</i> N
) {
  <i>int</i> idx = blockIdx.x *
            blockDim.x + threadIdx.x;

  <b>if</b> (idx &lt; N) {
    C[idx] = A[idx] + B[idx];
  }
}</code></pre>
          <div class="terminal-foot"><span class="success-dot"></span><strong>Accepted</strong><span>CUDA · Tesla T4</span><time>0.021 ms</time></div>
        </div>
      </section>

      <section class="roadmap-section" id="roadmap">
        <div class="section-shell">
          <div class="section-heading"><div><span class="kicker">LEARNING PATH</span><h2>五个阶段，逐层增加难度</h2></div><p>每个阶段只引入一组新问题。前面的能力，会成为后面算子的积木。</p></div>
          <div class="stage-grid">${stages.map(stageCard).join("")}</div>
        </div>
      </section>

      <section class="problem-section section-shell" id="problems">
        <div class="section-heading compact"><div><span class="kicker">PROBLEM SET</span><h2>题单与学习笔记</h2></div><p>标记状态、打开题目、记录自己的推导。进度只保存在你的浏览器中，可导出 JSON 备份。</p></div>
        <div class="problem-toolbar">
          <div class="search-box"><span>⌕</span><input id="problem-search" type="search" placeholder="搜索题目或知识点" /></div>
          <div class="filter-group" id="status-filters">
            <button class="active" data-status="all">全部</button><button data-status="todo">未开始</button><button data-status="doing">进行中</button><button data-status="done">已完成</button>
          </div>
          <select id="stage-filter" aria-label="按阶段筛选"><option value="all">全部阶段</option>${stages.map((s) => `<option value="${s.id}">${s.order} · ${s.title}</option>`).join("")}</select>
        </div>
        <div class="problem-layout">
          <div class="problem-list" id="problem-list"></div>
          <aside class="problem-detail" id="problem-detail"></aside>
        </div>
      </section>

      <section class="journal-section" id="journal">
        <div class="section-shell journal-grid">
          <div>
            <span class="kicker">YOUR JOURNAL</span><h2>单独属于 LeetGPU 的绿点</h2>
            <p class="journal-intro">完成一道题，就留下一个格子。它和 GitHub Contribution Calendar 分开，只描述你的 GPU 学习轨迹。</p>
            <div class="activity-card">
              <div class="activity-head"><div><strong>过去一年</strong><span id="activity-total"></span></div><div class="activity-badge">LEETGPU ACTIVITY</div></div>
              <div id="heatmap"></div>
            </div>
          </div>
          <div class="progress-card">
            <span class="progress-label">TOTAL PROGRESS</span>
            <div class="progress-number"><strong>${currentStats.percent}</strong><span>%</span></div>
            <div class="progress-track"><i style="width:${currentStats.percent}%"></i></div>
            <dl><div><dt>已完成</dt><dd>${currentStats.done}</dd></div><div><dt>进行中</dt><dd>${currentStats.doing}</dd></div><div><dt>全部题目</dt><dd>${currentStats.total}</dd></div></dl>
            <p>数据保存在 localStorage。换设备前记得导出，或把进度 JSON 提交到自己的仓库。</p>
          </div>
        </div>
      </section>

      <section class="method-section section-shell" id="method">
        <div class="section-heading compact"><div><span class="kicker">HOW TO PRACTICE</span><h2>不是“看懂”，是能从空白复现</h2></div></div>
        <div class="principle-grid">${principles.map((item, index) => `<article><span>0${index + 1}</span><h3>${item.title}</h3><p>${item.text}</p></article>`).join("")}</div>
        <div class="weekly-card"><div><span class="kicker">A REPEATABLE WEEK</span><h3>一题六步循环</h3><p>难题可以延长周期，但不要跳过复盘。</p></div><ol>${weeklyPlan.map((item) => `<li><strong>${item.day}</strong><span>${item.task}</span><small>${item.output}</small></li>`).join("")}</ol></div>
      </section>
    </main>

    <footer><div class="section-shell"><div class="brand"><span class="brand-mark"><span></span><span></span><span></span></span><span><strong>Kernel Path</strong><small>BUILD · MEASURE · UNDERSTAND</small></span></div><p>为 LeetGPU 学习者制作的静态题单与进度站。</p><a href="#top">回到顶部 ↑</a></div></footer>
    <div class="toast" id="toast" role="status"></div>
  `;
}

function stageCard(stage) {
  const stageProblems = problems.filter((item) => item.stage === stage.id);
  const done = stageProblems.filter((item) => statusOf(item.id) === "done").length;
  const current = done > 0 && done < stageProblems.length;
  return `<article class="stage-card ${current ? "current" : ""}">
    <div class="stage-number">${stage.order}</div>
    <span class="stage-status">${done === stageProblems.length ? "已完成" : current ? "正在学习" : `${stageProblems.length} 题`}</span>
    <h3>${stage.title}</h3><h4>${stage.subtitle}</h4><p>${stage.description}</p>
    <div class="stage-progress"><i style="width:${stageProblems.length ? (done / stageProblems.length) * 100 : 0}%"></i></div>
    <span>${done} / ${stageProblems.length}</span>
  </article>`;
}

function filteredProblems() {
  const query = searchText.trim().toLowerCase();
  return problems.filter((problem) => {
    const statusMatch = statusFilter === "all" || statusOf(problem.id) === statusFilter;
    const stageMatch = stageFilter === "all" || problem.stage === stageFilter;
    const haystack = `${problem.title} ${problem.titleZh} ${problem.concepts.join(" ")}`.toLowerCase();
    return statusMatch && stageMatch && (!query || haystack.includes(query));
  });
}

function renderProblems() {
  const list = document.querySelector("#problem-list");
  const filtered = filteredProblems();
  if (!filtered.some((item) => item.id === selectedProblem)) selectedProblem = filtered[0]?.id || null;
  list.innerHTML = filtered.length ? filtered.map(problemRow).join("") : `<div class="empty-state">没有符合条件的题目。</div>`;
  renderDetail();

  list.querySelectorAll(".problem-row").forEach((button) => {
    button.addEventListener("click", () => {
      selectedProblem = button.dataset.id;
      renderProblems();
    });
  });
}

function problemRow(problem) {
  const status = statusOf(problem.id);
  const statusLabel = { todo: "未开始", doing: "进行中", done: "已完成" }[status];
  return `<button class="problem-row ${selectedProblem === problem.id ? "selected" : ""}" data-id="${problem.id}">
    <span class="problem-index">${String(problem.no).padStart(2, "0")}</span>
    <span class="problem-main"><strong>${problem.titleZh}</strong><small>${problem.title}</small><em>${problem.concepts.slice(0, 2).join(" · ")}</em></span>
    <span class="difficulty ${problem.difficulty.toLowerCase()}">${problem.difficulty}</span>
    <span class="problem-status ${status}">${status === "done" ? icon.check : ""}${statusLabel}</span>
  </button>`;
}

function renderDetail() {
  const detail = document.querySelector("#problem-detail");
  const problem = problems.find((item) => item.id === selectedProblem);
  if (!problem) {
    detail.innerHTML = `<div class="empty-state">选择一道题查看详情。</div>`;
    return;
  }
  const entry = progress[problem.id] || {};
  detail.innerHTML = `
    <div class="detail-top"><span>${String(problem.no).padStart(2, "0")} / ${problem.language}</span><a href="${problem.url}" target="_blank" rel="noreferrer">打开 LeetGPU ${icon.arrow}</a></div>
    <h3>${problem.titleZh}</h3><h4>${problem.title}</h4><p class="detail-summary">${problem.summary}</p>
    <div class="formula">${problem.formula}</div>
    <div class="detail-block"><strong>先回答这几个问题</strong><ol>${problem.prompts.map((item) => `<li>${item}</li>`).join("")}</ol></div>
    <div class="detail-block"><strong>复盘要点</strong><ul>${problem.notes.map((item) => `<li>${item}</li>`).join("")}</ul></div>
    <div class="concept-list">${problem.concepts.map((item) => `<span>${item}</span>`).join("")}</div>
    <label class="note-label">我的记录<textarea id="problem-note" placeholder="写下报错、索引推导或优化结论……">${entry.note || ""}</textarea></label>
    <div class="status-actions">
      <button data-set-status="todo" class="${statusOf(problem.id) === "todo" ? "active" : ""}">未开始</button>
      <button data-set-status="doing" class="${statusOf(problem.id) === "doing" ? "active" : ""}">进行中</button>
      <button data-set-status="done" class="${statusOf(problem.id) === "done" ? "active" : ""}">完成题目</button>
    </div>
  `;

  detail.querySelectorAll("[data-set-status]").forEach((button) => {
    button.addEventListener("click", () => updateStatus(problem.id, button.dataset.setStatus));
  });
  detail.querySelector("#problem-note").addEventListener("change", (event) => {
    progress[problem.id] = { ...progress[problem.id], status: statusOf(problem.id), note: event.target.value };
    saveProgress(progress);
    showToast("笔记已保存在当前浏览器");
  });
}

function updateStatus(id, status) {
  const previous = progress[id] || {};
  progress[id] = {
    ...previous,
    status,
    completedAt: status === "done" ? previous.completedAt || new Date().toISOString().slice(0, 10) : undefined,
    attempts: status === "done" ? (previous.attempts || 0) + (previous.status === "done" ? 0 : 1) : previous.attempts,
  };
  saveProgress(progress);
  render();
  showToast(status === "done" ? "完成记录已加入 LeetGPU 绿点" : "题目状态已更新");
}

function bindEvents() {
  document.querySelector("#problem-search").value = searchText;
  document.querySelector("#stage-filter").value = stageFilter;
  document.querySelector("#problem-search").addEventListener("input", (event) => {
    searchText = event.target.value;
    renderProblems();
  });
  document.querySelector("#stage-filter").addEventListener("change", (event) => {
    stageFilter = event.target.value;
    renderProblems();
  });
  document.querySelectorAll("#status-filters button").forEach((button) => {
    button.classList.toggle("active", button.dataset.status === statusFilter);
    button.addEventListener("click", () => {
      statusFilter = button.dataset.status;
      document.querySelectorAll("#status-filters button").forEach((item) => item.classList.toggle("active", item === button));
      renderProblems();
    });
  });
  document.querySelector("#export-progress").addEventListener("click", () => exportProgress(progress));
  document.querySelector("#import-progress").addEventListener("change", async (event) => {
    const [file] = event.target.files;
    if (!file) return;
    try {
      progress = await importProgress(file);
      saveProgress(progress);
      render();
      showToast("进度导入成功");
    } catch (error) {
      showToast(`导入失败：${error.message}`);
    }
  });
}

function showToast(message) {
  const toast = document.querySelector("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("show"), 2200);
}

function render() {
  shell();
  renderProblems();
  const counts = activityCounts(progress);
  renderHeatmap(document.querySelector("#heatmap"), counts);
  const completed = Object.values(counts).reduce((sum, value) => sum + value, 0);
  document.querySelector("#activity-total").textContent = `${completed} 道题完成`;
  bindEvents();
}

render();
