import "./styles.css";
import { sections, pages, flatPages } from "./data/docs.js";
import { problems, starterProgress } from "./data/problems.js";
import { activityCounts, exportProgress, importProgress, loadProgress, saveProgress } from "./utils/progress.js";
import { renderHeatmap } from "./utils/heatmap.js";

const app = document.querySelector("#app");
const allPages = flatPages();
let progress = loadProgress(starterProgress);
let sidebarOpen = false;
let searchOpen = false;

const icons = {
  menu: `<svg viewBox="0 0 24 24"><path d="M4 7h16M4 12h16M4 17h16"/></svg>`,
  search: `<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></svg>`,
  github: `<svg viewBox="0 0 24 24"><path d="M12 2a10 10 0 0 0-3.2 19.5c.5.1.7-.2.7-.5v-1.7c-2.8.6-3.4-1.2-3.4-1.2-.5-1.2-1.1-1.5-1.1-1.5-.9-.6.1-.6.1-.6 1 .1 1.5 1 1.5 1 .9 1.5 2.4 1.1 2.9.8.1-.6.4-1.1.7-1.3-2.2-.3-4.6-1.1-4.6-5 0-1.1.4-2 1-2.7-.1-.2-.4-1.3.1-2.6 0 0 .8-.3 2.8 1a9.6 9.6 0 0 1 5 0c1.9-1.3 2.8-1 2.8-1 .5 1.3.2 2.4.1 2.6.6.7 1 1.6 1 2.7 0 3.9-2.4 4.7-4.6 5 .4.3.7.9.7 1.9V21c0 .3.2.6.7.5A10 10 0 0 0 12 2Z"/></svg>`,
  moon: `<svg viewBox="0 0 24 24"><path d="M20 15.7A8.5 8.5 0 0 1 8.3 4 8.5 8.5 0 1 0 20 15.7Z"/></svg>`,
  check: `<svg viewBox="0 0 20 20"><path d="m4 10 4 4 8-9"/></svg>`,
  close: `<svg viewBox="0 0 24 24"><path d="m6 6 12 12M18 6 6 18"/></svg>`,
};

function routeId() {
  const id = location.hash.replace(/^#\/?/, "").split("?")[0];
  return pages[id] ? id : "overview";
}

function pageIndex(id) {
  return allPages.findIndex((page) => page.id === id);
}

function shell(id) {
  const page = pages[id];
  const index = pageIndex(id);
  const previous = allPages[index - 1];
  const next = allPages[index + 1];
  const done = page.problemId && progress[page.problemId]?.status === "done";

  app.innerHTML = `
    <header class="topbar">
      <button class="icon-button mobile-menu" id="mobile-menu" aria-label="打开目录">${icons.menu}</button>
      <a href="#/overview" class="site-logo"><span class="logo-cube">K</span><span><strong>Kernel 随想录</strong><small>LEETGPU 系统学习路线</small></span></a>
      <nav><a href="#/overview">学习路线</a><a href="#/checklist">完整题单</a><a href="#/activity">刷题记录</a></nav>
      <div class="top-actions">
        <button class="search-trigger" id="search-trigger">${icons.search}<span>搜索文档</span><kbd>⌘ K</kbd></button>
        <button class="icon-button" id="theme-toggle" aria-label="切换主题">${icons.moon}</button>
        <a class="icon-button" href="https://github.com/ZaneZhang-67/leetgpu-roadmap" target="_blank" rel="noreferrer" aria-label="GitHub">${icons.github}</a>
      </div>
    </header>
    <div class="docs-layout">
      <aside class="sidebar ${sidebarOpen ? "open" : ""}">
        <div class="sidebar-head"><span>教程目录</span><button id="close-sidebar">${icons.close}</button></div>
        <nav class="sidebar-nav">${sections.map((section) => `
          <section><h3>${section.title}</h3>${section.pages.map((item) => `<a class="${item.id === id ? "active" : ""}" href="#/${item.id}"><span>${item.title}</span>${progress[item.id]?.status === "done" ? icons.check : ""}</a>`).join("")}</section>
        `).join("")}</nav>
        <div class="sidebar-progress"><div><span>路线进度</span><strong>${completedCount()} / ${problems.length}</strong></div><i><b style="width:${Math.round(completedCount()/problems.length*100)}%"></b></i></div>
      </aside>
      <div class="sidebar-mask ${sidebarOpen ? "show" : ""}" id="sidebar-mask"></div>
      <main class="doc-main">
        <article class="article">
          <div class="breadcrumb"><a href="#/overview">Kernel 随想录</a><span>›</span><span>${allPages[index]?.section || "学习路线"}</span></div>
          <div class="article-title-row">
            <h1>${page.title}<a class="heading-anchor" href="#/${id}">#</a></h1>
            ${page.problemId ? `<button id="complete-page" class="complete-button ${done ? "done" : ""}">${done ? icons.check + " 已完成这道题" : "标记为已完成"}</button>` : ""}
          </div>
          ${page.problemId ? "" : `<p class="lead">${page.description}</p>`}
          <div class="article-body">${specialPage(id) || page.body}</div>
          <div class="page-nav">
            ${previous ? `<a href="#/${previous.id}" class="prev"><small>上一篇</small><span>← ${previous.title}</span></a>` : "<span></span>"}
            ${next ? `<a href="#/${next.id}" class="next"><small>下一篇</small><span>${next.title} →</span></a>` : ""}
          </div>
          <div class="article-footer"><span>最后更新：2026-11-09</span><a href="https://github.com/ZaneZhang-67/leetgpu-roadmap" target="_blank" rel="noreferrer">在 GitHub 上查看源码</a></div>
        </article>
        <aside class="toc"><strong>本页目录</strong><nav id="toc-list"></nav><div class="toc-tools"><button id="export-progress">导出进度</button><label>导入进度<input id="import-progress" type="file" accept="application/json"></label></div></aside>
      </main>
    </div>
    <div class="search-modal ${searchOpen ? "show" : ""}" id="search-modal">
      <div class="search-panel"><div class="search-input">${icons.search}<input id="search-input" placeholder="搜索标题和文章内容…" autocomplete="off"><button id="close-search">ESC</button></div><div id="search-results" class="search-results"><p>输入关键词搜索教程</p></div></div>
    </div>
    <div id="toast" class="toast"></div>
  `;
}

function specialPage(id) {
  if (id === "checklist") return checklistMarkup();
  if (id === "activity") return activityMarkup();
  if (id === "interview") return interviewMarkup();
  return "";
}

function checklistMarkup() {
  return `<p>题单按知识依赖排列完成题目后点击状态按钮，数据会保存在当前浏览器</p><div class="checklist">${problems.map((p) => {
    const status = progress[p.id]?.status || "todo";
    return `<div class="check-row"><span class="check-no">${String(p.no).padStart(2,"0")}</span><div><strong>${p.title}（${p.titleZh}）</strong><small>${p.concepts.join(" / ")}</small></div><span class="tag ${p.difficulty.toLowerCase()}">${p.difficulty}</span><a href="${p.url}" target="_blank" rel="noreferrer">题目 ↗</a><button data-problem="${p.id}" class="status-${status}">${status === "done" ? "已完成" : status === "doing" ? "进行中" : "未开始"}</button></div>`;
  }).join("")}</div>`;
}

function activityMarkup() {
  return `<p>这张热力图只记录 LeetGPU 学习，不与 GitHub Contribution Calendar 混在一起</p><div class="activity-panel"><div class="activity-summary"><strong>${completedCount()}</strong><span>道题已完成</span></div><div id="heatmap"></div></div><h2 id="backup">进度备份</h2><p>进度保存在浏览器 localStorage你可以使用右侧的导入、导出按钮在不同设备间迁移</p>`;
}

function interviewMarkup() {
  return `<h2 id="index">线程与索引</h2><ul><li>怎样推导一维和二维全局索引？</li><li>为什么 block 数向上取整后需要边界判断？</li><li>一个 warp 包含多少线程？</li></ul><h2 id="memory">访存</h2><ul><li>什么是合并访存？</li><li>shared memory 解决了什么问题？</li><li>什么是 bank conflict？</li></ul><h2 id="gemm">GEMM</h2><ul><li>朴素 GEMM 为什么慢？</li><li>block tiling 和 thread tiling 分别复用什么？</li><li>W8A8 为什么用 INT32 累加？</li></ul>`;
}

function completedCount() {
  return problems.filter((p) => progress[p.id]?.status === "done").length;
}

function renderToc() {
  const toc = document.querySelector("#toc-list");
  const headings = [...document.querySelectorAll(".article-body h2")];
  toc.innerHTML = headings.length ? headings.map((h) => `<a href="#${h.id}">${h.textContent}</a>`).join("") : `<span>本页暂无小节</span>`;
}

function bind() {
  document.querySelector("#mobile-menu")?.addEventListener("click", () => { sidebarOpen = true; render(); });
  document.querySelector("#close-sidebar")?.addEventListener("click", () => { sidebarOpen = false; render(); });
  document.querySelector("#sidebar-mask")?.addEventListener("click", () => { sidebarOpen = false; render(); });
  document.querySelector("#search-trigger")?.addEventListener("click", openSearch);
  document.querySelector("#close-search")?.addEventListener("click", closeSearch);
  document.querySelector("#search-modal")?.addEventListener("click", (e) => { if (e.target.id === "search-modal") closeSearch(); });
  document.querySelector("#theme-toggle")?.addEventListener("click", toggleTheme);
  document.querySelector("#complete-page")?.addEventListener("click", () => {
    const id = routeId(); const pid = pages[id].problemId; const done = progress[pid]?.status === "done";
    progress[pid] = { ...progress[pid], status: done ? "todo" : "done", completedAt: done ? undefined : new Date().toISOString().slice(0,10) };
    saveProgress(progress); render(); toast(done ? "已取消完成状态" : "已加入学习热力图");
  });
  document.querySelectorAll("[data-problem]").forEach((button) => button.addEventListener("click", () => cycleStatus(button.dataset.problem)));
  document.querySelector("#export-progress")?.addEventListener("click", () => exportProgress(progress));
  document.querySelector("#import-progress")?.addEventListener("change", async (e) => {
    const file = e.target.files[0]; if (!file) return;
    try { progress = await importProgress(file); saveProgress(progress); render(); toast("进度导入成功"); } catch (err) { toast(`导入失败：${err.message}`); }
  });
  if (routeId() === "activity") renderHeatmap(document.querySelector("#heatmap"), activityCounts(progress));
  renderToc();
  bindSearchInput();
}

function cycleStatus(id) {
  const current = progress[id]?.status || "todo";
  const next = current === "todo" ? "doing" : current === "doing" ? "done" : "todo";
  progress[id] = { ...progress[id], status: next, completedAt: next === "done" ? new Date().toISOString().slice(0,10) : undefined };
  saveProgress(progress); render();
}

function openSearch() { searchOpen = true; render(); setTimeout(() => document.querySelector("#search-input")?.focus(), 0); }
function closeSearch() { searchOpen = false; render(); }
function bindSearchInput() {
  const input = document.querySelector("#search-input"); if (!input) return;
  input.addEventListener("input", () => {
    const q = input.value.trim().toLowerCase(); const results = document.querySelector("#search-results");
    if (!q) { results.innerHTML = "<p>输入关键词搜索教程</p>"; return; }
    const found = allPages.filter((x) => `${x.title} ${pages[x.id].description} ${pages[x.id].body.replace(/<[^>]+>/g," ")}`.toLowerCase().includes(q)).slice(0,10);
    results.innerHTML = found.length ? found.map((x) => `<a href="#/${x.id}"><small>${x.section}</small><strong>${x.title}</strong><span>${pages[x.id].description}</span></a>`).join("") : "<p>没有搜索结果</p>";
    results.querySelectorAll("a").forEach((a) => a.addEventListener("click", () => { searchOpen = false; }));
  });
}
function toggleTheme() { const light = document.documentElement.classList.toggle("light"); localStorage.setItem("kernel-doc-theme", light ? "light" : "dark"); }
function toast(message) { const el = document.querySelector("#toast"); el.textContent = message; el.classList.add("show"); setTimeout(() => el.classList.remove("show"), 1800); }
function render() { shell(routeId()); bind(); }

if (localStorage.getItem("kernel-doc-theme") === "light") document.documentElement.classList.add("light");
window.addEventListener("hashchange", () => { sidebarOpen = false; searchOpen = false; window.scrollTo(0,0); render(); });
window.addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openSearch(); } if (e.key === "Escape" && searchOpen) closeSearch(); });
if (!location.hash) history.replaceState(null, "", "#/overview");
render();
