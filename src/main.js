import "./styles.css";
import { navigation, pages, defaultPageId, loadPage, loadSearchIndex } from "./data/docs.js";
import { starterProgress } from "./data/problems.js";
import { loadProgress, saveProgress } from "./utils/progress.js";

const app = document.querySelector("#app");
const allPages = navigation;
let progress = loadProgress(starterProgress);
let sidebarOpen = false;
let searchOpen = false;
let searchItems = null;
let searchError = null;
let currentPage = null;
let renderVersion = 0;

const icons = {
  menu: `<svg viewBox="0 0 24 24"><path d="M4 7h16M4 12h16M4 17h16"/></svg>`,
  search: `<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></svg>`,
  github: `<svg viewBox="0 0 24 24"><path d="M12 2a10 10 0 0 0-3.2 19.5c.5.1.7-.2.7-.5v-1.7c-2.8.6-3.4-1.2-3.4-1.2-.5-1.2-1.1-1.5-1.1-1.5-.9-.6.1-.6.1-.6 1 .1 1.5 1 1.5 1 .9 1.5 2.4 1.1 2.9.8.1-.6.4-1.1.7-1.3-2.2-.3-4.6-1.1-4.6-5 0-1.1.4-2 1-2.7-.1-.2-.4-1.3.1-2.6 0 0 .8-.3 2.8 1a9.6 9.6 0 0 1 5 0c1.9-1.3 2.8-1 2.8-1 .5 1.3.2 2.4.1 2.6.6.7 1 1.6 1 2.7 0 3.9-2.4 4.7-4.6 5 .4.3.7.9.7 1.9V21c0 .3.2.6.7.5A10 10 0 0 0 12 2Z"/></svg>`,
  moon: `<svg viewBox="0 0 24 24"><path d="M20 15.7A8.5 8.5 0 0 1 8.3 4 8.5 8.5 0 1 0 20 15.7Z"/></svg>`,
  check: `<svg viewBox="0 0 20 20"><path d="m4 10 4 4 8-9"/></svg>`,
  close: `<svg viewBox="0 0 24 24"><path d="m6 6 12 12M18 6 6 18"/></svg>`,
};

function routeId() {
  const id = location.hash.replace(/^#\/?/, "").split("?")[0].split("#")[0];
  return Object.hasOwn(pages, id) ? id : defaultPageId;
}

function pageIndex(id) { return allPages.findIndex((page) => page.id === id); }
function completedCount() { return allPages.filter((page) => progress[page.id]?.status === "done").length; }

function shell(id, page, state = "ready") {
  const meta = pages[id];
  const index = pageIndex(id);
  const previous = allPages[index - 1];
  const next = allPages[index + 1];
  const done = progress[id]?.status === "done";
  const loadingBody = state === "loading" ? `<p role="status">正在加载文章：${meta.title}</p>` : "";
  const errorBody = state === "error" ? `<p role="alert">文章加载失败：${meta.title}（${id}），${page}</p>` : "";
  const readyBody = state === "ready" ? page.body : loadingBody || errorBody;

  app.innerHTML = `
    <header class="topbar">
      <button class="icon-button mobile-menu" id="mobile-menu" aria-label="打开目录">${icons.menu}</button>
      <a href="#/${defaultPageId}" class="site-logo"><span class="logo-cube">K</span><span><strong>Kernel 随想录</strong><small>LEETGPU 系统学习路线</small></span></a>
      <div class="top-actions">
        <button class="search-trigger" id="search-trigger">${icons.search}<span>搜索文档</span><kbd>⌘ K</kbd></button>
        <button class="icon-button" id="theme-toggle" aria-label="切换主题">${icons.moon}</button>
        <a class="icon-button" href="https://github.com/ZaneZhang-67/leetgpu-roadmap" target="_blank" rel="noreferrer" aria-label="GitHub">${icons.github}</a>
      </div>
    </header>
    <div class="docs-layout">
      <aside class="sidebar ${sidebarOpen ? "open" : ""}">
        <div class="sidebar-head"><span>教程目录</span><button id="close-sidebar">${icons.close}</button></div>
        <nav class="sidebar-nav">${allPages.map((item) => `<a class="${item.id === id ? "active" : ""}" href="#/${item.id}"><span>${item.title}</span>${progress[item.id]?.status === "done" ? icons.check : ""}</a>`).join("")}</nav>
        <div class="sidebar-progress"><div><span>路线进度</span><strong>${completedCount()} / ${allPages.length}</strong></div><i><b style="width:${Math.round(completedCount()/allPages.length*100)}%"></b></i></div>
      </aside>
      <div class="sidebar-mask ${sidebarOpen ? "show" : ""}" id="sidebar-mask"></div>
      <main class="doc-main">
        <article class="article">
          <div class="breadcrumb"><a href="#/${defaultPageId}">Kernel 随想录</a><span>›</span><span>${meta.title}</span></div>
          <div class="problem-hero">
            <div class="problem-heading">
              <h1>${meta.titleEn}<a class="heading-anchor" href="#/${id}" aria-label="链接到 ${meta.titleEn}">#</a></h1>
              <div class="problem-subtitle">
                <span class="problem-title-zh">${meta.titleZh}</span><span class="subtitle-separator" aria-hidden="true">·</span>
                <label class="completion-toggle" title="${done ? "取消完成状态" : "标记为已完成"}">
                  <input id="complete-page" type="checkbox" ${done ? "checked" : ""} aria-label="${done ? "取消完成状态" : "标记为已完成"}">
                  <span class="completion-icon" aria-hidden="true">${icons.check}</span><span class="completion-label">${done ? "已完成" : "未完成"}</span>
                </label>
              </div>
            </div>
            <a class="problem-link-button" href="${meta.problemUrl}" target="_blank" rel="noreferrer">在 LeetGPU 中练习 <span aria-hidden="true">↗</span></a>
          </div>
          <div class="problem-hero-divider" aria-hidden="true"></div>
          <div class="article-body">${readyBody}</div>
          ${state === "ready" ? `<div class="page-nav">${previous ? `<a href="#/${previous.id}" class="prev"><small>上一篇</small><span>← ${previous.title}</span></a>` : "<span></span>"}${next ? `<a href="#/${next.id}" class="next"><small>下一篇</small><span>${next.title} →</span></a>` : ""}</div>` : ""}
          <div class="article-footer"><span>最后更新：${meta.updatedAt || "未注明"}</span><a href="https://github.com/ZaneZhang-67/leetgpu-roadmap" target="_blank" rel="noreferrer">在 GitHub 上查看源码</a></div>
        </article>
        <aside class="toc"><strong>本页目录</strong><nav id="toc-list"></nav></aside>
      </main>
    </div>
    <div class="search-modal ${searchOpen ? "show" : ""}" id="search-modal">
      <div class="search-panel"><div class="search-input">${icons.search}<input id="search-input" placeholder="搜索标题和文章内容…" autocomplete="off" ${searchItems || searchError ? "" : "disabled"}><button id="close-search">ESC</button></div><div id="search-results" class="search-results"><p>${searchError ? `搜索索引加载失败：${searchError}` : searchItems ? "输入关键词搜索教程" : "正在加载搜索索引…"}</p></div></div>
    </div>
    <div id="toast" class="toast"></div>`;
}

function renderToc() {
  const toc = document.querySelector("#toc-list");
  if (!toc) return;
  const headings = [...document.querySelectorAll(".article-body h2")];
  toc.innerHTML = headings.length ? headings.map((heading) => `<a href="#/${routeId()}?section=${encodeURIComponent(heading.id)}" data-section="${heading.id}">${heading.textContent}</a>`).join("") : `<span>本页暂无小节</span>`;
  toc.querySelectorAll("[data-section]").forEach((link) => link.addEventListener("click", (event) => {
    event.preventDefault();
    document.getElementById(link.dataset.section)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }));
}

function bind() {
  document.querySelector("#mobile-menu")?.addEventListener("click", () => { sidebarOpen = true; renderCurrent(); });
  document.querySelector("#close-sidebar")?.addEventListener("click", () => { sidebarOpen = false; renderCurrent(); });
  document.querySelector("#sidebar-mask")?.addEventListener("click", () => { sidebarOpen = false; renderCurrent(); });
  document.querySelector("#search-trigger")?.addEventListener("click", openSearch);
  document.querySelector("#close-search")?.addEventListener("click", closeSearch);
  document.querySelector("#search-modal")?.addEventListener("click", (event) => { if (event.target.id === "search-modal") closeSearch(); });
  document.querySelector("#theme-toggle")?.addEventListener("click", toggleTheme);
  document.querySelectorAll("[data-copy-code]").forEach((button) => button.addEventListener("click", async () => {
    const code = currentPage?.markdownCode?.[button.dataset.copyCode];
    if (!code) throw new Error(`复制代码失败：${button.dataset.copyCode} 不在当前文章 ${routeId()} 中`);
    await navigator.clipboard.writeText(code);
    const original = button.textContent;
    button.textContent = "已复制"; button.classList.add("copied");
    window.setTimeout(() => { button.textContent = original; button.classList.remove("copied"); }, 1600);
  }));
  document.querySelector("#complete-page")?.addEventListener("change", (event) => {
    const id = routeId(); const checked = event.target.checked;
    progress[id] = { ...progress[id], status: checked ? "done" : "todo", completedAt: checked ? new Date().toISOString().slice(0,10) : undefined };
    saveProgress(progress); renderCurrent(); toast(checked ? "已标记为完成" : "已取消完成状态");
  });
  renderToc(); bindSearchInput();
}

function renderCurrent() {
  const id = routeId();
  shell(id, currentPage?.problemId === id ? currentPage : null, currentPage?.problemId === id ? "ready" : "loading");
  bind();
}

async function render() {
  const id = routeId();
  const version = ++renderVersion;
  currentPage = null;
  shell(id, null, "loading"); bind();
  try {
    const page = await loadPage(id);
    if (version !== renderVersion || id !== routeId()) return;
    currentPage = page;
    shell(id, page, "ready"); bind();
  } catch (error) {
    if (version !== renderVersion || id !== routeId()) return;
    const detail = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    shell(id, detail, "error"); bind();
    console.error(`文章加载失败 [${id}]`, error);
  }
}

async function openSearch() {
  searchOpen = true;
  if (searchItems) {
    renderCurrent();
    setTimeout(() => document.querySelector("#search-input")?.focus(), 0);
    return;
  }

  searchError = null;
  renderCurrent();
  try {
    searchItems = await loadSearchIndex();
  } catch (error) {
    searchError = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    console.error("全文搜索索引加载失败", error);
  }
  if (searchOpen) {
    renderCurrent();
    if (searchItems) setTimeout(() => document.querySelector("#search-input")?.focus(), 0);
  }
}
function closeSearch() { searchOpen = false; renderCurrent(); }
function bindSearchInput() {
  const input = document.querySelector("#search-input"); if (!input) return;
  input.addEventListener("input", () => {
    const query = input.value.trim().toLowerCase(); const results = document.querySelector("#search-results");
    if (!query) { results.innerHTML = "<p>输入关键词搜索教程</p>"; return; }
    const found = (searchItems || []).filter((item) => `${item.title} ${item.description} ${item.text}`.toLowerCase().includes(query)).slice(0, 10);
    results.innerHTML = found.length ? found.map((item) => `<a href="#/${item.id}"><strong>${item.title}</strong><span>${item.description}</span></a>`).join("") : "<p>没有搜索结果</p>";
    results.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => { searchOpen = false; }));
  });
}
function toggleTheme() { const light = document.documentElement.classList.toggle("light"); localStorage.setItem("kernel-doc-theme", light ? "light" : "dark"); }
function toast(message) { const element = document.querySelector("#toast"); if (!element) return; element.textContent = message; element.classList.add("show"); setTimeout(() => element.classList.remove("show"), 1800); }

if (localStorage.getItem("kernel-doc-theme") === "light") document.documentElement.classList.add("light");
window.addEventListener("hashchange", () => { sidebarOpen = false; searchOpen = false; window.scrollTo(0,0); render(); });
window.addEventListener("keydown", (event) => { if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); openSearch(); } if (event.key === "Escape" && searchOpen) closeSearch(); });
if (!location.hash) history.replaceState(null, "", `#/${defaultPageId}`);
render();
