import { mkdir, readFile, rm, writeFile } from "node:fs/promises"
import { dirname, relative, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import fg from "fast-glob"
import matter from "gray-matter"
import { marked } from "marked"
import { codeToHtml } from "shiki"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const files = (await fg("content/problems/*.md", { cwd: root, absolute: true })).sort()
const generatedDir = resolve(root, "src/generated")
const pageModulesDir = resolve(generatedDir, "pages")
const records = []
const ids = new Set()
const leetgpuIds = new Set()

const languageLabels = {
  cpp: "CUDA C++",
  c: "C",
  python: "Python",
  py: "Python",
  triton: "Triton Python",
  mojo: "Mojo",
  text: "文本",
  bash: "Shell",
  json: "JSON",
  yaml: "YAML",
}

function escapeHtml(value = "") {
  return String(value).replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
}

function codeTitle(language, role) {
  if (role === "reference") return "参考实现"
  if (language === "text") return "输入 / 输出"
  return "代码片段"
}

function codeBlockHtml({ id, code, language, index, role }) {
  const codeId = `${id}-markdown-code-${index}`
  return {
    codeId,
    fileName: codeTitle(language, role),
    label: languageLabels[language] || language.toUpperCase(),
    role,
    code,
  }
}

function normalizeUpdatedAt(value, file) {
  if (value === undefined || value === null || value === "") return ""
  if (value instanceof Date && !Number.isNaN(value.valueOf())) return value.toISOString().slice(0, 10)
  const text = String(value).trim()
  const match = /^(\d{4})-(\d{2})-(\d{2})(?:T.*)?$/.exec(text)
  if (!match) throw new Error(`${relative(root, file)} 的 updatedAt 必须是 YYYY-MM-DD`)
  const normalized = `${match[1]}-${match[2]}-${match[3]}`
  const date = new Date(`${normalized}T00:00:00Z`)
  if (Number.isNaN(date.valueOf()) || date.toISOString().slice(0, 10) !== normalized) {
    throw new Error(`${relative(root, file)} 的 updatedAt 不是有效日期：${text}`)
  }
  return normalized
}

function headingId(text, usedHeadingIds) {
  const base = String(text)
    .toLowerCase()
    .trim()
    .replace(/<[^>]+>/g, "")
    .replace(/&[a-z0-9#]+;/gi, "")
    .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
    .replace(/^-+|-+$/g, "") || "section"
  let id = base
  let suffix = 2
  while (usedHeadingIds.has(id)) id = `${base}-${suffix++}`
  usedHeadingIds.add(id)
  return id
}

function plainText(html) {
  return html
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ")
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, " ")
    .trim()
}

function searchableMarkdown(markdown) {
  return markdown.replace(/^```[^\n]*\n[\s\S]*?^```\s*$/gm, " ")
}

for (const file of files) {
  const source = await readFile(file, "utf8")
  const { data, content } = matter(source)
  for (const field of ["id", "leetgpuId", "titleEn", "titleZh", "description", "problemUrl", "updatedAt"]) {
    if (!data[field]) throw new Error(`${relative(root, file)} 缺少 front matter 字段：${field}`)
  }
  if (ids.has(data.id)) throw new Error(`题目 ID 重复：${data.id}`)
  ids.add(data.id)
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(data.id)) throw new Error(`${relative(root, file)} 的 id 必须使用小写 kebab-case`)
  if (!Number.isSafeInteger(data.leetgpuId) || data.leetgpuId <= 0) throw new Error(`${relative(root, file)} 的 leetgpuId 必须是正整数`)
  if (leetgpuIds.has(data.leetgpuId)) throw new Error(`LeetGPU ID 重复：${data.leetgpuId}`)
  leetgpuIds.add(data.leetgpuId)

  let codeIndex = 0
  const markdownCode = {}
  const tokens = marked.lexer(content, { gfm: true })
  const codeTokens = []
  let currentSection = ""
  marked.walkTokens(tokens, (token) => {
    if (token.type === "heading" && token.depth === 2) currentSection = token.text.trim()
    if (token.type === "code") {
      token.codeRole = currentSection === "参考实现" ? "reference" : "snippet"
      codeTokens.push(token)
    }
  })

  for (const token of codeTokens) {
    codeIndex += 1
    const language = token.lang || "text"
    const highlightLanguage = ({ cuda: "cpp", "c++": "cpp", py: "python", triton: "python" })[language] || language
    const block = codeBlockHtml({ id: data.id, code: token.text, language, index: codeIndex, role: token.codeRole })
    markdownCode[block.codeId] = token.text
    const highlighted = await codeToHtml(token.text, {
      lang: highlightLanguage,
      themes: { light: "light-plus", dark: "dark-plus" },
      defaultColor: false,
      transformers: [
        ...(language === "text" ? [{ pre(node) { this.addClassToHast(node, "plain-code") } }] : []),
        ...(language !== "text" ? [{
          pre(node) { this.addClassToHast(node, "has-line-numbers") },
          line(node, line) { node.properties["data-line"] = line },
        }] : []),
      ],
    })
    token.highlightedHtml = `<div class="code-block" data-code-id="${block.codeId}" data-code-role="${block.role}">
  <div class="code-block-header">
    <span class="code-file">${escapeHtml(block.fileName)}</span>
    <span class="code-language">${escapeHtml(block.label)}</span>
    <button class="copy-code-button" type="button" data-copy-code="${block.codeId}" aria-label="复制 ${escapeHtml(block.fileName)}">复制</button>
  </div>
  <div class="code-block-body">${highlighted}</div>
</div>`
  }

  const usedHeadingIds = new Set()
  const renderer = new marked.Renderer()
  renderer.image = ({ href, title, text }) => {
    const normalized = href.replace(/^\.\.\/\.\.\/public\//, "./")
    const titleAttribute = title ? ` title="${escapeHtml(title)}"` : ""
    return `<img class="article-image" src="${escapeHtml(normalized)}" alt="${escapeHtml(text)}"${titleAttribute} loading="lazy">`
  }
  renderer.heading = ({ tokens: headingTokens, depth }) => {
    const text = renderer.parser.parseInline(headingTokens)
    return `<h${depth} id="${escapeHtml(headingId(renderer.parser.parseInline(headingTokens).replace(/<[^>]+>/g, ""), usedHeadingIds))}">${text}</h${depth}>\n`
  }
  renderer.code = (token) => token.highlightedHtml
  const html = marked.parser(tokens, { gfm: true, renderer })
  if (html.includes("[object Promise]")) throw new Error(`${relative(root, file)} 生成了未解析的 Promise`)

  const page = {
    leetgpuId: data.leetgpuId,
    title: `${data.titleEn}（${data.titleZh}）`,
    titleEn: data.titleEn,
    titleZh: data.titleZh,
    description: data.description,
    problemId: data.id,
    problemUrl: data.problemUrl,
    updatedAt: normalizeUpdatedAt(data.updatedAt, file),
    markdownCode,
    body: html,
  }
  records.push({ page, searchText: plainText(`${data.titleEn} ${data.titleZh} ${data.description} ${searchableMarkdown(content)}`) })
}

records.sort((a, b) => a.page.leetgpuId - b.page.leetgpuId || a.page.problemId.localeCompare(b.page.problemId, "en"))
const expectedPageModulesDir = resolve(root, "src/generated/pages")
if (pageModulesDir !== expectedPageModulesDir) {
  throw new Error(`生成目录校验失败：${pageModulesDir}`)
}
await rm(pageModulesDir, { recursive: true, force: true })
await mkdir(pageModulesDir, { recursive: true })

for (const { page } of records) {
  const output = `// This file is generated by scripts/generate-content.mjs\nexport default ${JSON.stringify(page, null, 2)}\n`
  await writeFile(resolve(pageModulesDir, `${page.problemId}.js`), output, "utf8")
}

const index = Object.fromEntries(records.map(({ page }) => [page.problemId, {
  leetgpuId: page.leetgpuId,
  title: page.title,
  titleEn: page.titleEn,
  titleZh: page.titleZh,
  description: page.description,
  problemId: page.problemId,
  problemUrl: page.problemUrl,
  updatedAt: page.updatedAt,
}]))
const navigation = records.map(({ page }) => ({ id: page.problemId, title: page.title }))
const searchIndex = records.map(({ page, searchText }) => ({ id: page.problemId, title: page.title, description: page.description, text: searchText }))
const loaders = records.map(({ page }) => `  ${JSON.stringify(page.problemId)}: () => import("./pages/${page.problemId}.js"),`).join("\n")

await writeFile(resolve(generatedDir, "markdown-index.js"), `// This file is generated by scripts/generate-content.mjs\nexport const markdownPages = ${JSON.stringify(index, null, 2)}\n\nexport const markdownNavigation = ${JSON.stringify(navigation, null, 2)}\n`, "utf8")
await writeFile(resolve(generatedDir, "markdown-search-index.js"), `// This file is generated by scripts/generate-content.mjs\nexport const markdownSearchIndex = ${JSON.stringify(searchIndex, null, 2)}\n`, "utf8")
await writeFile(resolve(generatedDir, "markdown-pages.js"), `// This file is generated by scripts/generate-content.mjs\nexport const markdownPageLoaders = {\n${loaders}\n}\n`, "utf8")
console.log(`Generated ${records.length} Markdown pages as deterministic lazy-loaded modules`)
