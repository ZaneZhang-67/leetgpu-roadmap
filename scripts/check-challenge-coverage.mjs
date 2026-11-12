import { readFile } from "node:fs/promises"
import { basename, dirname, relative, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import fg from "fast-glob"
import matter from "gray-matter"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const catalog = JSON.parse(await readFile(resolve(root, "data/leetgpu-challenges.json"), "utf8"))
const files = await fg("content/problems/*.md", { cwd: root, absolute: true })
const requiredHeadings = ["题目定义", "模板代码解析", "实现思路", "参考实现"]
const pageIdAliases = new Map([[2, "naive-gemm"]])
const violations = []

if (catalog.count !== catalog.challenges.length) {
  violations.push(`data/leetgpu-challenges.json: count=${catalog.count}，实际题目数=${catalog.challenges.length}`)
}

const catalogIds = catalog.challenges.map(({ id }) => id)
const catalogSlugs = catalog.challenges.map(({ slug }) => slug)
if (new Set(catalogIds).size !== catalogIds.length) violations.push("data/leetgpu-challenges.json: 题目 ID 重复")
if (new Set(catalogSlugs).size !== catalogSlugs.length) violations.push("data/leetgpu-challenges.json: 题目 slug 重复")
if (catalogIds.some((id, index) => index > 0 && id <= catalogIds[index - 1])) {
  violations.push("data/leetgpu-challenges.json: 题目必须按 ID 严格递增排序")
}

const byId = new Map(catalog.challenges.map((challenge) => [challenge.id, challenge]))
const seenLeetgpuIds = new Map()
const seenPageIds = new Map()

for (const file of files) {
  const source = await readFile(file, "utf8")
  const { data, content } = matter(source)
  const path = relative(root, file)
  const pageId = String(data.id || "")
  const leetgpuId = Number(data.leetgpuId)

  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(pageId)) {
    violations.push(`${path}: id 必须使用小写 kebab-case`)
  }
  if (basename(file) !== `${pageId}.md`) {
    violations.push(`${path}: 文件名必须与 id 一致，应为 ${pageId}.md`)
  }
  if (seenPageIds.has(pageId)) {
    violations.push(`${path}: 与 ${seenPageIds.get(pageId)} 重复使用页面 ID ${pageId}`)
  } else {
    seenPageIds.set(pageId, path)
  }

  for (const field of ["titleEn", "titleZh", "description", "problemUrl", "updatedAt"]) {
    if (!data[field]) violations.push(`${path}: 缺少 front matter 字段 ${field}`)
  }

  const updatedAt = data.updatedAt instanceof Date
    ? data.updatedAt.toISOString().slice(0, 10)
    : String(data.updatedAt || "")
  if (!/^\d{4}-\d{2}-\d{2}$/.test(updatedAt)) {
    violations.push(`${path}: updatedAt 必须使用 YYYY-MM-DD`)
  }

  const headings = [...content.matchAll(/^##\s+(.+?)\s*$/gm)].map((match) => match[1])
  let headingCursor = -1
  for (const heading of requiredHeadings) {
    const index = headings.indexOf(heading)
    if (index === -1) violations.push(`${path}: 缺少“## ${heading}”`)
    else if (index <= headingCursor) violations.push(`${path}: “## ${heading}”章节顺序错误`)
    headingCursor = Math.max(headingCursor, index)
  }

  const referenceStart = content.indexOf("## 参考实现")
  const referenceSection = referenceStart === -1 ? "" : content.slice(referenceStart)
  const referenceFences = [...referenceSection.matchAll(/^```([^\s`]+)\s*\n[\s\S]*?^```\s*$/gm)]
  if (referenceFences.length !== 1) {
    violations.push(`${path}: 参考实现章节必须且只能包含一个带语言标记的代码块`)
  }
  const fences = [...content.matchAll(/^```(.*)$/gm)]
  if (fences.length % 2 !== 0) violations.push(`${path}: 代码围栏没有成对闭合`)
  for (let index = 0; index < fences.length; index += 2) {
    if (!fences[index][1].trim()) violations.push(`${path}: 第 ${index / 2 + 1} 个代码块缺少语言标记`)
  }
  if (/\bTODO\b|章节建设中|待补充/.test(content)) violations.push(`${path}: 发布文章仍含占位内容`)

  if (!Number.isSafeInteger(leetgpuId) || leetgpuId <= 0) {
    violations.push(`${path}: leetgpuId 必须是正整数`)
    continue
  }

  const challenge = byId.get(leetgpuId)
  if (!challenge) {
    violations.push(`${path}: LeetGPU 清单中不存在题目 ${leetgpuId}`)
    continue
  }

  if (seenLeetgpuIds.has(leetgpuId)) {
    violations.push(`${path}: 与 ${seenLeetgpuIds.get(leetgpuId)} 重复使用 leetgpuId ${leetgpuId}`)
  } else {
    seenLeetgpuIds.set(leetgpuId, path)
  }

  const expectedPageId = pageIdAliases.get(leetgpuId) || challenge.slug
  if (pageId !== expectedPageId) {
    violations.push(`${path}: 题目 ${leetgpuId} 的页面 ID 应为 ${expectedPageId}`)
  }
  if (data.titleEn !== challenge.title) {
    violations.push(`${path}: titleEn 应为“${challenge.title}”，实际为“${data.titleEn}”`)
  }

  const expectedUrl = `https://leetgpu.com/challenges/${challenge.slug}`
  if (data.problemUrl !== expectedUrl) {
    violations.push(`${path}: problemUrl 应为 ${expectedUrl}`)
  }
}

if (violations.length) {
  console.error(`LeetGPU 题目覆盖检查失败：\n${violations.join("\n")}`)
  process.exit(1)
}

const missing = catalog.challenges.filter(({ id }) => !seenLeetgpuIds.has(id))
const covered = catalog.challenges.length - missing.length
console.log(`LeetGPU challenge coverage: ${covered}/${catalog.challenges.length}`)

if (missing.length) {
  console.log(`Missing ${missing.length}: ${missing.map(({ id, title }) => `${id} ${title}`).join(" | ")}`)
}

if (process.argv.includes("--strict") && missing.length) {
  process.exit(1)
}
