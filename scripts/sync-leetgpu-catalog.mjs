import { mkdir, readFile, rename, writeFile } from "node:fs/promises"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const source = "https://api.leetgpu.com/api/v1/challenges"
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const outputPath = resolve(root, "data/leetgpu-challenges.json")
const checkOnly = process.argv.includes("--check")

if (checkOnly && process.argv.includes("--write")) {
  throw new Error("--check 与 --write 不能同时使用")
}

function slugify(title) {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
}

const response = await fetch(source, {
  headers: {
    Accept: "application/json",
    "User-Agent": "leetgpu-roadmap-catalog-sync",
  },
})

if (!response.ok) {
  throw new Error(`获取 LeetGPU 题目清单失败：HTTP ${response.status}`)
}

const payload = await response.json()
if (!Array.isArray(payload.challenges)) {
  throw new TypeError("LeetGPU 题目接口缺少 challenges 数组")
}

const allowedDifficulties = new Set(["easy", "medium", "hard"])
const ids = new Set()
const slugs = new Set()
const challenges = payload.challenges
  .filter((challenge) => challenge.listed !== false && challenge.hidden !== true)
  .map((challenge) => {
    const id = Number(challenge.id)
    const title = String(challenge.title || "").trim()
    const difficulty = String(challenge.difficultyLevel || "").toLowerCase()
    const accessTier = String(challenge.accessTier || "").toLowerCase()
    const slug = slugify(title)

    if (!Number.isSafeInteger(id) || id <= 0) throw new TypeError(`无效的题目 ID：${challenge.id}`)
    if (!title) throw new TypeError(`题目 ${id} 缺少标题`)
    if (!slug) throw new TypeError(`题目 ${id} 无法生成 slug`)
    if (!allowedDifficulties.has(difficulty)) throw new TypeError(`题目 ${id} 的难度无效：${difficulty}`)
    if (ids.has(id)) throw new Error(`题目 ID 重复：${id}`)
    if (slugs.has(slug)) throw new Error(`题目 slug 重复：${slug}`)

    ids.add(id)
    slugs.add(slug)
    return { id, slug, title, difficulty, accessTier }
  })
  .sort((a, b) => a.id - b.id)

const catalog = {
  schemaVersion: 1,
  source,
  count: challenges.length,
  challenges,
}
const serialized = `${JSON.stringify(catalog, null, 2)}\n`

if (checkOnly) {
  const current = await readFile(outputPath, "utf8")
  if (current !== serialized) {
    throw new Error("LeetGPU 题目快照与公开接口不一致，请运行 npm run sync:catalog")
  }
} else {
  await mkdir(dirname(outputPath), { recursive: true })
  const temporaryPath = `${outputPath}.tmp`
  await writeFile(temporaryPath, serialized, "utf8")
  await rename(temporaryPath, outputPath)
}

const counts = challenges.reduce((result, { difficulty }) => {
  result[difficulty] = (result[difficulty] || 0) + 1
  return result
}, {})
console.log(`${checkOnly ? "Verified" : "Synced"} ${challenges.length} LeetGPU challenges`)
console.log(`easy=${counts.easy || 0}, medium=${counts.medium || 0}, hard=${counts.hard || 0}`)
