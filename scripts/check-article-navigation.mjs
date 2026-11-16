import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import { navigation, pages, defaultPageId, loadPage } from "../src/data/docs.js"

const catalog = JSON.parse(await readFile(new URL("../data/leetgpu-challenges.json", import.meta.url), "utf8"))
const challengesById = new Map(catalog.challenges.map((challenge) => [challenge.id, challenge]))
const orderOf = (page) => page.kind === "guide" ? page.sortOrder : page.leetgpuId
const expectedIds = Object.values(pages)
  .sort((a, b) => orderOf(a) - orderOf(b) || a.problemId.localeCompare(b.problemId, "en"))
  .map(({ problemId }) => problemId)
const navigationIds = navigation.map(({ id }) => id)
const problemPages = Object.values(pages).filter(({ kind }) => kind !== "guide")

assert.deepEqual(navigationIds, expectedIds)
assert.deepEqual([...Object.keys(pages)].sort(), [...expectedIds].sort())
assert.equal(defaultPageId, expectedIds[0])
assert.equal(new Set(navigationIds).size, navigation.length)
assert.equal(new Set(Object.values(pages).map(orderOf)).size, navigation.length)
assert.equal(new Set(problemPages.map(({ leetgpuId }) => leetgpuId)).size, problemPages.length)

for (const { id, title, difficulty } of navigation) {
  const meta = pages[id]
  assert.ok(["easy", "medium", "hard"].includes(difficulty), `${id}: invalid difficulty ${difficulty}`)
  const page = await loadPage(id)
  assert.equal(title, meta.title, `${id}: navigation title must match the article title`)
  assert.equal(page.title, meta.title, `${id}: lazy page title must match the index`)
  assert.equal(page.problemId, id, `${id}: article and progress IDs must match`)
  assert.match(page.updatedAt, /^\d{4}-\d{2}-\d{2}$/, `${id}: updatedAt must be normalized as YYYY-MM-DD`)
  assert.ok(!page.body.includes("章节建设中"), `${id}: placeholder content is still visible`)
  assert.match(page.body, /<h2 id="[^"]+">/, `${id}: headings must have stable ids`)

  if (meta.kind !== "guide") {
    assert.equal(difficulty, challengesById.get(meta.leetgpuId)?.difficulty, `${id}: difficulty must match the LeetGPU catalog`)
    assert.ok(page.body.includes("参考实现"), `${id}: reference implementation is missing`)
    assert.ok(Object.keys(page.markdownCode).length, `${id}: code snippets are missing`)
    assert.ok(meta.problemUrl, `${id}: LeetGPU practice URL is missing`)
  } else {
    assert.equal(difficulty, "easy", `${id}: introductory guide must stay in the Easy group`)
    assert.equal(meta.problemUrl, "", `${id}: guide must not expose a LeetGPU practice URL`)
  }
}

console.log(`Article navigation contains ${navigation.length} lazy-loaded pages in configured order`)
