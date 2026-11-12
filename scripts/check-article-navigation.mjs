import assert from "node:assert/strict"
import { navigation, pages, defaultPageId, loadPage } from "../src/data/docs.js"

const expectedIds = Object.values(pages)
  .sort((a, b) => a.leetgpuId - b.leetgpuId)
  .map(({ problemId }) => problemId)
const navigationIds = navigation.map(({ id }) => id)

assert.deepEqual(navigationIds, expectedIds)
assert.deepEqual([...Object.keys(pages)].sort(), [...expectedIds].sort())
assert.equal(defaultPageId, expectedIds[0])
assert.equal(new Set(navigationIds).size, navigation.length)
assert.equal(new Set(Object.values(pages).map(({ leetgpuId }) => leetgpuId)).size, navigation.length)

for (const { id, title } of navigation) {
  const meta = pages[id]
  const page = await loadPage(id)
  assert.equal(title, meta.title, `${id}: navigation title must match the article title`)
  assert.equal(page.title, meta.title, `${id}: lazy page title must match the index`)
  assert.equal(page.problemId, id, `${id}: article and progress IDs must match`)
  assert.match(page.updatedAt, /^\d{4}-\d{2}-\d{2}$/, `${id}: updatedAt must be normalized as YYYY-MM-DD`)
  assert.ok(page.body.includes("参考实现"), `${id}: reference implementation is missing`)
  assert.ok(Object.keys(page.markdownCode).length, `${id}: code snippets are missing`)
  assert.ok(!page.body.includes("章节建设中"), `${id}: placeholder content is still visible`)
  assert.match(page.body, /<h2 id="[^"]+">/, `${id}: headings must have stable ids`)
}

console.log(`Article navigation contains ${navigation.length} lazy-loaded LeetGPU solutions in challenge ID order`)
