import assert from "node:assert/strict"
import { navigation, pages, defaultPageId } from "../src/data/docs.js"

const expectedIds = ["vector-addition", "naive-gemm", "matrix-transpose", "color-inversion"]
assert.deepEqual(navigation.map(({ id }) => id), expectedIds)
assert.deepEqual(Object.keys(pages), expectedIds)
assert.equal(defaultPageId, "vector-addition")
assert.equal(new Set(navigation.map(({ id }) => id)).size, navigation.length)

for (const { id, title } of navigation) {
  const page = pages[id]
  assert.equal(title, page.title, `${id}: navigation title must match the article title`)
  assert.equal(page.problemId, id, `${id}: article and progress IDs must match`)
  assert.ok(page.body.includes("参考实现"), `${id}: reference implementation is missing`)
  assert.ok(Object.keys(page.markdownCode).length, `${id}: code snippets are missing`)
  assert.ok(!page.body.includes("章节建设中"), `${id}: placeholder content is still visible`)
}

console.log("Article navigation contains Vector Addition, Matrix Multiplication, Matrix Transpose, and Color Inversion")
