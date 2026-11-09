import { readFileSync } from "node:fs"
import { codeSamples } from "../src/data/code-samples.js"

const generated = readFileSync(new URL("../src/generated/highlighted-code.js", import.meta.url), "utf8")
const references = Object.entries(codeSamples).filter(([, sample]) => sample.language === "cpp")
const violations = []

for (const [id, sample] of references) {
  if (sample.lineNumbers !== true) {
    violations.push(`${id}: 缺少 lineNumbers: true`)
    continue
  }

  const marker = `data-code-id=\\"${id}\\"`
  const start = generated.indexOf(marker)
  const next = generated.indexOf("data-code-id=\\\"", start + marker.length)
  const block = start === -1 ? "" : generated.slice(start, next === -1 ? undefined : next)
  const expectedLines = sample.code.split("\n").length
  const actualLines = (block.match(/data-line=\\"\d+\\"/g) || []).length

  if (!block.includes("has-line-numbers")) {
    violations.push(`${id}: 生成结果缺少 has-line-numbers`)
  }
  if (actualLines !== expectedLines) {
    violations.push(`${id}: 应生成 ${expectedLines} 个行号，实际生成 ${actualLines} 个`)
  }
}

if (violations.length) {
  console.error("参考实现行号检查失败：\n" + violations.join("\n"))
  process.exit(1)
}

console.log(`All ${references.length} reference implementations include line numbers`)
