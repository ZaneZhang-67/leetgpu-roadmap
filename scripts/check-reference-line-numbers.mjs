import { navigation, loadPage } from "../src/data/docs.js"

const violations = []

for (const { id } of navigation) {
  const page = await loadPage(id)
  const referenceHeading = page.body.search(/<h2 id="[^"]+">参考实现<\/h2>/)
  const nextHeading = referenceHeading === -1 ? -1 : page.body.indexOf("<h2 ", referenceHeading + 1)
  const referenceSection = referenceHeading === -1
    ? ""
    : page.body.slice(referenceHeading, nextHeading === -1 ? undefined : nextHeading)
  const codeIds = [...referenceSection.matchAll(/data-code-id="([^"]+)"/g)].map((match) => match[1])

  if (codeIds.length !== 1) {
    violations.push(`${id}: 参考实现应有一个代码块，实际为 ${codeIds.length}`)
    continue
  }

  const codeId = codeIds[0]
  const source = page.markdownCode[codeId]
  const expectedLines = typeof source === "string" ? source.split("\n").length : 0
  const actualLines = (referenceSection.match(/data-line="\d+"/g) || []).length

  if (!referenceSection.includes("has-line-numbers")) {
    violations.push(`${id}: Markdown 参考实现缺少 has-line-numbers`)
  }
  if (!source) {
    violations.push(`${id}: 参考实现原始代码未写入 markdownCode`)
  }
  if (actualLines !== expectedLines) {
    violations.push(`${id}: 参考实现应生成 ${expectedLines} 个行号，实际生成 ${actualLines} 个`)
  }
}

if (violations.length) {
  console.error(`参考实现行号检查失败：\n${violations.join("\n")}`)
  process.exit(1)
}

console.log(`All ${navigation.length} lazy Markdown reference implementations include complete line numbers`)
