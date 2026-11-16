import assert from "node:assert/strict"
import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { basename, dirname, join, resolve } from "node:path"
import test from "node:test"
import { checkArticleImages, checkRepositoryImages, collectArticleImages } from "./check-article-images.mjs"

async function fixture(t, names = []) {
  const parent = resolve(tmpdir())
  const root = resolve(await mkdtemp(join(parent, "leetgpu-image-check-")))
  t.after(async () => {
    // Validate the resolved absolute deletion target before removing this fixture.
    const target = resolve(root)
    assert.equal(target, root)
    assert.equal(dirname(target), parent)
    assert.match(basename(target), /^leetgpu-image-check-[a-zA-Z0-9]+$/)
    await rm(target, { recursive: true, force: true })
  })
  await mkdir(join(root, "public/uploads"), { recursive: true })
  for (const name of names) await writeFile(join(root, "public/uploads", name), "fixture")
  return root
}

const missingImage = "![same](uploads/missing.png)"

test("missing images report article path and original line after BOM/CRLF front matter", async (t) => {
  const root = await fixture(t)
  const source = '\ufeff---\r\ntitle: "![ignored](uploads/meta.png)"\r\n---\r\n\r\n# Heading\r\n\r\n![diagram](uploads/missing.png)\r\n'
  const result = await checkArticleImages(source, { root, file: "content/guides/intro.md" })
  assert.equal(result.imageCount, 1)
  assert.equal(result.errors.length, 1)
  assert.match(result.errors[0], /^content\/guides\/intro\.md:7: "uploads\/missing\.png"/)
  assert.match(result.errors[0], /missing image file public\/uploads\/missing\.png/)
})

test("nested quotes, ordered lists and task lists retain each image line", () => {
  const source = [
    "> Intro",
    ">",
    "> 1. Outer",
    ">",
    ">    - [ ] **![first](uploads/first.png)**",
    ">      > Nested",
    ">      > ![second](uploads/second.png)",
    "",
    "-\t![third](uploads/third.png)",
    "\t![fourth](uploads/fourth.png)",
  ].join("\n")
  assert.deepEqual(collectArticleImages(source).map(({ href, line }) => [href, line]), [
    ["uploads/first.png", 5],
    ["uploads/second.png", 7],
    ["uploads/third.png", 9],
    ["uploads/fourth.png", 10],
  ])
})

test("duplicates and full, collapsed and shortcut reference images use occurrence lines", async (t) => {
  const root = await fixture(t)
  const source = [
    "```md", missingImage, "```", "",
    `${missingImage} ${missingImage}`,
    "",
    "> - ![same][diagram]",
    ">   ![diagram][]",
    "",
    "![diagram]",
    "",
    "[diagram]: uploads/missing.png 'Title'",
  ].join("\n")
  const result = await checkArticleImages(source, { root, file: "content/problems/one.md" })
  assert.equal(result.imageCount, 5)
  assert.deepEqual(result.errors.map((error) => Number(/one\.md:(\d+):/.exec(error)[1])), [5, 5, 7, 8, 10])
  assert.ok(result.errors.every((error) => error.includes('"uploads/missing.png"')))
})

test("emphasis, strikethrough and normalized loose task lists retain image positions", () => {
  const source = [
    `**${missingImage}** ~~${missingImage}~~`,
    "",
    `> _${missingImage}_`,
    "",
    `- [ ]     ${missingImage}`,
    "      ",
    `      ${missingImage}`,
  ].join("\n")
  // The last image is an indented code sample inside a loose task-list item.
  assert.deepEqual(collectArticleImages(source).map(({ line }) => line), [1, 1, 3, 5])
})

test("multiline images, links around images and GFM table cells have exact lines", () => {
  const source = [
    "> ![multi",
    "> line](uploads/a.png)",
    "",
    "[![linked](uploads/b.png)](https://example.org)",
    "",
    "| ![a\\|b](uploads/c.png) | another |",
    "| --- | --- |",
    "| ![d](uploads/d.png) | ![e](uploads/e.png) |",
  ].join("\n")
  assert.deepEqual(collectArticleImages(source).map(({ line }) => line), [1, 4, 6, 8, 8])
})

test("literal Chinese names and the optional ./ prefix match native CMS filenames", async (t) => {
  const root = await fixture(t, ["中文图.png", "plain.png"])
  const result = await checkArticleImages("![direct](uploads/中文图.png)\n![plain](./uploads/plain.png)", { root })
  assert.equal(result.imageCount, 2)
  assert.deepEqual(result.errors, [])
})

test("percent encoding and query/hash suffixes fail instead of hiding CMS lookup mismatches", async (t) => {
  const root = await fixture(t, ["中文图.png", "100%.png", "plain.png"])
  for (const href of [
    "uploads/%E4%B8%AD%E6%96%87%E5%9B%BE.png", "uploads/100%25.png", "uploads/100%.png",
    "uploads/plain.png?size=2", "uploads/plain.png#view", "uploads/plain.png?size=2#view",
  ]) {
    const result = await checkArticleImages(`![x](${href})`, { root })
    assert.equal(result.errors.length, 1, href)
    assert.match(result.errors[0], /Decap CMS does not URL-decode or strip URL suffixes/)
  }
})

test("angle-bracket local destinations fail with native CMS reinsertion advice", async (t) => {
  const root = await fixture(t, ["plain.png", "中文 图.png"])
  const result = await checkArticleImages("![x](<uploads/plain.png>)\n![y](<uploads/中文 图.png>)", { root })
  assert.equal(result.errors.length, 2)
  assert.ok(result.errors.every((error) => /angle-bracket.*CMS media library/.test(error)))
})

test("legacy, root, nested, traversal and malformed local paths fail explicitly", async (t) => {
  const root = await fixture(t, ["valid.png"])
  const paths = [
    "../../public/valid.png", "/uploads/valid.png", "//example.org/a.png",
    "valid.png", "public/uploads/valid.png", "././uploads/valid.png",
    "uploads/sub/valid.png", "uploads/../valid.png", "uploads/",
    "uploads/.", "uploads/..", "uploads/%2e%2e", "uploads/%2Fetc.png",
    "uploads/a%5Cb.png", "uploads/a%00.png", "uploads/a%0A.png",
    "uploads/a\\b.png", "uploads/%FF.png", "uploads/%ZZ.png", "uploads/100%.png", "",
  ]
  for (const href of paths) {
    const result = await checkArticleImages(`![x](<${href}>)`, { root, file: "content/problems/bad.md" })
    assert.equal(result.imageCount, 1, href)
    assert.equal(result.errors.length, 1, href)
    assert.match(result.errors[0], /^content\/problems\/bad\.md:1:/, href)
    assert.doesNotMatch(result.errors[0], /missing image file/, href)
  }
})

test("HTTP(S) images are accepted without network access", async (t) => {
  const root = await fixture(t)
  const result = await checkArticleImages([
    "![https](https://does-not-exist.invalid/image.png?width=2#view)",
    "![http](http://127.0.0.1:1/image.png)",
    "![uppercase](HTTPS://example.invalid/image.png)",
  ].join("\n"), { root })
  assert.equal(result.imageCount, 3)
  assert.deepEqual(result.errors, [])
})

test("unsupported protocols and malformed HTTP(S) URLs are errors, not skipped", async (t) => {
  const root = await fixture(t)
  for (const href of [
    "htps://example.org/a.png", "ftp://example.org/a.png", "data:image/png;base64,YQ==",
    "javascript:alert(1)", "file:///tmp/a.png", "blob:https://example.org/id",
    "https:example.org/a.png", "https:/example.org/a.png", "https:///example.org/a.png",
    "https://", "http://[broken/a.png", "https://example.org:bad/a.png",
    "https://example.org/a\\b.png",
  ]) {
    const result = await checkArticleImages(`![x](<${href}>)`, { root })
    assert.equal(result.imageCount, 1, href)
    assert.equal(result.errors.length, 1, href)
    assert.match(result.errors[0], /supported|invalid HTTP\(S\)/, href)
  }
})

test("fenced, indented, inline and nested code examples are ignored", () => {
  const source = [
    "```markdown", missingImage, '<img src="uploads/missing.png">', "```", "",
    `    ${missingImage}`, "",
    `\`${missingImage}\` and \`<img src='uploads/missing.png'>\``, "",
    "> - example",
    ">   ~~~md",
    `>   ${missingImage}`,
    '>   <img src="uploads/missing.png">',
    ">   ~~~",
    "",
    "<!-- <img src='uploads/missing.png'> -->",
  ].join("\n")
  assert.deepEqual(collectArticleImages(source), [])
})

test("raw HTML img tags in blocks and inline prose require Markdown", async (t) => {
  const root = await fixture(t, ["exists.png"])
  const source = [
    "<div>",
    '<img src="uploads/exists.png">',
    "</div>",
    "",
    "Text <IMG src='https://example.org/a.png' />",
    "",
    "> - <img",
    ">     src='uploads/exists.png'>",
  ].join("\n")
  const result = await checkArticleImages(source, { root, file: "content/guides/html.md" })
  assert.equal(result.errors.length, 3)
  assert.deepEqual(result.errors.map((error) => Number(/html\.md:(\d+):/.exec(error)[1])), [2, 5, 7])
  assert.ok(result.errors.every((error) => error.includes("use Markdown")))
})

test("a directory or symlink does not count as a regular uploaded file", async (t) => {
  const root = await fixture(t, ["real.png"])
  await mkdir(join(root, "public/uploads/folder.png"))
  await symlink("real.png", join(root, "public/uploads/link.png"))
  const result = await checkArticleImages("![folder](uploads/folder.png)\n![link](uploads/link.png)", { root })
  assert.equal(result.errors.length, 2)
  assert.ok(result.errors.every((error) => error.includes("regular file")))
})

test("repository check reads only immediate guide/problem Markdown files", async (t) => {
  const root = await fixture(t, ["present.png"])
  await mkdir(join(root, "content/guides"), { recursive: true })
  await mkdir(join(root, "content/problems/nested"), { recursive: true })
  await writeFile(join(root, "content/guides/intro.md"), "![ok](uploads/present.png)")
  await writeFile(join(root, "content/problems/one.md"), `# One\n\n${missingImage}`)
  await writeFile(join(root, "content/problems/nested/ignored.md"), missingImage)
  await writeFile(join(root, "content/problems/ignored.txt"), missingImage)
  await writeFile(join(root, "README.md"), missingImage)
  const result = await checkRepositoryImages(root)
  assert.equal(result.articleCount, 2)
  assert.equal(result.imageCount, 2)
  assert.equal(result.errors.length, 1)
  assert.match(result.errors[0], /^content\/problems\/one\.md:3:/)
})
