import { lstat, readFile } from "node:fs/promises"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import fg from "fast-glob"
import matter from "gray-matter"
import { marked } from "marked"

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const lineCount = (text) => (text.match(/\n/g) || []).length

// Marked has raw tokens, but no source positions. Match sibling tokens in order,
// line by line: list/blockquote prefixes disappear from children, not line breaks.
// Consuming every sibling (including code) avoids locating a duplicate in a sample.
function locateRaw(raw, source, cursor) {
  let start = cursor
  const lines = raw.split("\n")
  for (let index = 0; index < lines.length; index += 1) {
    // Marked also collapses whitespace after task-list checkbox markers.
    const pattern = lines[index].replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/[ \t]+/g, "[ \\t]+")
    const match = new RegExp(pattern).exec(source.slice(cursor))
    const found = match ? cursor + match.index : -1
    const lineEnd = source.indexOf("\n", cursor)
    if (found < 0 || (index > 0 && lineEnd >= 0 && found > lineEnd)) {
      throw new Error(`Could not locate Markdown token: ${JSON.stringify(raw)}`)
    }
    if (index === 0) start = found
    cursor = found + match[0].length
    if (index < lines.length - 1) {
      const newline = source.indexOf("\n", cursor)
      if (newline < 0) throw new Error(`Could not locate Markdown token: ${JSON.stringify(raw)}`)
      cursor = newline + 1
    }
  }
  return { start, end: cursor }
}

/** Collect rendered Markdown images and raw HTML img tags, with article line numbers. */
export function collectArticleImages(source) {
  const normalized = source.replace(/\r\n?/g, "\n")
  const { content } = matter(normalized)
  const firstLine = lineCount(normalized.slice(0, normalized.length - content.length)) + 1
  const images = []

  function visit(tokens, text, baseLine) {
    let cursor = 0
    for (const token of tokens) {
      if (!token.raw) continue
      const { start, end } = locateRaw(token.raw, text, cursor)
      const line = baseLine + lineCount(text.slice(0, start))
      cursor = end
      if (token.type === "image") {
        images.push({ href: token.href, line, kind: "markdown", raw: token.raw })
        // Nested tokens here describe alt text, not additional rendered images.
      } else if (token.type === "html") {
        // Comments are not rendered tags. Keep newlines for precise diagnostics.
        const html = token.raw.replace(/<!--[\s\S]*?(?:-->|$)/g, (comment) => comment.replace(/[^\n]/g, " "))
        for (const match of html.matchAll(/<img(?=[\s/>])/gi)) {
          images.push({ href: "<img>", line: line + lineCount(html.slice(0, match.index)), kind: "html" })
        }
      } else if (token.type === "list") {
        visit(token.items, token.raw, line)
      } else if (token.type === "table") {
        // GFM cells occupy one source line; marked unescapes pipes in cell text.
        for (const cell of token.header) visit(cell.tokens, cell.text, line)
        token.rows.forEach((row, index) => {
          for (const cell of row) visit(cell.tokens, cell.text, line + index + 2)
        })
      } else if (token.tokens) {
        visit(token.tokens, token.raw, line)
      }
    }
  }

  visit(marked.lexer(content, { gfm: true }), content, firstLine)
  return images
}

function localImageName(href) {
  if (/^[a-z][a-z\d+.-]*:/i.test(href)) {
    if (!/^https?:\/\//i.test(href)) throw new Error("only http:// and https:// external images are supported")
    // URL() alone accepts incomplete forms such as https:///host and backslashes.
    if (!/^https?:\/\/[^/\\\s?#]+(?:[/?#]|$)/i.test(href) || /[\\\u0000-\u0020\u007f]/u.test(href)) {
      throw new Error("invalid HTTP(S) image URL")
    }
    try {
      new URL(href)
    } catch {
      throw new Error("invalid HTTP(S) image URL")
    }
    return null
  }

  // Decap 3.8.4 resolves local media with basename(mediaPath), without URL
  // decoding or stripping query/hash. Require a literal name usable by both CMS
  // and the published site instead of accepting URLs only the browser resolves.
  if (/[?#%]/u.test(href)) {
    throw new Error("local images must use literal filenames without %, query or hash; Decap CMS does not URL-decode or strip URL suffixes; reinsert the image using the CMS media library")
  }
  const match = /^(?:\.\/)?uploads\/([^/]+)$/u.exec(href)
  if (!match) {
    throw new Error("local images must use uploads/<filename> or ./uploads/<filename>; root paths, ../../public and subdirectories are not supported")
  }
  const name = match[1]
  if (name === "." || name === ".." || /[/\\\u0000-\u001f\u007f]/u.test(name)) {
    throw new Error("image filename must be a single basename without traversal, separators or control characters")
  }
  return name
}

/** Validate one article without changing files or requesting external URLs. */
export async function checkArticleImages(source, { root = repositoryRoot, file = "<article>" } = {}) {
  const errors = []
  let images
  try {
    images = collectArticleImages(source)
  } catch (error) {
    return { imageCount: 0, errors: [`${file}: Markdown parsing failed: ${error.message}`] }
  }
  for (const image of images) {
    try {
      if (image.kind === "html") throw new Error("use Markdown ![alt](uploads/<filename>) instead of raw HTML <img>")
      const name = localImageName(image.href)
      if (name === null) continue
      if (/^!\[[\s\S]*\]\(\s*</u.test(image.raw)) {
        throw new Error("angle-bracket local destinations are not supported by native Decap CMS; reinsert the image using the CMS media library")
      }
      const target = resolve(root, "public/uploads", name)
      let info
      try {
        info = await lstat(target)
      } catch (error) {
        if (error.code === "ENOENT") throw new Error(`missing image file public/uploads/${name}`)
        throw error
      }
      if (!info.isFile()) throw new Error(`image must be a regular file (not a directory or symlink): public/uploads/${name}`)
    } catch (error) {
      errors.push(`${file}:${image.line}: ${JSON.stringify(image.href)} — ${error.message}`)
    }
  }
  return { imageCount: images.length, errors }
}

/** Match the same two non-recursive article collections as generate-content. */
export async function checkRepositoryImages(root = repositoryRoot) {
  const files = (await fg(["content/guides/*.md", "content/problems/*.md"], { cwd: root, onlyFiles: true })).sort()
  const result = { articleCount: files.length, imageCount: 0, errors: [] }
  for (const file of files) {
    const source = await readFile(resolve(root, file), "utf8")
    const checked = await checkArticleImages(source, { root, file })
    result.imageCount += checked.imageCount
    result.errors.push(...checked.errors)
  }
  return result
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const result = await checkRepositoryImages()
    if (result.errors.length) {
      console.error(`Article image check failed (${result.errors.length}):\n${result.errors.join("\n")}`)
      process.exitCode = 1
    } else {
      console.log(`Article images checked: ${result.articleCount} articles, ${result.imageCount} images`)
    }
  } catch (error) {
    console.error(`Article image check failed: ${error.message}`)
    process.exitCode = 1
  }
}
