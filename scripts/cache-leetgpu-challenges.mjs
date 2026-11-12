import { mkdir, readFile, rename, writeFile } from "node:fs/promises"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const catalogPath = resolve(root, "data/leetgpu-challenges.json")
const cacheDirectory = resolve(root, ".cache/leetgpu")
const catalog = JSON.parse(await readFile(catalogPath, "utf8"))
const args = process.argv.slice(2)
const idsArgument = args.find((argument) => argument.startsWith("--ids="))
const concurrencyArgument = args.find((argument) => argument.startsWith("--concurrency="))
const requestedIds = idsArgument
  ? new Set(idsArgument.slice("--ids=".length).split(",").filter(Boolean).map(Number))
  : null
const concurrency = concurrencyArgument ? Number(concurrencyArgument.slice("--concurrency=".length)) : 1
const refresh = args.includes("--refresh")

if (requestedIds && [...requestedIds].some((id) => !Number.isSafeInteger(id) || id <= 0)) {
  throw new TypeError("--ids 必须是用逗号分隔的正整数")
}
if (!Number.isSafeInteger(concurrency) || concurrency < 1 || concurrency > 4) {
  throw new TypeError("--concurrency 必须是 1 到 4 之间的整数")
}

const selected = catalog.challenges.filter(({ id }) => !requestedIds || requestedIds.has(id))
if (requestedIds && selected.length !== requestedIds.size) {
  const known = new Set(selected.map(({ id }) => id))
  const unknown = [...requestedIds].filter((id) => !known.has(id))
  throw new Error(`题目清单中不存在这些 ID：${unknown.join(", ")}`)
}

await mkdir(cacheDirectory, { recursive: true })

const sleep = (milliseconds) => new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds))

async function hasValidCache(outputPath, expectedId) {
  if (refresh) return false
  try {
    const payload = JSON.parse(await readFile(outputPath, "utf8"))
    return Number(payload.id) === expectedId && typeof payload.spec === "string" && Array.isArray(payload.starterCode)
  } catch (error) {
    if (error.code === "ENOENT" || error instanceof SyntaxError) return false
    throw error
  }
}

async function fetchChallenge(challenge) {
  const outputPath = resolve(cacheDirectory, `${challenge.id}-${challenge.slug}.json`)
  if (await hasValidCache(outputPath, challenge.id)) return "skipped"

  const url = `https://api.leetgpu.com/api/v1/challenges/${challenge.id}`
  for (let attempt = 1; attempt <= 6; attempt += 1) {
    const response = await fetch(url, {
      headers: {
        Accept: "application/json",
        "User-Agent": "leetgpu-roadmap-challenge-cache",
      },
    })

    if (response.ok) {
      const payload = await response.json()
      if (Number(payload.id) !== challenge.id) {
        throw new Error(`题目 ${challenge.id} 返回了错误的 ID：${payload.id}`)
      }
      if (typeof payload.spec !== "string" || !Array.isArray(payload.starterCode)) {
        throw new TypeError(`题目 ${challenge.id} 缺少 spec 或 starterCode`)
      }
      const temporaryPath = `${outputPath}.tmp`
      await writeFile(temporaryPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8")
      await rename(temporaryPath, outputPath)
      await sleep(150)
      return "fetched"
    }

    if (response.status !== 429 || attempt === 6) {
      throw new Error(`获取题目 ${challenge.id} 失败：HTTP ${response.status}`)
    }

    const retryAfter = Number(response.headers.get("retry-after"))
    const waitMilliseconds = Number.isFinite(retryAfter) && retryAfter > 0
      ? retryAfter * 1000
      : 1000 * (2 ** (attempt - 1))
    await sleep(waitMilliseconds)
  }
}

let cursor = 0
let fetched = 0
let skipped = 0
async function worker() {
  while (cursor < selected.length) {
    const challenge = selected[cursor]
    cursor += 1
    const result = await fetchChallenge(challenge)
    if (result === "fetched") fetched += 1
    else skipped += 1
  }
}

await Promise.all(Array.from({ length: Math.min(concurrency, selected.length) }, worker))
console.log(`Challenge cache ready: fetched=${fetched}, skipped=${skipped}, total=${selected.length}`)
console.log(cacheDirectory)
