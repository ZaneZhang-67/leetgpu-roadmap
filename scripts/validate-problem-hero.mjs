import { chromium } from "playwright"
import { mkdir } from "node:fs/promises"

const outputDir = new URL("../artifacts/problem-hero/", import.meta.url)
await mkdir(outputDir, { recursive: true })

const browser = await chromium.launch({ headless: true })
const context = await browser.newContext({ colorScheme: "dark" })
const page = await context.newPage()
const url = "http://127.0.0.1:4173/#/vector-addition"

async function openAt(width, height, name) {
  await page.setViewportSize({ width, height })
  await page.goto(url)
  await page.waitForSelector(".problem-hero")
  if (width <= 760) await page.evaluate(() => document.querySelector("#close-sidebar")?.click())
  await page.screenshot({ path: new URL(`${name}.png`, outputDir).pathname, fullPage: true })
  return page.evaluate(() => {
    const hero = document.querySelector(".problem-hero")
    const heading = document.querySelector(".problem-heading h1")
    const link = document.querySelector(".problem-link-button")
    const subtitle = document.querySelector(".problem-subtitle")
    const divider = document.querySelector(".problem-hero-divider")
    const bodyHeading = document.querySelector(".article-body h2")
    const rect = (node) => node.getBoundingClientRect()
    return {
      viewport: { width: innerWidth, scrollWidth: document.documentElement.scrollWidth },
      title: heading.firstChild.textContent,
      titleSize: getComputedStyle(heading).fontSize,
      titleWeight: getComputedStyle(heading).fontWeight,
      subtitle: subtitle.textContent.replace(/\s+/g, " ").trim(),
      linkText: link.textContent.replace(/\s+/g, " ").trim(),
      linkHeight: rect(link).height,
      linkTopDelta: Math.abs(rect(link).top - rect(heading).top),
      linkTarget: link.target,
      heroColumns: getComputedStyle(hero).gridTemplateColumns,
      dividerToBody: rect(bodyHeading).top - rect(divider).bottom,
    }
  })
}

const desktop = await openAt(1440, 900, "vector-addition-desktop")
const mobile = await openAt(390, 844, "vector-addition-mobile")

const longTitleLayout = await page.evaluate(() => {
  const title = document.querySelector(".problem-heading h1")
  const subtitle = document.querySelector(".problem-title-zh")
  const originalTitle = title.firstChild.textContent
  const originalSubtitle = subtitle.textContent
  title.firstChild.textContent = "A Very Long CUDA Vector Addition Challenge Title"
  subtitle.textContent = "具有较长中文名称的向量逐元素加法练习"
  const noOverflow = document.documentElement.scrollWidth === innerWidth
  const titleHeight = title.getBoundingClientRect().height
  title.firstChild.textContent = originalTitle
  subtitle.textContent = originalSubtitle
  return { noOverflow, titleHeight }
})

const checkbox = page.locator("#complete-page")
const initial = await checkbox.isChecked()
await checkbox.focus()
const focusOutline = await page.locator(".completion-toggle").evaluate((node) => getComputedStyle(node).outlineStyle)
await page.keyboard.press("Space")
await page.waitForTimeout(100)
const toggled = await page.locator("#complete-page").isChecked()
const stored = await page.evaluate(() => JSON.parse(localStorage.getItem("kernel-path-progress-v1") || "{}")["vector-addition"])
await page.keyboard.press("Space")

const anchor = page.locator(".heading-anchor")
const anchorHref = await anchor.getAttribute("href")
await anchor.focus()
const anchorOpacity = await anchor.evaluate((node) => getComputedStyle(node).opacity)

const result = {
  desktop,
  mobile,
  longTitleLayout,
  interaction: {
    initial,
    toggled,
    persistedStatus: stored?.status,
    focusOutline,
    anchorHref,
    anchorOpacity,
  },
}

console.log(JSON.stringify(result, null, 2))

const failures = []
if (desktop.title !== "Vector Addition") failures.push("desktop English title")
if (!desktop.subtitle.includes("向量加法")) failures.push("Chinese subtitle")
if (!desktop.linkText.includes("在 LeetGPU 中练习")) failures.push("CTA text")
if (desktop.linkHeight < 35 || desktop.linkHeight > 38) failures.push("CTA height")
if (desktop.linkTopDelta > 3) failures.push("CTA title alignment")
if (desktop.linkTarget !== "_blank") failures.push("CTA target")
if (desktop.dividerToBody < 31 || desktop.dividerToBody > 34) failures.push("divider spacing")
if (desktop.viewport.width !== desktop.viewport.scrollWidth) failures.push("desktop horizontal overflow")
if (mobile.viewport.width !== mobile.viewport.scrollWidth) failures.push("mobile horizontal overflow")
if (mobile.titleSize !== "28px") failures.push("mobile title size")
if (!mobile.heroColumns.startsWith("354px")) failures.push("mobile single column")
if (!longTitleLayout.noOverflow || longTitleLayout.titleHeight <= 34) failures.push("long title wrapping")
if (initial === toggled) failures.push("completion toggle")
if (stored?.status !== (toggled ? "done" : "todo")) failures.push("completion persistence")
if (focusOutline === "none") failures.push("completion keyboard focus")
if (anchorHref !== "#/vector-addition" || Number(anchorOpacity) <= 0) failures.push("anchor access")

await browser.close()
if (failures.length) {
  console.error("Validation failures: " + failures.join(", "))
  process.exit(1)
}
