import { markdownNavigation, markdownPages } from "../generated/markdown-index.js"
import { markdownPageLoaders } from "../generated/markdown-pages.js"

export const navigation = markdownNavigation
export const pages = markdownPages
export const defaultPageId = navigation[0].id

const pagePromises = new Map()
let searchIndexPromise

export function loadPage(id) {
  const loader = markdownPageLoaders[id]
  if (!loader) return Promise.reject(new Error(`文章加载器不存在：${id}`))
  if (!pagePromises.has(id)) {
    pagePromises.set(id, loader().then(({ default: page }) => {
      if (!page || page.problemId !== id) throw new Error(`文章模块 ID 不匹配：期望 ${id}，实际 ${page?.problemId ?? "空"}`)
      return page
    }).catch((error) => {
      pagePromises.delete(id)
      throw error
    }))
  }
  return pagePromises.get(id)
}

export function loadSearchIndex() {
  if (!searchIndexPromise) {
    searchIndexPromise = import("../generated/markdown-search-index.js")
      .then(({ markdownSearchIndex }) => {
        if (!Array.isArray(markdownSearchIndex)) throw new TypeError("全文搜索索引格式错误")
        return markdownSearchIndex
      })
      .catch((error) => {
        searchIndexPromise = undefined
        throw error
      })
  }
  return searchIndexPromise
}
