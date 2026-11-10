import { markdownPages } from "../generated/markdown-pages.js"

export const navigation = [
  { id: "vector-addition", title: "Vector Addition（向量加法）" },
  { id: "naive-gemm", title: "朴素 GEMM" },
]

export const defaultPageId = navigation[0].id

export const pages = Object.fromEntries(navigation.map(({ id }) => {
  const page = markdownPages[id]
  if (!page) throw new Error(`目录对应的文章不存在：${id}`)
  return [id, page]
}))
