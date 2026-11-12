import { markdownPages } from "../generated/markdown-pages.js"

export const navigation = [
  { id: "vector-addition", title: "Vector Addition（向量加法）" },
  { id: "matrix-addition", title: "Matrix Addition（矩阵加法）" },
  { id: "naive-gemm", title: "Matrix Multiplication（矩阵乘法）" },
  { id: "matrix-transpose", title: "Matrix Transpose（矩阵转置）" },
  { id: "color-inversion", title: "Color Inversion（图像颜色反转）" },
]

export const defaultPageId = navigation[0].id

export const pages = Object.fromEntries(navigation.map(({ id }) => {
  const page = markdownPages[id]
  if (!page) throw new Error(`目录对应的文章不存在：${id}`)
  return [id, page]
}))
