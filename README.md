# Kernel Path

一个可直接部署到 GitHub Pages 的 LeetGPU 刷题路线与学习记录站

## 功能

- 按 LeetGPU 官方题号自动生成文章目录
- 左侧目录、站内搜索和上一篇 / 下一篇导航使用同一份文章列表
- 未开始 / 进行中 / 已完成状态管理
- 本地学习笔记
- 独立的 LeetGPU 年度绿点热力图
- GitHub Pages 自动部署

## 本地运行

要求 Node.js 22 或更高版本

```bash
pnpm install
pnpm run dev
```

构建生产版本：

```bash
pnpm run build
pnpm run preview
```

生成目录为 `dist/`

## 部署到 GitHub Pages

1. 在 GitHub 创建一个新仓库，例如 `leetgpu-roadmap`
2. 把本项目推送到仓库的 `main` 分支
3. 打开仓库的 **Settings → Pages**
4. 在 **Build and deployment** 中，将 Source 设为 **GitHub Actions**
5. `.github/workflows/deploy.yml` 会自动构建并发布

`vite.config.js` 使用相对资源路径，因此用户主页和项目主页都可以部署

## 在线编辑文章

题目正文保存在 `content/problems/*.md`，可以直接通过 GitHub 网页或 [github.dev](https://github.dev/ZaneZhang-67/leetgpu-roadmap) 编辑

站点同时提供 Decap CMS 后台：

```text
https://zanezhang-67.github.io/leetgpu-roadmap/admin/
```

后台字段定义位于 `public/admin/config.yml`；保存文章后，Decap CMS 会提交 Markdown 文件到 `main` 分支，并触发 GitHub Pages 自动部署

GitHub 后端登录需要 OAuth 服务首次启用时，按 [Decap CMS 的 GitHub 后端文档](https://decapcms.org/docs/github-backend/) 配置 GitHub OAuth App，并在 `public/admin/config.yml` 的 `backend` 中填写 OAuth 服务的 `base_url` 与 `auth_endpoint`；本地编辑内容时可以直接修改 Markdown 并运行构建，无需登录 CMS

Markdown 文件的 front matter 必须包含：

- `id`
- `leetgpuId`
- `titleEn`
- `titleZh`
- `description`
- `problemUrl`
- `updatedAt`

构建时，`scripts/generate-content.mjs` 会生成轻量索引、搜索索引、动态加载器和 `src/generated/pages/*.js` 单页模块；`src/generated/` 下的这些产物均由脚本维护，不要手工修改

## 题目清单与文章目录

公开题目清单保存在 `data/leetgpu-challenges.json`，来源是 LeetGPU 的公开题目接口按当前线上数据重新同步：

```bash
npm run sync:catalog
npm run check:catalog
```

`sync:catalog` 更新快照，`check:catalog` 只检查已提交快照是否仍与公开接口一致；日常构建只读取本地快照，不依赖外部网络

文章目录由 `scripts/generate-content.mjs` 根据各篇 Markdown 的 `leetgpuId` 自动生成，不需要手工维护 `src/data/docs.js`；新增文章后运行完整构建，即可更新左侧目录、站内搜索和上下篇顺序

覆盖率检查会验证题号、英文标题、题目链接和重复项：

```bash
npm run check:coverage
node scripts/check-challenge-coverage.mjs --strict
```

普通构建允许分批补齐题解；`--strict` 只在要求全部公开题目都已覆盖时通过

编写题解前，可以把题面与各语言 starter 缓存到本地 `.cache/leetgpu/`，该目录不会提交到 Git：

```bash
npm run cache:challenges -- --ids=1,2,3
npm run cache:challenges
```

## 进度数据

进度默认保存在浏览器 `localStorage`，不会发送到服务器

如果希望让仓库内置自己的初始进度，可以修改 `src/data/problems.js` 中的 `starterProgress`

## 注意

这是独立学习站，不隶属于 LeetGPU；题目链接指向 LeetGPU，站内只维护个人路线、提示和复盘记录
