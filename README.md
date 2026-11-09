# Kernel Path

一个可直接部署到 GitHub Pages 的 LeetGPU 刷题路线与学习记录站

## 功能

- 五阶段 CUDA / GPU 算子学习路线
- LeetGPU 题单、思考提示和复盘要点
- 未开始 / 进行中 / 已完成状态管理
- 本地学习笔记
- 独立的 LeetGPU 年度绿点热力图
- 进度 JSON 导入与导出
- GitHub Pages 自动部署

## 本地运行

要求 Node.js 20 或更高版本

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

## 修改题单

题目数据位于：

```text
src/data/problems.js
```

每道题可以维护：

- 阶段、难度和 LeetGPU 链接
- 知识点标签
- 数学公式
- 解题前提示
- 完成后的复盘要点

## 进度数据

进度默认保存在浏览器 `localStorage`，不会发送到服务器

页面右上角提供：

- **导出进度**：下载 JSON 备份
- **导入进度**：在另一台设备恢复

如果希望让仓库内置自己的初始进度，可以修改 `src/data/problems.js` 中的 `starterProgress`

## 注意

这是独立学习站，不隶属于 LeetGPU题目链接指向 LeetGPU，站内只维护个人路线、提示和复盘记录
