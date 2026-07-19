# HEP-ML arXiv Digest

每天从 arXiv 抓取统计学习和机器学习论文，用国内模型筛选其对高能实验物理的潜在价值，并生成可订阅的 Atom、RSS 和静态网页。

## 是否需要本地电脑一直运行？

不需要。推荐流程是：

1. 在本地用 `--demo` 做一次离线验收；
2. 可选地用 DeepSeek API 做一次真实小规模验收；
3. 推送到 GitHub；
4. GitHub Actions 每天自动运行并发布 GitHub Pages。

部署后，本地电脑可以关闭。只有修改配置、提示词或排查失败时才需要再次在本地运行。

## 当前默认方案

- 数据源：`stat.ML`、`cs.LG`、`physics.data-an`、`hep-ex` RSS；
- 初筛：`deepseek-v4-flash`，关闭思考；
- 深度分析：`deepseek-v4-pro`，开启思考；
- 每日候选配额：方法雷达 40 篇、HEP 直接应用 10 篇、探索位 10 篇；
- 每日最多深度分析 5 篇；如果模型主动推荐不足，会从非无关候选中补足到 3 篇；
- 状态：`data/state.json`；
- 首次运行：从 arXiv API 回填最近 120 篇，再按规则筛选；
- 输出：综合 Atom/RSS、方法雷达 RSS、HEP 直接应用 RSS 和静态网页；
- 调度：北京时间约 08:17；
- 托管：GitHub Pages。

## 1. 本地安装

要求 Python 3.11 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Windows PowerShell 激活命令：

```powershell
.venv\Scripts\Activate.ps1
```

## 2. 无密钥离线测试

离线演示不会访问 arXiv，也不会调用模型：

```bash
python -m hepml_digest \
  --demo \
  --state-file build/demo-state.json \
  --output-dir build/demo-public
```

预览：

```bash
python -m http.server 8000 --directory build/demo-public
```

访问 <http://127.0.0.1:8000/>。

运行测试：

```bash
pytest
```

## 3. 真实本地运行（可选）

创建 DeepSeek API Key 后：

```bash
export DEEPSEEK_API_KEY='你的 API Key'
export MAX_CANDIDATES=5
export MAX_DEEP_REVIEWS=1
export SITE_URL='http://127.0.0.1:8000'
python -m hepml_digest
```

第一次建议保留 `5 + 1` 的限制，确认结果后再改回默认值。

程序的关键容错行为：

- 跨分类论文按 `arXiv ID + version` 去重；
- 已筛选版本不会重复调用模型；
- 深度分析失败会保持 `pending`，下一天重试；
- RSS 抓取或模型失败不会主动清空历史 Feed；
- JSON 必须通过 Pydantic 字段和值域校验；
- 状态和发布文件使用原子替换写入。

## 4. GitHub 部署

### 创建仓库并推送

在 GitHub 创建公开空仓库，例如 `hepml-digest`。本项目不会自动执行以下远程操作，需要你确认后手动运行：

```bash
git remote add origin git@github.com:<用户名>/hepml-digest.git
git branch -M main
git push -u origin main
```

### 添加模型密钥

仓库页面进入：

```text
Settings → Secrets and variables → Actions → New repository secret
```

创建名为 `DEEPSEEK_API_KEY` 的 Secret。

### 启用 Pages

这一步必须在第一次运行工作流前手动完成。仓库刚创建时，Pages API 尚不存在，
`configure-pages` 会返回 `Get Pages site failed: Not Found`。

进入：

```text
Settings → Pages → Build and deployment → Source
```

选择 `GitHub Actions`。

然后进入 Actions，手动运行一次 `Daily HEP-ML Digest`。发布地址通常为：

```text
https://<用户名>.github.io/hepml-digest/
https://<用户名>.github.io/hepml-digest/atom.xml
https://<用户名>.github.io/hepml-digest/rss.xml
https://<用户名>.github.io/hepml-digest/methods.xml
https://<用户名>.github.io/hepml-digest/hep-applications.xml
```

## 5. 配置

所有运行配置均可通过环境变量覆盖：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 无 | 必需的 API Key |
| `SCREENING_MODEL` | `deepseek-v4-flash` | 摘要筛选模型 |
| `REVIEW_MODEL` | `deepseek-v4-pro` | 深度分析模型 |
| `ARXIV_CATEGORIES` | 四个默认分类 | 逗号分隔 |
| `MAX_CANDIDATES` | `60` | 每次最多初筛数 |
| `BOOTSTRAP_RESULTS` | `120` | 状态为空时回填的最近论文数 |
| `METHOD_CANDIDATE_SLOTS` | `40` | 通用统计/机器学习方法候选配额 |
| `HEP_APPLICATION_SLOTS` | `10` | HEP 直接应用候选配额 |
| `DISCOVERY_SLOTS` | `10` | 不依赖关键词排名的探索位 |
| `MAX_DEEP_REVIEWS` | `5` | 每次最多深度分析数 |
| `MIN_DEEP_REVIEWS` | `3` | 每日深评目标；只从非 irrelevant 候选补足 |
| `PUBLISH_THRESHOLD` | `0.55` | RSS 发布阈值 |
| `REVIEW_THRESHOLD` | `0.45` | 模型主动推荐进入深评的最低相关性 |
| `FEED_MAX_ITEMS` | `300` | Feed 最大条目数 |
| `FEEDBACK_REPOSITORY` | 空 | GitHub `owner/repo`，启用预填 Issue 反馈入口 |
| `SITE_URL` | 本地地址 | 发布站点根 URL |
| `STATE_FILE` | `data/state.json` | 状态文件 |
| `OUTPUT_DIR` | `public` | 静态输出目录 |

示例见 [.env.example](.env.example)，但不要提交包含真实密钥的 `.env`。

## 6. 调整科研判断标准

主要提示词位于：

- [prompts/screen.txt](prompts/screen.txt)：相关性筛选；
- [prompts/review.txt](prompts/review.txt)：HEP 应用、风险和验证方案。

建议先累计两周人工反馈，再修改阈值或扩展分类。不要仅根据一次日报主观更换模型。

每个网页和 Feed 条目都可以提供“提交反馈”链接。GitHub Actions 会自动把
`FEEDBACK_REPOSITORY` 设置为当前仓库；反馈 Issue 包含保留、映射合理性、
过度推测、栏目调整和全文精读等勾选项，可逐步积累人工评估集。

### 两类内容的边界

- **方法雷达**：来自 `stat.ML`、`cs.LG`、`physics.data-an` 等分类，重点是把通用方法映射到可检验的 HEP 场景；
- **HEP 直接应用**：包含 `hep-ex`、`physics.ins-det` 或 `nucl-ex` 分类，论文已经直接面向实验或探测器问题。

综合 Feed 保留两类内容，也可以只订阅 `methods.xml` 或
`hep-applications.xml`。

## 7. 当前范围

当前版本只分析标题和摘要，尚未自动下载论文 PDF。这是有意设计：先验证论文召回和 HEP 映射准确率，再给每日 Top 3--5 加入全文解析，可以显著降低成本和误报调试难度。

详细设计、国内模型比较和 PDF 正文扩展方案见：

- [国内模型 HEP-ML 日报低成本部署指南](docs/deploy_zh.md)

## 8. 常见问题

### 工作流成功，但 GitHub Pages 没有地址

确认 Pages 的 Source 已设置为 `GitHub Actions`，并检查 deploy job 是否获得 `pages: write` 和 `id-token: write`。

如果日志包含 `Get Pages site failed: Not Found`，说明 Pages 尚未首次启用。进入：

```text
https://github.com/<用户名>/<仓库名>/settings/pages
```

将 `Build and deployment → Source` 设置为 `GitHub Actions`，保存后重新运行失败的工作流。
不建议给工作流配置高权限 PAT 来自动完成这项只需执行一次的设置。

### 日志提示 Node.js 20 deprecated

项目工作流使用 Node 24 版本的官方 actions：`checkout@v6`、
`setup-python@v6`、`configure-pages@v6`、`upload-pages-artifact@v5`
和 `deploy-pages@v5`。
如果仍看到该警告，检查远程仓库是否已经包含最新的工作流提交。

### RSS 可以打开，但没有任何论文条目

arXiv 在周末可能返回空的分类 RSS。项目在状态库为空时会通过 arXiv API
一次性回填最近 120 篇论文；后续无新论文时会继续保留历史条目。
如果首次运行使用的是旧版本工作流，更新代码后重新手动运行一次即可。

### 定时任务没有准点运行

GitHub 调度可能延迟。当前使用 UTC 00:17，约为北京时间 08:17，故意避开整点高峰。

### API 返回空 JSON

DeepSeek 官方文档说明 JSON 模式偶尔可能返回空内容。程序会校验并重试；连续失败的论文不会写入已完成状态。

### 想从 DeepSeek 切换千问

模型调用集中在 `src/hepml_digest/llm.py`。可以新增实现 `Analyzer` 协议的 `QwenAnalyzer`，抓取、状态与发布层无需修改。
