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
- 每日最多初筛 60 篇、深度分析 5 篇；
- 状态：`data/state.json`；
- 输出：`public/atom.xml`、`public/rss.xml`、`public/index.html`；
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
| `DISCOVERY_SLOTS` | `5` | 无关键词候选探索位 |
| `MAX_DEEP_REVIEWS` | `5` | 每次最多深度分析数 |
| `PUBLISH_THRESHOLD` | `0.55` | RSS 发布阈值 |
| `REVIEW_THRESHOLD` | `0.72` | 深度分析阈值 |
| `FEED_MAX_ITEMS` | `300` | Feed 最大条目数 |
| `SITE_URL` | 本地地址 | 发布站点根 URL |
| `STATE_FILE` | `data/state.json` | 状态文件 |
| `OUTPUT_DIR` | `public` | 静态输出目录 |

示例见 [.env.example](.env.example)，但不要提交包含真实密钥的 `.env`。

## 6. 调整科研判断标准

主要提示词位于：

- [prompts/screen.txt](prompts/screen.txt)：相关性筛选；
- [prompts/review.txt](prompts/review.txt)：HEP 应用、风险和验证方案。

建议先累计两周人工反馈，再修改阈值或扩展分类。不要仅根据一次日报主观更换模型。

## 7. 当前范围

当前版本只分析标题和摘要，尚未自动下载论文 PDF。这是有意设计：先验证论文召回和 HEP 映射准确率，再给每日 Top 3--5 加入全文解析，可以显著降低成本和误报调试难度。

详细设计、国内模型比较和 PDF 正文扩展方案见：

- [国内模型 HEP-ML 日报低成本部署指南](docs/deploy_zh.md)

## 8. 常见问题

### 工作流成功，但 GitHub Pages 没有地址

确认 Pages 的 Source 已设置为 `GitHub Actions`，并检查 deploy job 是否获得 `pages: write` 和 `id-token: write`。

### 定时任务没有准点运行

GitHub 调度可能延迟。当前使用 UTC 00:17，约为北京时间 08:17，故意避开整点高峰。

### API 返回空 JSON

DeepSeek 官方文档说明 JSON 模式偶尔可能返回空内容。程序会校验并重试；连续失败的论文不会写入已完成状态。

### 想从 DeepSeek 切换千问

模型调用集中在 `src/hepml_digest/llm.py`。可以新增实现 `Analyzer` 协议的 `QwenAnalyzer`，抓取、状态与发布层无需修改。
