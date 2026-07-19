---
title: "国内模型驱动的 HEP-ML arXiv 日报"
subtitle: "个人低成本部署指南：抓取、筛选、总结与 RSS 发布"
author: "面向粒子实验物理个人研究工作流"
date: "2026-07-19"
lang: zh-CN
documentclass: ctexart
classoption:
  - UTF8
  - 11pt
geometry:
  - a4paper
  - margin=2.2cm
toc: true
toc-depth: 3
numbersections: true
colorlinks: true
linkcolor: blue
urlcolor: blue
header-includes:
  - |
    \usepackage{longtable,booktabs,array,xcolor}
    \usepackage{fvextra}
    \DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,commandchars=\\\{\}}
    \setCJKmainfont{Microsoft YaHei}
    \setCJKsansfont{Microsoft YaHei}
    \setCJKmonofont{LXGW WenKai Mono}
    \setmainfont{TeX Gyre Pagella}
    \setsansfont{TeX Gyre Heros}
    \setmonofont{LXGW WenKai Mono}
    \setlength{\emergencystretch}{3em}
---

# 结论先行

对于个人科研使用，推荐以下组合：

> **DeepSeek-V4-Flash 做批量摘要筛选，DeepSeek-V4-Pro 只分析每日 Top 3--5；GitHub Actions 每天运行，GitHub Pages 免费发布 Atom/RSS。**

这套方案不需要云服务器、不需要 GPU，也不需要 MCP。典型月成本约为 **5--20 元人民币**。第一版只使用标题和摘要；系统运行稳定后，再给 Top 3--5 增加 PDF 正文分析。

推荐理由不是笼统的模型排行榜，而是它与本项目的工程要求匹配：

1. DeepSeek 官方 API 兼容 OpenAI SDK，迁移和维护简单。
2. 支持 JSON 输出，便于自动化校验、重试和生成 RSS。
3. Flash 适合高频分类和摘要；Pro 只承担少量复杂的 HEP 迁移分析。
4. 官方 API 当前单价较低，个人用量不需要购买包月服务。

截至 2026-07-19，DeepSeek 官方标价如下：

| 模型 | 用途 | 未命中缓存输入 | 输出 |
|---|---|---:|---:|
| `deepseek-v4-flash` | 粗筛、分类、短摘要 | 1 元/百万 token | 2 元/百万 token |
| `deepseek-v4-pro` | Top 论文深度分析 | 3 元/百万 token | 6 元/百万 token |

价格与模型名可能变化，部署前应再次查看 [DeepSeek 官方价格页](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)。旧别名 `deepseek-chat` 和 `deepseek-reasoner` 官方计划于北京时间 2026-07-24 23:59 弃用，因此新项目应直接使用 `deepseek-v4-flash` 和 `deepseek-v4-pro`。

# 国内模型如何选择

## 推荐顺序

### 首选：DeepSeek 官方 API

建议配置：

```text
筛选模型：deepseek-v4-flash，关闭思考
深度模型：deepseek-v4-pro，开启思考
接口地址：https://api.deepseek.com
```

它最适合当前项目的原因是 API 简洁、价格低、JSON 输出可用。需要注意，JSON Output 保证的是“有效 JSON”，不是业务字段一定正确，所以程序仍应使用 Pydantic 校验字段、枚举和值域。官方 JSON 文档也提示，偶尔可能返回空内容，应实现重试。

参考：

- [DeepSeek 首次调用 API](https://api-docs.deepseek.com/zh-cn/)
- [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/)
- [DeepSeek 对话补全参数](https://api-docs.deepseek.com/zh-cn/api/create-chat-completion/)

### 备选：千问 API

如果已经使用阿里云，或者后续希望把定时任务、对象存储和模型统一放在一个云账号，可以选择千问 Flash/Plus：

```text
筛选：Qwen Flash 类模型
深度分析：Qwen Plus 类模型
```

千问也提供 OpenAI 兼容调用。官方文档说明，部分思考模型不支持直接使用 `response_format`；此时应关闭思考完成结构化筛选，或再让 Flash 模型修复 JSON。千问 Batch 当前按实时价格的 50% 计费，但日处理规模很小时没有必要先增加异步批处理复杂度。

参考：

- [千问模型价格](https://platform.qianwenai.com/docs/developer-guides/getting-started/pricing)
- [千问结构化输出常见问题](https://platform.qianwenai.com/docs/resources/faq-text-generation)
- [千问批量调用](https://platform.qianwenai.com/docs/developer-guides/text-generation/batch)

### 什么时候考虑 GLM 或 MiniMax

如果已有相应平台余额、单位合同或现成 API 网关，可以将其作为备选供应商。但对这个日报项目，模型的首要指标是：

1. JSON 输出稳定性；
2. 长摘要和学术英文理解；
3. 中文总结质量；
4. API 延迟与限流；
5. 单次失败后的可重试性；
6. 实际 `precision@10`，而不是通用排行榜。

因此不要同时接入四家模型。先用一个主供应商跑两周，并用约 100 篇人工标注论文做评估，再决定是否加入备用供应商。

## 是否应该本地部署开源模型

若已有 16--24 GB 显存 GPU 或长期运行的工作站，可用 Qwen/DeepSeek 蒸馏模型做第一级筛选。但不建议为了这个项目专门购买显卡：

- API 月费通常只有几元到几十元；
- 本地推理还有电费、驱动、模型更新和宕机维护；
- 小型量化模型在细致的 HEP 方法迁移分析上更容易产生似是而非的结论。

较合理的混合方式是：本地模型做“是否值得送审”的二分类，官方 API 只分析 Top 3--5。

# 预算估算

假设每天抓取 200 篇，关键词规则保留 60 篇，Flash 分析这 60 个摘要，Pro 分析其中 5 篇：

| 阶段 | 月输入量示例 | 月输出量示例 | 估算 |
|---|---:|---:|---:|
| Flash 摘要筛选 | 120 万 token | 30 万 token | 约 1.8 元 |
| Pro Top 5 摘要/正文节选 | 120 万 token | 15 万 token | 约 4.5 元 |
| 合计 |  |  | 约 6--10 元/月 |

如果每天把 20 篇完整 PDF 全部发送给模型，成本和失败率都会明显上升。推荐设置硬上限：

```text
MAX_CANDIDATES=60
MAX_DEEP_REVIEWS=5
MAX_FULLTEXT_CHARS=40000
MAX_RETRIES=3
```

# 最小可用系统

## 系统流程

```text
arXiv 分类 RSS
       |
       v
提取标题、摘要、作者、arXiv ID
       |
       v
本地状态去重 + 关键词召回
       |
       v
V4-Flash 输出结构化相关性评分
       |
       v
相关性最高的 3--5 篇
       |
       v
V4-Pro 生成 HEP 应用分析
       |
       v
生成 public/atom.xml、rss.xml、index.html
       |
       v
GitHub Pages 发布
```

第一版不必使用 embedding、向量数据库、LangChain、Agent 或 MCP。它们不会自动提高报告质量，却会增加调试面。

## 输出应区分事实和迁移设想

深度报告必须明确区分：

- `paper_claims`：论文摘要或正文明确声称的结果；
- `hep_opportunities`：模型提出的 HEP 迁移方向；
- `transfer_risks`：迁移时可能失效的假设；
- `evidence_level`：`direct`、`transferable` 或 `speculative`；
- `validation_plan`：最小可行验证实验。

特别检查以下 HEP 问题：负权事件、系统误差、nuisance parameter、simulation-to-data shift、校准、触发延迟、探测器几何和 Lorentz/置换对称性。

# 部署前准备

## 注册和准备

需要：

1. 一个 GitHub 账号；
2. 一个公开 GitHub 仓库，例如 `hepml-digest`；
3. 一个 DeepSeek 开放平台 API Key；
4. 给 API 账号充值少量余额，例如 10--20 元；
5. 本地安装 Git 和 Python 3.11 或更高版本，用于首次测试。

如果日报或研究兴趣不方便公开，可用私人仓库运行 Actions，但 GitHub Free 的私人仓库不能免费使用 Pages。此时可把 `public/` 同步到对象存储，或在已有 NAS/服务器上用 Nginx 发布。

## 创建目录

```bash
mkdir hepml-digest
cd hepml-digest
git init
mkdir -p src prompts data public .github/workflows
touch data/seen.json
```

`data/seen.json` 初始内容为：

```json
{}
```

建议目录结构：

```text
hepml-digest/
├── src/
│   ├── fetch.py
│   ├── models.py
│   ├── llm.py
│   ├── pipeline.py
│   └── publish.py
├── prompts/
│   ├── screen.txt
│   └── review.txt
├── data/
│   └── seen.json
├── public/
├── requirements.txt
└── .github/workflows/daily.yml
```

# 项目配置

## Python 依赖

`requirements.txt`：

```text
feedparser>=6.0,<7
feedgen>=1.0,<2
openai>=1.75,<3
pydantic>=2.10,<3
tenacity>=9,<10
```

本地安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 环境变量

本地测试时：

```bash
export DEEPSEEK_API_KEY='你的密钥'
export SCREENING_MODEL='deepseek-v4-flash'
export REVIEW_MODEL='deepseek-v4-pro'
```

密钥不要写进 `.env` 后提交，也不要打印到日志。

## 筛选提示词

`prompts/screen.txt`：

```text
你是一名粒子实验物理与统计学习交叉领域研究员。
根据论文标题和摘要，判断其方法对高能实验物理是否值得进一步阅读。

只输出 JSON，不要输出 Markdown。字段必须为：
relevance: 0 到 1 的数；
method: 论文的主要统计/机器学习方法；
hep_tasks: 字符串数组；
reason: 不超过 80 个中文字符；
needs_deep_review: 布尔值；
evidence_level: direct、transferable、speculative 或 irrelevant。

不要仅凭出现 neural network 就判为相关。重点考虑重建、粒子鉴别、
触发、异常检测、快速模拟、展开、无似然推断、domain shift、
不确定度量化、校准以及系统误差处理。
```

## 深度分析提示词

`prompts/review.txt`：

```text
你是一名谨慎的粒子实验物理学家。请基于提供的标题和摘要分析论文，
而不是编造论文没有报告的实验结果。

只输出 JSON，字段必须为：
summary_cn: 方法摘要，150--250 个中文字符；
paper_claims: 论文摘要明确支持的结论数组；
hep_opportunities: 可能的 HEP 应用数组；
transfer_risks: 迁移风险数组；
validation_plan: 最小可行验证方案；
evidence_level: direct、transferable 或 speculative；
confidence: 0 到 1 的数。

请检查系统误差、负权事件、simulation-to-data shift、校准、
nuisance parameter、探测器几何、推理延迟等问题。
把 HEP 应用明确标记为迁移设想，不要写成论文已经验证的事实。
```

# 核心代码实现要点

## 数据模型与校验

`src/models.py` 至少定义：

```python
from typing import Literal
from pydantic import BaseModel, Field


Evidence = Literal[
    "direct", "transferable", "speculative", "irrelevant"
]


class Screening(BaseModel):
    relevance: float = Field(ge=0, le=1)
    method: str
    hep_tasks: list[str]
    reason: str
    needs_deep_review: bool
    evidence_level: Evidence


class Review(BaseModel):
    summary_cn: str
    paper_claims: list[str]
    hep_opportunities: list[str]
    transfer_risks: list[str]
    validation_plan: str
    evidence_level: Literal[
        "direct", "transferable", "speculative"
    ]
    confidence: float = Field(ge=0, le=1)
```

不能因为响应是合法 JSON 就直接信任。必须执行：

```python
result = Screening.model_validate_json(response_text)
```

如果校验失败，重试最多三次；仍失败则把论文记为 `failed`，下一次任务再处理。

## DeepSeek 调用封装

`src/llm.py` 的核心形式：

```python
import os
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def request_json(model: str, system: str, user: str,
                 thinking: bool = False) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        max_tokens=1800,
        stream=False,
        extra_body={
            "thinking": {
                "type": "enabled" if thinking else "disabled"
            }
        },
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("模型返回空内容")
    return content
```

筛选时 `thinking=False`；深度分析时 `thinking=True`。筛选任务开启思考通常只会增加输出 token 和费用。

## RSS 抓取

建议先抓取：

```python
FEEDS = [
    "https://rss.arxiv.org/rss/stat.ML",
    "https://rss.arxiv.org/rss/cs.LG",
    "https://rss.arxiv.org/rss/physics.data-an",
    "https://rss.arxiv.org/rss/hep-ex",
]
```

抓取时必须：

1. 设置明确 User-Agent；
2. 用 arXiv 基础 ID 去重跨分类论文；
3. 记录首次看到的时间，不用 `published == today`；
4. RSS 失败时不要清空旧报告；
5. 每次最多处理固定数量论文。

可从链接提取基础 ID：

```python
import re


def arxiv_id(url: str) -> str:
    match = re.search(r"/(?:abs|pdf)/(\d{4}\.\d{4,5})", url)
    if not match:
        raise ValueError(f"无法识别 arXiv ID: {url}")
    return match.group(1)
```

论文版本更新时，可以另存 `version`，但 RSS 条目的稳定 GUID 应继续使用基础 ID，避免阅读器重复显示同一篇论文。

## 规则召回

在调用 API 前，先用本地规则压缩候选数：

```python
TERMS = {
    "simulation-based inference": 5,
    "likelihood-free": 5,
    "uncertainty quantification": 4,
    "anomaly detection": 4,
    "domain adaptation": 4,
    "equivariant": 3,
    "calibration": 3,
    "density estimation": 3,
    "optimal transport": 2,
    "graph neural network": 2,
    "generative model": 2,
}


def keyword_score(title: str, abstract: str) -> int:
    text = f"{title} {abstract}".lower()
    return sum(weight for term, weight in TERMS.items()
               if term in text)
```

不要要求分数必须大于零。每天仍应随机保留约 5 篇零分论文，防止关键词体系让系统永远发现不了新方法。

## 两阶段调度

核心逻辑：

```python
candidates = fetch_new_papers()
candidates = deduplicate(candidates)
candidates = local_prefilter(candidates, limit=60)

screened = []
for paper in candidates:
    result = screen_with_flash(paper)
    screened.append((paper, result))

top = sorted(
    screened,
    key=lambda pair: pair[1].relevance,
    reverse=True,
)[:5]

reviews = [review_with_pro(paper) for paper, _ in top]
publish_feeds(reviews)
save_state(candidates)
```

只有 API 调用和最终文件全部成功后，才更新 `seen.json`。否则一次中断可能把未完成论文错误标记为已处理。

# 发布 Atom/RSS

每篇论文条目至少包含：

```text
id/guid：arxiv:<基础 ID>:hepml-v1
title：论文标题
link：arXiv abstract 页面
published：arXiv 时间
updated：本次分析时间
content：中文方法摘要、HEP 应用、风险和验证建议
category：HEP task 和 evidence level
```

保留最近 90 天或最近 300 条记录。不要每天完全覆盖成只有当天内容，否则错过一次刷新后，RSS 阅读器可能无法补回条目。

推荐同时生成：

```text
public/atom.xml    主订阅地址
public/rss.xml     兼容旧阅读器
public/index.html  浏览器阅读页面
```

# GitHub Actions 自动部署

## 添加 API Key

进入仓库：

```text
Settings
→ Secrets and variables
→ Actions
→ New repository secret
```

创建：

```text
DEEPSEEK_API_KEY
```

GitHub Secrets 会加密保存密钥，并只在工作流显式引用时注入。不要在调试日志中输出环境变量。参考 [GitHub Actions Secrets](https://docs.github.com/en/actions/concepts/security/secrets)。

## 工作流

`.github/workflows/daily.yml`：

```yaml
name: Daily HEP-ML Digest

on:
  schedule:
    - cron: "17 0 * * *"
  workflow_dispatch:

concurrency:
  group: daily-hepml-digest
  cancel-in-progress: false

permissions:
  contents: write
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - run: pip install -r requirements.txt

      - name: Build digest
        env:
          DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
          SCREENING_MODEL: deepseek-v4-flash
          REVIEW_MODEL: deepseek-v4-pro
        run: python -m src.pipeline

      - name: Persist state and reports
        run: |
          git config user.name "hepml-digest-bot"
          git config user.email "hepml-digest-bot@users.noreply.github.com"
          git add data public
          git diff --cached --quiet || git commit -m "daily digest"
          git push

      - uses: actions/configure-pages@v5

      - uses: actions/upload-pages-artifact@v4
        with:
          path: public

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

这里选择 UTC 00:17，即北京时间 08:17。避开整点可以减少 GitHub Actions 高峰期排队。GitHub 官方说明定时任务默认以 UTC 运行，并可能因高负载延迟；定时工作流只运行默认分支上的版本。参考 [GitHub Actions 定时触发](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)。

GitHub Pages 的自定义工作流应上传 Pages artifact，再由 `deploy-pages` 部署，而不是假设仓库中的任意 `public/` 目录会自动发布。参考 [GitHub Pages 自定义工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。

## 启用 Pages

进入：

```text
Settings → Pages → Build and deployment → Source
```

选择：

```text
GitHub Actions
```

之后在 Actions 页面手动执行一次 `workflow_dispatch`。成功后订阅地址通常是：

```text
https://<用户名>.github.io/hepml-digest/atom.xml
https://<用户名>.github.io/hepml-digest/rss.xml
```

# 本地验收

## 首次运行

```bash
export DEEPSEEK_API_KEY='你的密钥'
python -m src.pipeline
```

检查：

```bash
python -c "import json; json.load(open('data/seen.json'))"
python -c "import xml.etree.ElementTree as E; E.parse('public/atom.xml')"
python -c "import xml.etree.ElementTree as E; E.parse('public/rss.xml')"
```

然后运行一个本地静态服务器：

```bash
python -m http.server 8000 --directory public
```

浏览器访问 `http://127.0.0.1:8000/`。

## 上线验收清单

- [ ] 相同论文跨 `stat.ML` 和 `cs.LG` 只出现一次；
- [ ] 第二次运行不会重复调用已处理论文；
- [ ] JSON 校验失败会重试但不死循环；
- [ ] 单篇失败不会清空旧 RSS；
- [ ] Atom/RSS XML 可以解析；
- [ ] API Key 不出现在代码和 Actions 日志；
- [ ] 报告区分论文原始结论与 HEP 迁移设想；
- [ ] 每次运行候选数和深度分析数有硬上限；
- [ ] 手动触发和定时触发都能部署 Pages；
- [ ] RSS 阅读器中 GUID 稳定，不重复刷屏。

# 第二阶段：增加 PDF 正文分析

运行两周并完成至少 100 篇人工反馈后，再增加 PDF：

1. 只下载 Top 3--5；
2. 限制 PDF 大小，例如 20 MB；
3. 优先抽取 abstract、introduction、method、experiments、limitations；
4. 删除参考文献，避免浪费 token；
5. 对扫描版或解析失败的 PDF 回退到摘要；
6. 不把 PDF 提交进 Git 仓库；
7. 临时文件在任务结束时删除；
8. 在报告中记录 `source=abstract` 或 `source=fulltext`。

可选依赖：

```text
pymupdf>=1.25,<2
```

伪代码：

```python
def extract_selected_text(pdf_path) -> str:
    text = extract_pdf_text(pdf_path)
    sections = select_sections(
        text,
        names=[
            "abstract", "introduction", "method",
            "experiments", "limitations", "conclusion",
        ],
    )
    return sections[:40_000]
```

40,000 字符只是费用与上下文控制阈值，不代表字符数等于 token 数。实际费用应以 API 响应中的 `usage` 字段累计。

# 成本和安全保护

## 程序内预算闸门

每次运行记录：

```text
候选论文数
Flash 调用数
Pro 调用数
prompt tokens
completion tokens
失败与重试次数
估算人民币费用
```

达到以下任一条件立即停止新增 API 调用，但继续发布已有结果：

```text
候选数超过 60
深度分析数超过 5
单次运行总输入超过 300,000 token
连续 API 失败超过 5 次
```

## 故障处理

| 故障 | 处理 |
|---|---|
| HTTP 401 | 检查 Secret 名称与 Key |
| HTTP 402 | 余额不足，停止调用但保留旧 RSS |
| HTTP 429 | 指数退避，不要立即并发重试 |
| HTTP 500/503 | 最多重试 3 次，记录失败论文 |
| 空 JSON | 重试并缩短提示词 |
| JSON 字段错误 | Pydantic 拒绝，重试或标记失败 |
| arXiv RSS 暂时不可用 | 发布旧 Feed，不覆盖为空文件 |
| Git push 冲突 | 使用 concurrency，避免两个任务并行 |

DeepSeek 官方错误码说明可见 [API 错误码](https://api-docs.deepseek.com/zh-cn/quick_start/error_codes/)。

# 质量评估

低成本不应以牺牲科研可靠性为代价。建议在第一个月人工标注 100 篇论文：

```text
relevant: 是否值得粒子实验物理学家阅读
hep_task: 对应哪个 HEP 任务
mapping_valid: HEP 映射是否合理
unsupported_claim: 是否把推测写成论文事实
priority: 1--5
```

重点统计：

- `precision@5`：日报前五篇真正有用的比例；
- 重要论文漏报数；
- 不受原文支持的结论比例；
- 每天人工阅读时间；
- 单篇有效推荐的平均成本。

只有评估显示摘要不足时，才扩大 PDF 正文读取范围。只有单模型失败率确实影响使用时，才增加第二供应商。

# 国内网络环境下的替代部署

如果 GitHub Pages 访问不稳定，有两种保持低成本的替代方案。

## 已有 NAS 或常开工作站

使用系统 cron 每天运行：

```cron
17 8 * * * cd /opt/hepml-digest && .venv/bin/python -m src.pipeline
```

用 Nginx/Caddy 只读发布 `public/`。这是最便宜的国内访问方案，但要自行负责公网入口、HTTPS 和安全更新。

## 国内对象存储静态托管

Actions 生成文件后，将 `public/` 同步到阿里云 OSS、腾讯云 COS 等对象存储。RSS/XML/HTML 数据量很小，存储和流量费用通常很低。具体价格和公网域名/HTTPS政策变化较快，应按所选云厂商当前文档配置，不建议在代码里绑定某一家 SDK；可把发布层封装成独立命令。

# 最终推荐配置

个人第一版直接采用：

```text
来源：stat.ML + cs.LG + physics.data-an + hep-ex
抓取：arXiv RSS
规则召回：最多 60 篇/日
摘要筛选：deepseek-v4-flash，关闭思考
深度分析：deepseek-v4-pro，开启思考，最多 5 篇/日
状态：JSON，提交到 Git
输出：Atom + RSS + HTML
调度：GitHub Actions，北京时间约 08:17
托管：GitHub Pages
预算：先充值 10--20 元，并设置程序硬上限
```

运行两周后再决定：

1. 是否增加 PDF 正文；
2. 是否加入 INSPIRE-HEP 元数据；
3. 是否加入邮件；
4. 是否加入 embedding；
5. 是否封装 MCP 供交互式查询。

这个顺序可以让绝大多数工程风险在几元成本内暴露出来，并保留清晰的可复现数据链。

# 参考资料

1. DeepSeek，模型与价格：<https://api-docs.deepseek.com/zh-cn/quick_start/pricing>
2. DeepSeek，首次调用 API：<https://api-docs.deepseek.com/zh-cn/>
3. DeepSeek，JSON Output：<https://api-docs.deepseek.com/guides/json_mode/>
4. DeepSeek，对话补全：<https://api-docs.deepseek.com/zh-cn/api/create-chat-completion/>
5. DeepSeek，错误码：<https://api-docs.deepseek.com/zh-cn/quick_start/error_codes/>
6. 千问，模型价格：<https://platform.qianwenai.com/docs/developer-guides/getting-started/pricing>
7. 千问，文本生成 FAQ：<https://platform.qianwenai.com/docs/resources/faq-text-generation>
8. 千问，批量调用：<https://platform.qianwenai.com/docs/developer-guides/text-generation/batch>
9. GitHub，Actions Secrets：<https://docs.github.com/en/actions/concepts/security/secrets>
10. GitHub，定时工作流：<https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows>
11. GitHub，Pages 自定义工作流：<https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages>

> 本文的价格、模型名称和云平台限制截至 2026-07-19。模型服务更新频繁，实际部署时应再次核对官方价格和弃用公告。
