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
- 深度分析：`deepseek-v4-pro`，低强度思考，单次输出上限 8192 tokens；
- 每日候选配额：方法雷达 40 篇、HEP 直接应用 10 篇、探索位 10 篇；
- 每日最多深度分析 5 篇，只处理模型推荐且同时达到深评和发布阈值的候选，不设最低数量；
- 状态：`data/state.json`；
- 首次运行：从 arXiv API 回填最近 120 篇；后续按更新日期分页补抓；
- 候选队列：原始论文先写入状态，超额或初筛失败的候选留待后续处理；
- 输出：综合 Atom/RSS、方法雷达 RSS、HEP 直接应用 RSS 和静态网页；
- 调度：北京时间约 08:17；
- 托管：GitHub Pages。

## 1. 本地安装

要求 Python 3.11 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps -e .
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
- 已筛选版本不会重复调用模型，模型或 prompt 改变时会按日配额重算；
- 深度分析失败会保持 `pending`，下一天重试；
- 超额候选和初筛失败候选会保留，离开 RSS 窗口后仍可继续处理；
- 分页补抓中断会保留已抓页面，不推进补抓游标；
- 初筛达到总模型调用时间或输入 token 预算的 70% 后停止新增请求，为深评留出余量；
- RSS 抓取或模型失败不会主动清空历史 Feed；
- JSON 必须通过 Pydantic 字段和值域校验；
- 状态读改写使用进程锁，并按配置间隔 checkpoint；
- 所有发布内容完成序列化后才开始原子替换文件。

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
| `DEEPSEEK_API_KEY` | 无 | 必需；使用中转站时填写中转站的 Key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容接口的根地址 |
| `ANALYSIS_REVISION` | `1` | 手动分析版本号，模型别名对应服务改变时用于触发重算 |
| `SCREENING_MODEL` | `deepseek-v4-flash` | 摘要筛选模型 |
| `REVIEW_MODEL` | `deepseek-v4-pro` | 深度分析模型 |
| `REVIEW_MAX_TOKENS` | `8192` | 单次深评的输出上限，包含思考与最终回答 |
| `REVIEW_REASONING_EFFORT` | `low` | 深评思考强度，可选 `low`、`high`、`max` |
| `ARXIV_CATEGORIES` | 四个默认分类 | 逗号分隔 |
| `MAX_CANDIDATES` | `60` | 每次最多初筛数 |
| `BOOTSTRAP_RESULTS` | `120` | 状态为空时回填的最近论文数 |
| `BACKFILL_LOOKBACK_DAYS` | `7` | 从上次完整补抓时间向前重叠扫描的天数 |
| `BACKFILL_MAX_PAGES` | `25` | 每次补抓页数上限，每页最多 200 篇 |
| `METHOD_CANDIDATE_SLOTS` | `40` | 通用统计/机器学习方法候选配额 |
| `HEP_APPLICATION_SLOTS` | `10` | HEP 直接应用候选配额 |
| `DISCOVERY_SLOTS` | `10` | 不依赖关键词排名的探索位 |
| `MAX_DEEP_REVIEWS` | `5` | 每次最多深度分析数 |
| `PUBLISH_THRESHOLD` | `0.55` | RSS 发布阈值 |
| `REVIEW_THRESHOLD` | `0.45` | 深评相关性阈值；同时还需达到 `PUBLISH_THRESHOLD` |
| `FEED_MAX_ITEMS` | `300` | Feed 最大条目数 |
| `REQUEST_TIMEOUT_SECONDS` | `120` | 模型请求超时秒数；本地、分析器与工作流默认一致，工作流支持同名 Variable 覆盖 |
| `API_TIME_BUDGET_SECONDS` | `1440` | 达到后停止新增 API 调用并发布已有结果 |
| `SCREENING_BUDGET_FRACTION` | `0.7` | 初筛时间及输入 token 预算占比；深评关闭时不限制 |
| `MAX_CONSECUTIVE_API_FAILURES` | `5` | 连续失败达到此数后停止新增 API 调用 |
| `MAX_PROMPT_TOKENS` | `300000` | 单次运行输入 token 预算 |
| `REANALYSIS_SLOTS` | `5` | 模型或 prompt 变化后每日保留的历史重算名额 |
| `CHECKPOINT_INTERVAL` | `5` | 每完成多少次分析将状态增量写盘 |
| `AGING_SLOTS` | `5` | 每个候选排序池优先处理旧任务的名额上限；同时不超过配额的约 20%（向上取整）；`0` 关闭 |
| `MAX_PAPER_FAILURES` | `3` | 同一任务连续因响应格式失败多少轮后隔离；一轮含内部重试 |
| `ALERT_AFTER_RUNS` | `3` | 来源失败、分析失败或队列增长的连续运行告警阈值 |
| `ALERT_BACKLOG_DAYS` | `7` | 最老任务等待天数的告警阈值 |
| `MAX_ESTIMATED_COST_RMB` | `1` | 本次运行估算费用停止线，`0` 禁止模型调用；不是账单硬上限 |
| `SCREENING_CACHE_HIT_PRICE_PER_MILLION_RMB` | `0.02` | 初筛缓存命中输入价格 |
| `SCREENING_INPUT_PRICE_PER_MILLION_RMB` | `1` | 初筛缓存未命中输入价格 |
| `SCREENING_OUTPUT_PRICE_PER_MILLION_RMB` | `4` | 初筛输出价格，用于费用估算 |
| `REVIEW_CACHE_HIT_PRICE_PER_MILLION_RMB` | `0.15` | 深评缓存命中输入价格 |
| `REVIEW_INPUT_PRICE_PER_MILLION_RMB` | `4.5` | 深评缓存未命中输入价格 |
| `REVIEW_OUTPUT_PRICE_PER_MILLION_RMB` | `13.5` | 深评输出价格，用于费用估算 |
| `FEEDBACK_REPOSITORY` | 空 | GitHub `owner/repo`，启用预填 Issue 反馈入口 |
| `SITE_URL` | 本地地址 | 发布站点根 URL |
| `STATE_FILE` | `data/state.json` | 状态文件 |
| `OUTPUT_DIR` | `public` | 静态输出目录 |

示例见 [.env.example](.env.example)，但不要提交包含真实密钥的 `.env`。
程序不会自动读取 `.env`；需要在 shell 中导出环境变量，或通过 GitHub Actions 注入。

深评和历史待评任务统一要求模型的 `needs_deep_review=true`、相关性至少为
`max(REVIEW_THRESHOLD, PUBLISH_THRESHOLD)`，且证据等级不是 `irrelevant`。
旧队列中不符合这些条件的待评记录会改为 `not_selected` 并清除对应重试元数据，论文记录保留。

模型调用次数、输入/输出 token 和估算费用会写入命令输出；在 GitHub
Actions 中还会写入对应 run 的 Step Summary。价格可能变化，更新价格配置后
再使用费用估算结果。

### 切换中转站

本地运行时设置：

```bash
export DEEPSEEK_BASE_URL='https://relay.example/v1'
export DEEPSEEK_API_KEY='替换为中转站的 Key'
export SCREENING_MODEL='中转站提供的初筛模型名'
export REVIEW_MODEL='中转站提供的深评模型名'
```

地址按中转站文档填写，通常包含 `/v1`，不要填写完整的
`/chat/completions` 路径。初筛和深评共用这一个地址与 Key。
当前请求使用 Chat Completions、`response_format: json_object` 和 DeepSeek 的
`thinking` 参数；中转站必须支持或正确转发这些字段，不能仅凭“OpenAI 兼容”
就认定任意模型都能直接替换。

GitHub 部署时，在 `Settings → Secrets and variables → Actions` 中：

1. 在 **Secrets** 中更新 `DEEPSEEK_API_KEY`，填入中转站的 Key；
2. 在 **Variables** 中添加 `DEEPSEEK_BASE_URL`、`SCREENING_MODEL` 和 `REVIEW_MODEL`；
3. 按中转站的实际人民币单价设置上表六项价格变量，避免沿用官方价格估算。

未设置这些 Variables 时，工作流仍使用官方地址和默认模型。
移除 `DEEPSEEK_BASE_URL` 并恢复官方 Key、模型名和价格变量，即可切回官方 API。
只使用可信中转站：论文摘要及提示词会发送给该服务，不要混入未公开或受限材料。

分析指纹包含去除凭据的接口地址、模型名、提示词、响应 schema、思考模式及强度、输出长度及
请求格式版本，不包含 API Key。切换地址会触发重算，单纯轮换 Key 不会。
若中转站在同一模型名下更换实际模型，可调高 `ANALYSIS_REVISION`；GitHub 上通过同名
Repository Variable 设置。新的指纹规则和缺少指纹的旧记录会触发重算，仍受候选、深评和
费用预算约束。`REANALYSIS_SLOTS` 是预留名额，不是重算数量硬上限；空余候选名额也可能
用于历史重算，切换服务前应确认费用配置。

### 更新时间与费用

截至 2026-10-29，[DeepSeek 官方价格页](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)
规定：北京时间周一至周五（不含中国法定节假日）09:00–12:00、14:00–18:00
为高峰时段，其余时间为空闲时段；空闲单价为高峰单价的一半。
默认估价采用官方空闲单价，不会根据实际调用时间自动切换。

工作流使用 UTC `17 0 * * *`，即北京时间每天约 08:17 触发，发布会晚于触发时间。
08:17 已属于空闲时段，因此无需为了官方优惠改到晚上。
GitHub 调度可能延迟；中转站是否有分时优惠，以中转站自己的计费规则为准。
官方目前推荐 Flash 模型名 `deepseek-flash`；本项目保留的
`deepseek-v4-flash` 别名仍可调用，但实际由当前 Flash 模型提供服务。

### 补抓、队列和预算

原始候选保存于 `data/state.json` 的 `pending_papers`，不会因为超过每日配额、
初筛失败或离开 RSS 窗口而丢弃；初筛成功后才从队列移除。旧状态文件会自动升级，
无需清空历史记录。队列不自动过期，若 `queued_papers` 持续增长，应缩小分类范围或
适当提高每日配额，而不是把“已保存”视为“已处理”。

`work` 保存任务首次入队时间、失败次数及下次重试时间。初筛、深评都为较早入队的任务
保留少量名额，避免仅按关键词或相关性排序导致长期等待。失败后按 1、2、4、8、16、24
小时退避，之后最多每天重试；响应格式反复失败达到 `MAX_PAPER_FAILURES` 轮后隔离，
论文仍保留。临时网络错误和限流不会把论文隔离；401/403、400/404 分别按认证或配置
故障处理，当次停止模型调用，不增加单篇论文的失败次数。

修复模型或提示词后，指纹变化会自动重置对应任务的冷却和隔离状态。只想重试单个隔离任务时，
应先停止运行，再仅删除该任务的 `work` 条目；不要删除原始候选或整个状态库。

首次回填以 `BOOTSTRAP_RESULTS` 为限；后续通过 arXiv API 按最后更新日期倒序分页扫描，
覆盖停机期间，并向上次完整补抓时间之前重叠 7 天，以缓解周末发布和索引延迟。
这也能补抓旧论文的新版本，而不仅是新提交。每页先落盘，只有完整扫描窗口后才推进
`last_backfill_at`。达到页数上限或解析、网络失败时保留旧游标，并报告 `fetch_errors`；
未完成窗口的起点也会保存，避免尚无成功游标时反复失败导致窗口向前滑动。
达到上限时需提高 `BACKFILL_MAX_PAGES`，否则可能反复扫描相同前页。
超过重叠窗口的索引延迟仍可能漏抓，不能据此保证来源召回率为 100%。

初筛和深评仍共享总预算，但初筛默认在使用约 70% 的时间或输入 token 后停止，
剩余额度供深评使用；设 `SCREENING_BUDGET_FRACTION=1` 可取消预留。
预算在每次请求（含内部重试）之前检查，单次在途请求可能越过界限，因此不是严格费用上限。
关闭深评时，初筛可使用全部预算。抓取时间不计入模型调用时间预算，但仍受工作流总超时约束。
`queued_papers`、`pending_reviews`、`fetch_errors` 和 `screening_budget_exhausted`
都会写入运行结果及 GitHub Actions Summary。

深评默认采用低强度思考和 8192 tokens 输出预算，避免较小的预算在思考阶段耗尽。
可通过 `REVIEW_MAX_TOKENS`、`REVIEW_REASONING_EFFORT` 调整，GitHub 工作流支持同名 Variables。
思考消耗也计入输出费用；提高单次输出上限会增加单次最大开销，金额停止线不会随之提高。
例如 0.1 元的软停止线仍可能被一次尚在进行的深评请求越过。

运行结果记录两阶段的 `*_finish_reason` 以及 `reasoning_tokens`；不保存原始思考文本。
若接口明确返回 `finish_reason=length`，报告 `output_limit` 并停止调用，不按相同参数盲目
重试，也不将论文隔离。应先检查输出预算或思考强度，再决定是否复测。

每次请求及重试前也检查估算金额。输入（区分缓存）和输出费用累计达到
`MAX_ESTIMATED_COST_RMB` 后停止后续请求；费用计算与最终报告共用同一实现。
如果接口没有返回完整的输入/输出用量，程序报告 `unmetered_usage` 并停止后续调用，
不会把未知用量当成零费用继续执行。单次在途请求、未返回用量的失败请求和价格配置误差
仍可能使账单越过停止线；严格消费限制应在服务商后台设置。

### 运行健康状态

状态库的 `health` 记录连续失败、队列增长次数及最近一次来源和分析成功时间。
结果包含 `status`、`stop_reason`、`alerts`、`quarantined_tasks` 和 `oldest_wait_days`。
空闲日没有新论文不算失败；因冷却而没有分析尝试的运行不会额外累计分析失败次数。
只要有初筛或深评失败，当次即标记 `degraded` 并告警；其他阶段成功不会掩盖失败，
连续分析失败计数也会累计含部分失败的运行。
补抓未完成、任务隔离、长期等待或达到连续运行阈值时，会输出告警并写入 Actions Summary
及工作流注释。默认不会发送邮件或创建 Issue。

认证、配置错误或输出截断会在状态保存和诊断输出后以非零状态退出；现有工作流会持久化状态并标记失败，
不覆盖线上页面。其他退化情况继续发布可用结果，但通过 `degraded` 状态及告警明确标识。

生产和 CI 使用带哈希的 `requirements.lock` 锁定运行时依赖；构建后端版本固定在
`pyproject.toml`。更新依赖后运行：

```bash
uv pip compile pyproject.toml --extra dev --universal --generate-hashes \
  --output-file requirements.lock
```

## 6. 调整科研判断标准

主要提示词位于：

- [prompts/screen.txt](prompts/screen.txt)：相关性筛选；
- [prompts/review.txt](prompts/review.txt)：HEP 应用、风险和验证方案。

深评只以标题和摘要为证据。“摘要未说明”必须作为待核查事项，不能据此断言论文
未处理系统误差、nuisance 参数或其他问题；模拟结果也不能写成真实数据验证。
修改深评提示词会使其指纹变化，旧深评按预算重算，不要求重新初筛。
这些约束不保证模型每次都遵从，生成内容仍需人工抽查。

建议先累计两周人工反馈，再修改阈值或扩展分类。不要仅根据一次日报主观更换模型。

每个网页和 Feed 条目都可以提供“提交反馈”链接。GitHub Actions 会自动把
`FEEDBACK_REPOSITORY` 设置为当前仓库；反馈 Issue 包含保留、映射合理性、
过度推测、栏目调整和全文精读等勾选项，可逐步积累人工评估集。

### 本地人工评估

从已有状态导出最多 100 篇待标注论文，不调用模型或网络：

```bash
python -m hepml_digest --export-evaluation build/evaluation.json --evaluation-limit 100
```

样本交替覆盖可推荐、初筛未推荐和未处理候选，仅保留每篇论文的最新已知版本。
文件包含标题、摘要及分析快照；人工填写每项的：

- `human_keep`：应推荐填 `true`，不应推荐填 `false`，尚未判断保留 `null`；
- `hep_mapping_overstated`：当前分析过度推测填 `true`，没有过度推测填 `false`，未判断保留 `null`；
- `notes`：判断依据或补充意见。

不要修改论文版本或模型分析快照。导出命令拒绝覆盖已有文件，重新抽样请使用新文件名。
网页反馈 Issue 可以作为人工标注依据，但不会自动同步或自动生成标签。

```bash
python -m hepml_digest --evaluate build/evaluation.json
```

评估使用当前状态和发布阈值，输出推荐精确率 `precision`、样本内召回率 `sample_recall`、
未处理但人工认为相关的数量 `unprocessed_relevant`，以及已标注分析的过度推测比例。
推荐判断包含 `irrelevant` 硬过滤，不包含 Feed 数量截断。
未标注项不参与质量统计，分母为零时结果为 `null`；已缺失或被新版本取代的样本单独报告。
重分析后，与原快照不一致的映射标签计入 `stale_mapping_labels`，需重新人工判断。
这是分组抽取样本的诊断，不是总体无偏估计，也无法测出从未抓到的论文；不能把它当成
全量论文的召回率。建议保留一批不参与调参的人工标注样本，用于复核改动效果。

### 两类内容的边界

- **方法雷达**：来自 `stat.ML`、`cs.LG`、`physics.data-an` 等分类，重点是把通用方法映射到可检验的 HEP 场景；
- **HEP 直接应用**：包含 `hep-ex`、`physics.ins-det` 或 `nucl-ex` 分类，论文已经直接面向实验或探测器问题。

综合 Feed 保留两类内容，也可以只订阅 `methods.xml` 或
`hep-applications.xml`。

## 7. 当前范围

当前版本只分析标题和摘要，尚未自动下载论文 PDF。这是有意设计：先验证论文召回和 HEP 映射准确率，再给每日 Top 3--5 加入全文解析，可以显著降低成本和误报调试难度。

### arXiv 以外的来源

以下是可扩展方向，当前代码尚未接入：

| 来源 | 适合补充的内容 |
|---|---|
| [INSPIRE-HEP](https://inspirehep.net/) | HEP 文献、DOI、期刊信息、引用关系；建议优先接入其 REST API |
| [Semantic Scholar](https://www.semanticscholar.org/product/api) | 跨学科检索、引用关系和相似论文推荐 |
| [OpenAlex](https://docs.openalex.org/how-to-use-the-api/api-overview) | 跨学科文献、来源和开放获取位置 |
| [OpenReview](https://openreview.net/)、[PMLR](https://proceedings.mlr.press/) | 机器学习会议论文，补充方法类候选 |
| [Crossref](https://www.crossref.org/documentation/retrieve-metadata/rest-api/)、期刊 RSS | DOI 与正式出版信息；摘要不一定齐全 |
| [Unpaywall](https://unpaywall.org/products/api) | 根据 DOI 查找合法开放获取版本，不是全文搜索引擎 |

“检索到论文”不等于“可下载全文”；全文应来自开放获取链接、作者版本或有权限的机构订阅。
多源接入前应扩展目前必填的 `Paper.arxiv_id`，保存 DOI、各源 ID 和来源信息，
按 arXiv ID、DOI 合并同一论文，避免预印本和期刊版重复调用模型。

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

分析器须提供两阶段指纹、`AnalysisUsage` 和 `request_guard`；每次请求及重试前调用 guard，
并在 API 边界累计用量和失败次数。管线直接使用这个明确接口，不再兼容缺少统计或预算接口的实现。
