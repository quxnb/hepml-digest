# HEP-ML arXiv Digest

从 arXiv 抓取统计学习和机器学习论文，筛选其对高能实验物理的潜在价值，生成中文网页、Atom 和 RSS。
当前只分析标题和摘要，不读取全文；生成的迁移设想不是论文已验证的结论。

## 快速开始

需要 Python 3.11 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps -e .
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。

离线演示不访问 arXiv 或模型，也不需要 Key：

```bash
python -m hepml_digest --demo
python -m http.server 8000 --directory build/demo-public
```

打开 <http://127.0.0.1:8000/>。演示状态保存在 `build/demo-state.json`，不覆盖生产状态。

## 配置模型

本地通过环境变量配置。GitHub 中进入 `Settings → Secrets and variables → Actions`：

- **Repository secrets**：添加 `DEEPSEEK_API_KEY`，填入模型服务或中转站的 Key。
- **Repository variables**：添加 API 模式、地址和模型名，不要放到 Environment variables。

例如，使用提供 `gpt-6.1-sol` 的中转站：

```text
LLM_API_STYLE=openai
DEEPSEEK_BASE_URL=https://relay.example/v1
SCREENING_MODEL=gpt-6.1-sol
REVIEW_MODEL=gpt-6.1-sol
```

模型名与地址以中转站文档为准，地址不要包含完整的 `/chat/completions`。
`DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL` 是保留的变量名，也用于非 DeepSeek 服务。

两种 API 模式共用一套流程，不自动猜测或切换：

| `LLM_API_STYLE` | 请求格式 |
|---|---|
| `deepseek`（默认） | `max_tokens`、DeepSeek `thinking`；默认 Flash 初筛、Pro 深评 |
| `openai` | `max_completion_tokens`，不发送 `thinking`；面向支持推理参数的 GPT 模型 |

地址留空时分别使用 DeepSeek 或 OpenAI 官方 API。两种模式都要求 Chat Completions 和 JSON 输出；
仅提供 Responses API 的中转站不能直接使用。OpenAI 模式初筛使用低强度推理和 4096 tokens 上限，
深评默认使用低强度推理和 8192 tokens 上限。

常用配置见 [.env.example](.env.example)，完整默认值以 [config.py](src/hepml_digest/config.py) 为准。
程序不会自动读取 `.env`；本地需将配置导出到 shell。不要提交真实密钥或发送到反馈 Issue。

## GitHub 部署与订阅

1. 将代码推送到自己仓库的默认分支，按上节配置 Secret 和 Repository variables。
2. 在 `Settings → Pages → Build and deployment → Source` 选择 **GitHub Actions**。
3. 进入 `Actions → Daily HEP-ML Digest → Run workflow`，手动运行一次。
4. 检查运行 Summary、网页和 Feed；部署后无需本地电脑常开。

定时目标为北京时间每天 08:17，但 GitHub 调度可能延迟，不保证准点。
**CI** 运行测试和离线演示，不调用付费模型；**Dependabot** 检查依赖更新。

站点和订阅地址通常为：

```text
https://<用户名>.github.io/<仓库名>/
https://<用户名>.github.io/<仓库名>/rss.xml
https://<用户名>.github.io/<仓库名>/atom.xml
```

还可订阅 `methods.xml`（方法雷达）或 `hep-applications.xml`（HEP 直接应用）。
不要使用 GitHub 的 `/blob/` 页面作为订阅地址。

GitHub Pages 在部分国内网络下不稳定，能否订阅取决于阅读器实际抓取所在的网络。
个人阅读器能正常抓取时，无需增加另一套部署。

## 本地真实运行（可选）

将同名模型配置导出为环境变量后，用独立目录做小规模验收；这会调用付费 API：

```bash
MAX_CANDIDATES=1 MAX_DEEP_REVIEWS=1 python -m hepml_digest \
  --state-file build/smoke-state.json \
  --output-dir build/smoke-public
```

## 运行边界与保护

- 默认来源为 `stat.ML`、`cs.LG`、`physics.data-an`、`hep-ex`，每次运行最多初筛 60 篇、深评 5 篇。
- 深评只处理模型推荐、相关性同时达到深评和发布阈值、且非 `irrelevant` 的论文，不设最低数量。
- `data/state.json` 保存结果、待处理队列和补抓进度；失败或超额候选留待后续运行，不要清空状态来重试。
- 去重、JSON 校验、进程锁、checkpoint 和原子写盘用于避免重复调用、丢失进度或损坏发布文件。
- 次数、输入 token、单次输出、运行时间及连续失败限制仍然有效；重试也会检查预算。
- 程序不估算金额，不维护价格变量。实际费用和消费限额在服务商后台查看、设置。

旧的价格和金额限制变量不再读取，可从 GitHub Variables 删除。

只累计接口报告的 token。若 `missing_usage_calls > 0`，汇总不完整，会告警但不停止有效分析；
此时不能依赖输入 token 总额判断全部消耗，调用次数、输出上限和时间保护仍适用。

修改模型、API 模式或提示词会按预算重新分析。中转站在同名模型下更换实际模型时，可修改
`ANALYSIS_REVISION` 触发重算；轮换 Key 不需要重算。

## 排查失败或没有更新

先查看该次运行的 **Build digest** 和 **Summary**，不要只看 Actions 绿灯或页面生成时间：

- `status`、`stop_reason`、`alerts`：是否退化或触发保护。
- `screened`、`reviewed`：本次成功处理数；`published` 是 Feed 保留的条目总数，不是当天新增数。
- `queued_papers`、`oldest_wait_days`：是否持续积压。
- `*_finish_reason`：正常通常为 `stop`；`output_limit` 表示截断，应检查输出预算和推理强度，勿盲目重试。
- 请求错误日志只保留阶段、尝试序号、`exception_type` 和 `http_status`，不输出密钥、URL 或响应正文。

GitHub 日志通常使用 UTC，例如 `10:10 UTC` 对应北京时间 `18:10`，不表示调度时区设置错误。

`APIConnectionError` 应先检查地址和运行环境的网络可达性；4xx 检查 Key、额度、模型名和参数兼容性；
429 或 5xx 检查限流和服务端状态。认证、配置错误或输出截断会保存状态后将工作流标记失败，保留线上页面。

没有新条目也可能是没有合格论文，不应为了增加数量而降低阈值。模型生成内容仍需人工抽查；
“摘要未说明”不能写成“论文未处理”，模拟结果不能写成真实数据验证。

## 开发与人工评估

```bash
pytest
python -m hepml_digest --export-evaluation build/evaluation.json
# 人工填写标签后再评估
python -m hepml_digest --evaluate build/evaluation.json
```

导出和评估均不调用模型或网络，导出不会覆盖已有文件。填写 `human_keep`、`hep_mapping_overstated`
为 `true` / `false`，未判断保留 `null`。样本包含推荐、未推荐和未处理论文；样本内指标不代表全量论文召回率。

科研判断提示词位于 [prompts/](prompts/)。调整阈值或扩展来源前，先积累人工反馈，而不是继续增加供应商或框架。

运行依赖锁定在 `requirements.lock`。更改依赖后重新生成：

```bash
uv pip compile pyproject.toml --extra dev --universal --generate-hashes \
  --output-file requirements.lock
```
