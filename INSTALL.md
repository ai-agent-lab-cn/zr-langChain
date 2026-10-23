# zrLangChain · Experiment Reliability V1 安装与验收

> 解压到**本地 VS Code 已打开的 `zr-langChain/` 仓库根目录**；这是增量交付包，不会自动提交到 GitHub。

## 覆盖与新增文件

- 覆盖：`08-langgraph-hitl/main.py`、`08-langgraph-hitl/README.md`
- 保留：两张第八章 Mermaid 图（文件内容未变）
- 新增：`scripts/experiment_errors.py`、`scripts/experiment_runner.py`
- 更新：`scripts/langsmith_report.py`、`scripts/README.md`
- 新增：`tests/test_experiment_reliability.py`
- 原有：`tests/test_langsmith_report.py` 原样保留

若已有 Trace Analyzer，请确保使用工具包中的更新版，以便识别 `--batch-latest`。

## DeepSeek 环境配置

在本地根目录 `.env` 中新增 `DEEPSEEK_API_KEY=...`、`DEEPSEEK_MODEL=deepseek-flash`，保留 `LANGSMITH_TRACING` 相关配置。请确认 DeepSeek 账户余额。原有 Gemini Key 可以留给前七章，但本章不再读取。**不要提交 `.env`。**

## VS Code 终端执行

```bash
cd /Users/zrr/Desktop/zr-langChain
# 仅当根 .gitignore 中没有该行时追加，避免 .runs/ 中的本地状态进入 Git
grep -qxF ".runs/" .gitignore 2>/dev/null || printf '\n.runs/\n' >> .gitignore

uv add langchain-deepseek
uv sync
uv run python -m unittest discover -s tests -p 'test_*.py' -v
uv run python 08-langgraph-hitl/main.py --self-test

# 在线首次运行：新批次
uv run python 08-langgraph-hitl/main.py --case all

# 额度恢复后：跳过本批已 PASS 的场景
uv run python 08-langgraph-hitl/main.py --case all --resume-batch

# 分析本批 Trace
uv run python scripts/langsmith_report.py --chapter 08 --batch-latest --since-hours 24 --expect-online --expect-all
```

批次文件 `.runs/chapter08/latest.json` 只有模型、场景状态、时间、非敏感错误分类与批次 ID，不包含 API Key 或模型请求正文。**将 `.runs/` 加入 Git 忽略列表。**

## 提交规范

通过本地测试后，在 VS Code 本地 Git 提交，不直接修改远程：

```bash
git status
git diff --stat
git add 08-langgraph-hitl/ scripts/ tests/ .gitignore
git commit -m "feat: zrLangChain 实验可靠性增强：配额熔断与断点验收"
git push origin main
```

**注意：** 之前已经失败的在线运行不含本版 `batch_id` / `.runs` 数据，无法被新参数恢复；安装后的**第一次在线运行必须不带 `--resume-batch`**，之后才可以按批次续跑。

## 真实验收状态

工具模拟测试可在不安装 LangGraph 的环境运行；已有第八章的**旧版**真实 LangGraph 离线自检来自用户提供的四组成功日志。新的 `main.py` 增加了在线调度代码，仍应在本机重跑 `--self-test`。DeepSeek 本轮已经收到明确每日配额耗尽 429，**在线成功验收及 LangSmith 成功 Trace 尚未验证**。
