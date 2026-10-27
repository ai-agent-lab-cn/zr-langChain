# zrLangChain 安装与运行

适用于当前 12 章仓库。所有命令从仓库根目录执行；不需要逐章创建虚拟环境，也不需要重新 `uv init`。

## 一、准备环境

需要 Python 3.11 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。已有项目直接进入本地目录；首次获取仓库可执行：

```bash
git clone https://github.com/ai-agent-lab-cn/zr-langChain.git
cd zr-langChain
uv sync --locked
```

远端内容以实际仓库为准；本次本地收尾及第 12 章未提交修改不等于已发布到 GitHub。使用本地交付包时直接解压并进入包内项目根目录。

`--locked` 在依赖声明与锁文件不一致时停止，避免无意改变锁定版本。首次同步可能联网下载 Python 或依赖；已经同步且本地缓存齐备时，可用 `uv sync --locked --offline`。

## 二、先运行不需要密钥的实验

```bash
uv run --locked --offline python -B 12-agent-evaluation/main.py --self-test

LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false \
  uv run --locked --offline python -B -m unittest discover -s tests -p 'test_*.py' -v
```

第 12 章预期 A—D 全部 `PASS`；本次环境的项目回归为 `172 tests / OK`，没有跳过项。测试数量会随后续改动变化。`uv --offline` 不会自动阻止应用访问模型服务，应使用对应章节明确提供的离线模式。

## 三、按需配置在线服务

```bash
test -e .env || cp .env.example .env
```

编辑 `.env`，保留已有有效配置：

| 变量 | 用途 |
| --- | --- |
| `GEMINI_API_KEY`、`GEMINI_MODEL` | 第 01—07 章的 Gemini 在线示例 |
| `GEMINI_EMBEDDING_MODEL` | 第 06 章在线 Embedding；该章 `--retrieval-only` 仍可能计费 |
| `DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL` | 第 08—10 章的 DeepSeek 在线示例 |
| `LANGSMITH_TRACING` | 支持自动追踪的章节是否上传 Trace；模板默认 `false` |
| `LANGSMITH_API_KEY`、`LANGSMITH_PROJECT` | LangSmith 上传或只读分析所需配置 |

模型 ID 应选择账号当前可用且支持相应 Tool Calling / Structured Output 能力的型号。模板保留仓库记录过的默认 ID，本次没有验证在线可用性。第一章已显式将 `GEMINI_API_KEY` 传给模型，避免遗留的 `GOOGLE_API_KEY` 覆盖本章配置。

```bash
# 配置有效 Gemini Key 后，运行最小 Agent；会请求模型。
uv run --locked python 01-basic-agent/main.py
```

若需要 LangSmith Trace，按章内说明开启追踪，并确认发送的内容仅为允许上传的教学数据。第 10 章要求显式 `--online`，上传还需 `--trace`；第 11、12 章当前只支持本地离线实验，不读取 `.env`。

## 四、运行模式速查

| 章节 | 本地自检 | 在线入口 |
| --- | --- | --- |
| 01—02 | 没有统一的 `--self-test` 参数 | `main.py`；具体场景见章内 README |
| 03—09 | `main.py --self-test` | 不带 `--self-test` 会进入在线分支；按章内参数运行 |
| 10 | `main.py --self-test`；默认离线 | `main.py --online`；追踪另加 `--trace` |
| 11—12 | `main.py --self-test`；始终离线 | 当前未提供 |

第 05 章自检需要真实本地 MCP stdio 子进程；第 06 章离线自检使用本地确定性 Embedding；这些都不能替代在线模型能力验收。

如需保存第 12 章证据，指定尚不存在的文件：

```bash
uv run --locked --offline python -B 12-agent-evaluation/main.py --self-test \
  --evidence .runs/chapter12/my-offline-run.json
```

重复路径会拒绝覆盖。只想查看历史结果时，直接阅读 [验收记录](./reports/chapter12/acceptance.md)，无需重跑在线实验。

## 五、排障与收尾

- 缺少依赖：先在根目录完成 `uv sync --locked`；`--offline` 缓存不足时不能完成首次安装。
- 缺少 Key / 模型不可用：核对 `.env`、提供商与模型 ID；不要把密钥贴入日志或提交 Git。
- DeepSeek `402`：余额不足，停止付费路径；`429`：请求频率或并发限制，不能仅凭状态码断言每日配额耗尽。
- 测试输出出现 `FAILED/BLOCKED` 字样：部分测试在验证故障路径，判断测试结果应看最终 `unittest` 汇总与退出码。
- 内存状态随进程退出丢失属于当前教学实现边界，不代表已验证持久化。

历史第 8 章增量包变更保留在 [CHANGELOG.md](./CHANGELOG.md)，不再作为当前安装步骤。当前版本和复验范围见 [最终检查记录](./reports/final-closeout-20261027.md)。Git commit / push 由用户另行确认后执行。

官方依据：[uv 锁定与同步](https://docs.astral.sh/uv/concepts/projects/sync/) · [LangChain Quickstart](https://docs.langchain.com/oss/python/langchain/quickstart/) · [DeepSeek 错误码](https://api-docs.deepseek.com/quick_start/error_codes/)。
