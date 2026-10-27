# 第 12 章：评测与可观测性验收记录

> 本文件是第 12 章验收历史的统一入口。后续按批次追加，保留原始结果、证据与源码内容指纹，不覆盖旧记录。

## 批次 docs-initial-20261026：基础知识与实验设计准备

- 范围：第 12 章 `README.md`、`basic-knowledge.md` 的**草稿文件包**；没有修改用户 Mac 本地仓库，也没有写入远程 GitHub。
- 依据：2026-10-26 读取 GitHub `ai-agent-lab-cn/zr-langChain` 根 README、第 10/11 章与上章 `reports/chapter11/acceptance.md`；当前远端尚无 `12-agent-evaluation/`。
- 官方资料：LangSmith Evaluation Concepts、How to Evaluate Agents、Trajectory Evals、Evaluation Types。
- 已计划实验：A Golden Dataset、B Tool & Trajectory、C Safety & HITL、D Regression & Observability。
- 验收状态：`DOC_DRAFT_ONLY`；本章行为测试 **NOT_RUN**，DeepSeek 在线 **NOT_RUN**，LangSmith 服务端 **NOT_RUN**。
- 本轮不报告实际 Agent 成功率、真实 Token 或执行次数；第 11 章历史结果不可用作第 12 章的新证据。
- 生成文件的 SHA-256 记录在交付包校验时填写；未来代码实验另保存**当次真实本地运行源码**的指纹，不能把本草稿哈希当作源码指纹。
- 无 Git commit / push，无模型调用、无 LangSmith 数据发送。

后续代码及运行验收批次应附：UTC/本地日期、目标 commit 或内容指纹、测试命令/退出码、数据集和 Evaluator 版本、A—D 逐项观测、真实执行次数、Trace/Experiment 标识（若有）、失败与未覆盖项。不要回写本节以制造历史通过记录。

- 本交付稿文件 SHA-256 `12-agent-evaluation/README.md`：`7e22b6d89153d55bd03618b072f1aeb59a14b6b129e8a27a83c4f7b962190b32`。
- 本交付稿文件 SHA-256 `12-agent-evaluation/basic-knowledge.md`：`c1b6fa1d0fa386708253ecdc9d212f1a2708fb2c6b1462590de389b65376d7cf`。


## 批次 local-code-20261026：A—D 离线代码增量实现与当前容器检查

- 交付位置：本对话可下载的独立更新包；**未修改用户 Mac `/Users/zrr/Desktop/zr-langChain`，未向 GitHub 写入**。
- 已编写：`12-agent-evaluation/main.py`、`evaluation_core.py`、`graph_cases.py`、`hitl_case.py`，以及 `tests/test_chapter12.py`；原本两份章节 Markdown 直接增量更新，未新增平行导读文档。
- 数据集版本：`ch12-golden-v1`；Code Evaluator 版本：`ch12-code-evaluator-v1`。
- 本次四个执行模块源码内容指纹（`main.source_sha256()`）：`ce4f0b711c24a10192fc865ab79572def5f3a6584d0be15502b197fd515de68f`。
- 执行环境：本容器 Python 3.13.5；**不是**用户 Mac 的 Python 3.11 / uv.lock 环境。该容器未安装 `langgraph`、`langchain_core`、`langsmith`，且无法解析 PyPI 域名以安装依赖。
- 实际执行：`python -B -m unittest discover -s tests -p test_chapter12.py -v`，**22 tests，OK，其中 18 PASS、4 SKIP**。通过的是纯 Python Code Evaluator、回归对齐、证据防覆盖和 CLI 异常停止契约；被跳过的四项是真实 LangGraph A—D 集成测试，**不是通过**。
- `ast.parse` 对四个运行模块和新增测试文件通过；CLI `--help` 输出正常。Markdown 链接做了静态存在性检查；Mermaid 六个代码围栏已检查边界，但本环境没有 `mmdc`，**真实 Mermaid 渲染 NOT_RUN**。
- 第 12 章真实 LangGraph `--self-test`：**NOT_RUN**；DeepSeek 在线：**NOT_RUN**；LangSmith 服务端评测/Trace：**NOT_RUN**；不产生新模型或服务端用量。
- 安全约束：不读取 `.env`、不提供在线入口、禁用自动追踪；专家工具和业务账本均为 Fake、无真实数据写入；`usage_metadata` 保持 `null`。
- 根目录 `README.patch` 是对当前已读取的 GitHub 根 README 的**建议差异**，移除 13、14 章并使 12 章标记学习中；不会自动修改用户本地 README。合入前请先检查 `git status` 和补丁内容。
- 待 Mac 上真实验收：本章全部单测与 `--self-test`、输出文件首次创建/重复拒绝、README / Mermaid 渲染、对已有项目的必要回归；这些完成前维持 `📖 学习中`。
- 未执行 git add/commit/push，未改写本文件此前 `docs-initial-20261026` 批次。


## 批次 final-local-20261027：Mac 直接复验与系列收尾

- 本次直接读取并检查 Mac 本地工作区，不依赖此前聊天中的“已通过”声明。前两批记录原样保留；此前用户给出的本章 22 项 / A—D 全 PASS 日志属于独立历史证据，未据其补造指纹。
- 基础 Git HEAD：`63f7317db5e17c32f614c7ee9fad4d2c91dedd1f`；本次运行包含工作区未提交的第 12 章，不能仅用该 HEAD 复现；四模块源码指纹：`ce4f0b711c24a10192fc865ab79572def5f3a6584d0be15502b197fd515de68f`。
- Python `3.11.15`；依赖：langchain==1.4.6; langgraph==1.2.16; langchain-google-genai==4.4.2; langchain-deepseek==1.1.1; langsmith==0.14.11; pydantic==2.14.1; numpy==2.4.6。
- `uv.lock` SHA-256：`89067a8d8ecd4f8500581c3b010972f1d6a3f2fc23f39410a8765351b1ac310a`；本轮未同步、安装或升级依赖。
- 实际命令：`.venv/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v`，**163 tests / OK，0 SKIP，退出码 0**；其中本章 22 项全部通过。耗时 1.670s。
- 实际命令：`.venv/bin/python -B 12-agent-evaluation/main.py --self-test --evidence <本次新路径>`；A—D 全部 PASS，退出码 0。
- 新证据：[offline-final-20261027.json](./offline-final-20261027.json)；batch `ch12-offline-210119981dd8`；执行时间 `2026-10-27T03:15:52.709259+00:00`；Dataset `ch12-golden-v1` / Evaluator `ch12-code-evaluator-v1`。
- 运行时关闭自动追踪与 dotenv 加载，并通过独立 Python 启动钩子禁止 socket 连接；无真实提供商调用、无 LangSmith 服务端访问，`usage_metadata=null`。
- A 实际执行 4 条 Golden Dataset；B 核对真实 ToolNode 消息与两种负例；C 检查拒绝、越权、批准与 Fake 账本幂等；D 比较同数据集 Baseline / Candidate。具体计数见新证据，不推导线上质量。
- 本章状态统一为“离线教学验收通过”；本轮修改为文档、配置模板和证据补录，未改执行代码与测试。
- 本次未重新渲染既有 Mermaid 图；保留围栏与图源码，文档检查范围见 [最终检查记录](../final-review-20261027.md)。
- DeepSeek / LLM-as-Judge 在线：NOT_RUN；LangSmith 服务端：NOT_RUN；生产能力：NOT_EVALUATED。未执行 git add / commit / push。


## 批次 full-closeout-20261027：整库源码审查后的最终复验

- 本次范围扩大至全部章节、脚本、测试、依赖与文档；前述文档收尾批次原样保留。
- 修复原始消息时序证据丢失与回归比较忽略候选 FAIL；Evaluator 升为 `ch12-code-evaluator-v2`，Dataset 仍为 `ch12-golden-v1`。此前 v1 成绩不冒充 v2 验收。
- 实际工作区执行项目 unittest：**172 tests / OK，0 SKIP，退出码 0，2.164s**。新增 `test_closeout.py` 9 项测试，包含多组负例；先观察失败，再修复通过。原批次恢复测试更新为执行隔离包装与原调度函数的完整链。
- 实际执行第 03—12 章 10 个 `--self-test`，全部退出 0；第 12 章 A—D 全部 PASS。
- 本次四模块源码 SHA-256：`11c54521464d94c5125add5010f9d7658259181f10a09158a51396d3b09a80cd`；证据生成时间 `2026-10-27T03:44:52.616210+00:00`，batch `ch12-offline-08a136c177cc`。
- 新证据：[offline-full-final-20261027.json](./offline-full-final-20261027.json)；原 [offline-final-20261027.json](./offline-final-20261027.json) 保留，不回写旧指纹或 v1 结果。
- 复用现有 Python 3.11.15 / 既有依赖，未升级；网络连接被测试启动钩子禁止，dotenv 加载禁用，自动追踪关闭。继承追踪开启的负例额外拦截上传方法，实际网络请求未成功。
- DeepSeek、Gemini、LLM-as-Judge 在线：NOT_RUN；LangSmith 服务端验收：NOT_RUN；生产能力：NOT_EVALUATED。未 commit / push。
- 具体问题、代码覆盖文件清单和限制见 [整库最终收尾记录](../final-closeout-20261027.md)。
