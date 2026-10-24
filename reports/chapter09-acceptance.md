# 第09章真实验收记录

日期：2026-10-24（Asia/Shanghai）。直接在 `/Users/zrr/Desktop/zr-langChain` 实施；开始时 `main...origin/main`，工作区干净。未执行 commit/push 或 GitHub API 写入。

## 实施与保护

新增 main.py、章节 README、Mermaid、5项回归测试及只读 Trace 工具；更新根 README。原 basic-knowledge.md 保持原样，未修改 .env、依赖清单及锁文件。

## 实际执行

- `uv sync`：退出0；Resolved 115 packages / Checked 108 packages。
- `uv run python -m unittest discover -s tests -v`：45项，OK，退出0。日志中“失败实验”文字来自既有异常路径单元测试，不是最终测试失败。
- `uv run python 09-langgraph-multi-agent/main.py --self-test`：两个真实 Graph 场景 PASS，退出0，零 DeepSeek。
- `uv run python 09-langgraph-multi-agent/main.py --case all --evidence <online-evidence.json>`：两个在线场景 PASS，退出0。共5次LLM，无重复实验。
- `uv run python scripts/verify_chapter09_trace.py <online-evidence.json> <langsmith-trace.json>`：只读两条精确 Trace，PASS，退出0。

## 修复与证据

初始测试证实 main.py 缺失。实现后发现动态加载测试模块未注册 sys.modules 导致 TypedDict 注解解析失败，以及带 checkpointer 的未知请求测试缺少 thread_id；均修正测试配置后完整通过。未以模拟结果替代在线调用。

- supervisor：run_id `local-run-001`；LLM 3 次；tokens {'input': 1046, 'output': 427, 'total': 1473}；路径 `supervisor → account_agent → supervisor`；[LangSmith Trace](LOCAL_TRACE_ONLY)。
- handoff：run_id `local-run-002`；LLM 2 次；tokens {'input': 427, 'output': 88, 'total': 515}；路径 `triage → logistics_agent`；[LangSmith Trace](LOCAL_TRACE_ONLY)。

服务端检查：所有 runs 已结束且无 error；父 run 均在同一 Trace；Supervisor 工具的后代包含专员 LLM；root.outputs 的 ToolMessage/tool_call_id、步骤及 owner 匹配。Handoff 本地真实 checkpoint 无待执行节点且 active_agent=logistics_agent。只汇总 llm runs，合计1,988 tokens。

## 边界与风险

在线只覆盖账号委派与物流移交各一例；不宣称全领域正确率、多轮持久化、生产就绪或真实业务执行。SDK 有弃用提示，详见 trace-verification.log。原始密钥未输出，报告仅保留教学路由、run_id、token与调用树；LangSmith保留教学消息。未换算费用。

## 文件指纹

- `09-langgraph-multi-agent/main.py` SHA256 `4d42b30e9b04fbae162f400f96a8fff9083f582f92f51a6c1c39a361cbef691d`
- `scripts/verify_chapter09_trace.py` SHA256 `42a523df6e94df6095996dbf348cd0951aa5e53a3ddaabc33d3e7449550df77d`
- `09-langgraph-multi-agent/basic-knowledge.md` SHA256 `6d11827e0eb023dfd1a862e02af5d599cf26e6c82fe6ed8cc9f54441d4b14f21`

## 推荐 Commit（未执行）

`feat(ch09): add LangGraph supervisor and handoff demos with verified traces`

## 公开版本脱敏

本报告和 chapter09/ 文件中的 Run 标识为匿名别名；真实 UUID、项目标识与 Trace URL 仅保留在 Git 忽略的 `.runs/chapter09-private-evidence/`。在线验收结果与 Token 数值保留，公开文件不能直接用于远程重查。
