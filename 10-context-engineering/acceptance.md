# 第 10 章代码实施与离线验收记录

日期：2026-10-24（Asia/Shanghai）。项目：`/Users/zrr/Desktop/zr-langChain`。

**结论：离线结构、DeepSeek A—D 小样本与 LangSmith Trace 验收均通过（2026-10-24）；结论仅限本章虚构教学样本。**

最近一次提交准备检查见第八节；此前离线复验见第六节，在线实验与服务端验收见第七节。第一至六节保留此前记录，其中“15 tests”“未执行在线”均表示当时状态。

## 一、真实执行记录

| 检查 | 实际结果 |
| --- | --- |
| `uv sync --locked` | 成功；Resolved 115 packages / Checked 108 packages |
| `uv run --locked --offline python -m unittest discover -s tests -p 'test_context_engineering.py' -v` | **15 tests，OK** |
| `uv run --locked --offline python 10-context-engineering/main.py --self-test --evidence .runs/chapter10/implementation-20261024.json` | **A、B、C、D 均 PASS，退出码 0** |
| `uv run --locked --offline python -m unittest discover -s tests -p 'test_multi_agent.py' -v` | **5 tests，OK** |
| 项目依赖声明与锁文件 | 指纹与实施前一致，无新增项目依赖 |
| 在线模型请求 / LangSmith | **未执行**；未读取本地 `.env` |

自检证据保存在本地忽略目录 `.runs/chapter10/implementation-20261024.json`。其中模型调用 ID 由本地回调产生，不是服务端 Trace 验收凭据；全部场景 `trace_requested=false`、`usage_metadata=null`。

## 二、离线场景实际观测

| 实验 | 通过的检查 | 离线模型调用数 |
| --- | --- | --- |
| A | State 保留历史；实际模型输入已经裁剪；其他线程不含 A100 | 2 |
| B | 摘要写回 State；旧原始消息被替换；回答使用更正后的 B200 | 1（另有非模型的确定性摘要服务替身） |
| C | 同用户跨线程读取偏好；其他租户作用域不共享；旧版本冲突拒绝；到期与删除后不可从仓库召回 | 6 |
| D | 两个真实专家 Agent 被调用；专家输入隔离；工具结果字段筛选；2 组工具调用与结果配对 | 4（含两个专家模型调用） |

以上数量是离线模型替身执行数，不是 DeepSeek 使用量。未提供“减少了多少真实 Token / 费用”或在线正确率结论。

## 三、15 项测试覆盖

- 完整保留并行工具交互；最新完整轮次预算不足时拒绝调用。
- 孤立、重复、未完成的工具消息协议拒绝。
- 同一线程连续工具调用使用独立调用 ID。
- 记忆的同用户读取、跨用户 / 租户隔离、版本冲突、过期隐藏、删除与墓碑。
- 非白名单偏好和非法有效期拒绝，任意记忆文本不能成为系统指令。
- 猜中他人线程或订单号时，应用与工具层拒绝访问。
- A—D 的真实 Agent、Middleware、Store、Checkpoint 与输入边界。
- `--self-test` 不读 `.env`，即使禁止建立网络连接仍能通过。
- 证据文件拒绝覆盖；失败记录为 FAILED，后续未跑项标为 NOT_RUN，异常正文不落盘。

最初测试因新模块尚不存在失败；连续工具调用与失败证据两处缺陷另有实际失败测试，再修复至通过。未把初始导入失败描述为全部行为均完成严格红绿循环。

## 四、实现边界与遗留验收

1. 身份来源为教学调用方提供的 `AppContext`，并未实现真实登录、服务端认证或生产级 ACL。
2. 版本锁、线程归属及删除墓碑仅限当前进程；没有跨进程事务、数据库后端、自动 TTL 清理或向量检索。
3. 删除覆盖 Store 读取入口，不清除旧 State、Checkpoint、摘要、Trace 与备份副本。
4. 历史预算按消息估算，不是完整模型窗口、总费用或提供商实际 Token 的硬上限。
5. B 的离线摘要是确定性服务替身；真实摘要质量与约束保持能力待 DeepSeek 验证。
6. `--online` / `--trace` 入口已有代码，真实模型、服务端 Trace、输入输出关联和完整 Token 用量均待验收。
7. 本地 `.env` 未读取；没有 Commit / Push，没有通过 GitHub API 写入远程。

## 五、审查与文档修正

独立只读审查未发现新增 P1/P2 问题；额外多轮离线诊断三次触发摘要后仍保留 B200，协议校验通过。该诊断不是在线质量验收。

本次修正基础知识文档的真实导入缺陷：`count_tokens_approximately` 应从 `langchain_core.messages.utils` 导入。保留此前扩写的 2.3 节图解与其余教学结构，只更新进度说明和相关导入路径。

推荐 Commit：`feat(ch10): 实现 Context Engineering 与 Memory 离线实验`。

## 六、本地检查与优化复验（2026-10-24）

### Standards：实现与约定

发现并修复 1 项 P2：`MemoryRepository.search()` 的 Store 查询是 namespace 前缀匹配，原来会在根偏好合法时返回子空间内未经验证的同名条目。现增加精确 namespace 相等判断；回归测试确认子空间非法 / 过期条目不会返回。

### Spec：验收设计一致性

发现并修复 1 项 P2：B 原验收只检查 B200 出现在摘要或最终回答，未严格验证 A100 的作废关系。现摘要使用 JSON 事实，检查 `active_order_id` 为 B200、`superseded_order_ids` 包含 A100 且不包含 B200。仅有正确最终答案、但遗漏作废关系的摘要会明确失败。

主检查还复现并修复 C 的假阳性：模型可以直接从动态 Prompt 获得偏好，不调用工具仍让旧验收通过。现逐次要求真实的 `read_preference` ToolMessage、正确读取值与唯一完整调用配对。

### 附加的最小优化

- 保留真实 Middleware 标记的前导摘要，避免后续 Trim 静默丢掉续接事实；摘要加当前请求超预算时明确失败。
- 摘要识别使用当前版本的 `lc_source="summarization"` 标记；读取摘要 JSON 时使用其当前前缀格式。均已核对当前安装源码，升级依赖后需重新验证，不作为跨版本保证。
- 移除未使用导入及枚举变量；保留其余中文教学注释，不引入新依赖或生产架构。

### 实际复验结果

| 检查 | 结果 |
| --- | --- |
| 实际 Mac 项目中的第 10 章 unittest | **20 tests，OK** |
| 新增的 5 项回归测试 | 修复前全部实际失败，修复后全部通过 |
| A—D `--self-test` | **全部 PASS，退出码 0** |
| 新证据文件 | `.runs/chapter10/review-20261024.json`，未覆盖首次实施证据 |
| B 的附加检查 | `correction_relation_preserved=true` |
| C 的附加检查 | `preference_tool_reads_verified=true` |
| 两路只读复核 | 原 P2 均已消除，未发现新增 P1/P2 |
| DeepSeek / LangSmith | **仍未执行** |

独立复核另连续执行 12 轮摘要诊断：每轮保留 B200，摘要中 A100 作废关系保持，实际模型输入保留摘要，累计消息协议校验通过。此诊断仍是离线行为证据。

本次不重复无关全量测试，不改动根 README、基础知识文档或前九章代码，未 Commit / Push。此前教学身份、内存生命周期、删除副本与 Token 估算的限制继续适用。

## 七、DeepSeek 小样本与 LangSmith 服务端验收（2026-10-24）

### 7.1 实验范围与执行顺序

使用本地 `.env` 中的 `deepseek-flash` 配置，密钥未写入报告或验收 JSON。先执行一次无 Trace 的 A 先导实验，通过后再对 A—D 各执行一次带 Trace 的样本。两阶段都使用真实 DeepSeek；没有替换在线答案、自动重跑已通过场景或修改实验断言。

| 批次 | 样本 | 模型调用数 | 输入 Tokens | 输出 Tokens | 总 Tokens |
| --- | --- | --- | --- | --- | --- |
| 先导，无 Trace | A | 2 | 765 | 289 | 1,054 |
| 带 Trace | A—D 各一次 | 14 | 5,541 | 1,688 | 7,229 |
| 本轮合计 | 上述两个批次 | **16** | **6,306** | **1,977** | **8,283** |

用量为提供商返回值。Reasoning Tokens 已包含在输出中，缓存详情也不重复加到输入；未按价格估算费用。先导没有开启追踪，因此只有本地提供商用量，没有对应的服务端 Trace 验收。

### 7.2 A—D 的真实在线结果

| 场景 | 在线检查 | LLM 调用数 | 输入 / 输出 / 总 Tokens |
| --- | --- | --- | --- |
| A | State 保留历史；实际模型输入裁剪；另一线程未读取 A100 | 2 | 765 / 292 / 1,057 |
| B | 摘要模型与主模型真实调用；State 中 A100 作废、B200 有效；主模型收到摘要 | 2 | 1,095 / 304 / 1,399 |
| C | 三次真实偏好工具读取，返回简洁 / 简洁 / 默认；跨线程与租户边界符合预期；本地版本、过期、删除检查通过 | 6 | 2,394 / 109 / 2,503 |
| D | 两位专家模型位于对应委派工具的后代链；输入隔离、字段筛选与两组工具结果配对通过 | 4 | 1,287 / 983 / 2,270 |

C 的版本、过期与删除是应用策略检查，并未因此变成数据库或完整历史副本删除验收。D 是 Subagents as Tools，不包含新增 Handoff 流程或 Handoff 状态转移验收。

### 7.3 LangSmith 只读核对结果

通过本次记录的精确模型 run ID 定位服务端 Trace，使用现有 v2 只读适配层查询，未公开私有 Trace 链接。

- 找到 **14 个模型 run，7 个根 Trace，70 个 span**；根、父级、完整祖先链、Trace 归属、唯一 ID 与完成状态检查通过。
- A 的两条根 Trace 对应两次独立线程调用；B 为一条根 Trace；C 的三个独立调用形成三条根 Trace；D 为一条根 Trace。
- 所有 span 均已完成且无错误；LLM 集合与本地观测的 14 个模型 run ID 一致。
- 根输出和每次模型输入的工具协议检查通过：没有孤立、重复、悬空或名称不匹配的 ToolMessage / tool_call_id。
- B 的 `summary_once` 下确有独立 LLM，摘要事实正确，并被主模型实际接收。
- C 服务端有三次 `read_preference` 工具执行，返回值符合各自作用域。
- D 服务端有两个委派工具，且各自有专家 LLM 后代；账号 / 物流输入隔离，电话、调试字段未进入模型输入。
- 每个叶子 LLM 都具有输入、输出和总 Token 字段，且总数等于输入加输出；按场景核对调用用量多重集，与本地提供商返回值一致。
- 只汇总唯一叶子 LLM，不叠加父级 Graph、工具或根 run 的聚合用量。每个场景的结构树、模型 ID、配对与用量保存为脱敏 JSON。

### 7.4 证据与边界

本地忽略目录中的证据：

- `.runs/chapter10/deepseek-A-20261024-first.json`：先导 A，未开启 Trace。
- `.runs/chapter10/deepseek-traced-20261024-first.json`：A—D 在线原始脱敏结果。
- `.runs/chapter10/langsmith-audit-20261024.json`：之后独立完成的服务端审计结果。

原始实验记录中的 `trace_acceptance="pending"` 保持不变，表示模型实验返回时尚未审计；本节结论来自之后的独立 LangSmith 审计 `status="PASS"`，没有回写或改造原始证据。

原始输入输出只在内存中用于验收，没有导出到本地验收 JSON；开启 Trace 的虚构输入输出保留在用户自己的 LangSmith 项目中。

本轮完成的是小样本阶段验收，不推导长期正确率、性能提升或真实业务效果；未验证数据库耐久性、生产级权限、完整删除链，也未计算费用。功能代码和测试没有修改，没有 Commit / Push。

## 八、代码、测试与依赖的提交准备检查（2026-10-24）

### 8.1 可复现验收工具

已将此前临时审计逻辑纳入 `scripts/verify_chapter10_trace.py`，新增 `tests/test_chapter10_trace.py` 的 11 项离线边界测试，更新章节与 scripts README 的调用说明。工具只读取已存在的 LangSmith 记录，不创建模型或重跑实验，不把原始输入输出落盘。

提交前审查与修复：

- 模型记录采用 `runs.retrieve()` 按精确 ID 读取，避免 `runs.query()` 默认一天窗口。曾尝试较早的查询起点，实际服务端返回 BadRequest；改用精确读取后通过只读复验，未扩大模型请求。
- 严格校验摘要 JSON 对象、作废订单和约束的数组类型，并比较 State 与主模型实际接收到的标记摘要事实；字段名出现或最终答案正确均不足以单独通过。
- 仅对本次精确读取额外选取 INPUTS，不改变共享适配层默认字段；重复、嵌套 LLM、部分批次、缺失用量和输出覆盖均拒绝。

### 8.2 实际验证结果

| 检查 | 结果 |
| --- | --- |
| `uv sync --locked` | 成功；115 packages resolved / 108 checked |
| 完整 `unittest discover -s tests -v` | **80 tests，OK**，含20项章节功能测试、11项验收器测试及现有项目回归 |
| 最终仓库脚本只读复验 | **PASS：7 根 Trace、14 个 LLM、70 个 span，7,229 Tokens** |
| 最终审计证据 | `.runs/chapter10/precommit-final-audit-20261024.json`，未覆盖原证据 |
| 依赖声明与锁文件 | 与本轮开始前指纹一致，无新增依赖、无版本升级 |
| 远程同步检查 | `git fetch origin` 后，本地 HEAD 与 origin/main 均为 `24644bb`，ahead 0 / behind 0（检查时状态） |

全量测试中的“配额耗尽 / BLOCKED”输出来自既有故障注入测试夹具，不是本轮真实 DeepSeek 实验失败。提交准备阶段没有新增 DeepSeek 模型调用，在线 Token 合计仍为第七节记录的 8,283。

### 8.3 提交范围

本次交付包含根 README、第10章六份文档 / 实现文件、章节测试、Trace 验收脚本与测试、scripts README，共11个文件。不纳入 `.env`、`.runs`、缓存或依赖文件；密钥按实际配置值和常见字面形式检查，未发现误入待提交文件。

本节记录提交前检查，不代表已完成 Commit / Push，也不代表远程 CI 已通过。推荐 Commit：`feat(ch10): 完成上下文工程与记忆管理实战`。
