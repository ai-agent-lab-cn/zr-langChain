# LangSmith SmithDB 迁移验收

日期：2026-10-24（Asia/Shanghai）。第09章提交信息已统一为中文，当前提交为 `bc2483d`（原 `63f0807` 已替换）。本迁移完成本地实施与验收后，用户已人工确认提交远程；实际提交与推送结果以 Git 记录为准。

## 改动

- 新增 `scripts/langsmith_reader.py`：统一新SDK读取，显式选择字段，使用 project_id，适配大写 run_type、完整 parent_run_ids 与 token_details.raw。
- 通用报告使用异步 `runs.query` 扫描根 Run，SDK自动跨页，本地多读1条判断扫描上限；显式时间边界。
- 两个报告/验收工具通过 `traces.list_runs` 按精确 Trace ID 获取完整调用树，不重复读取根 Run，不回退旧接口。
- 保留原业务验收规则和未知Token边界；拒绝缺根、缺父、跨Trace、重复ID及不一致祖先链，超出本地分析上限时不生成部分PASS。
- 无新增依赖，`langsmith==0.14.11` 已支持；Agent代码和 basic-knowledge.md 均未触碰。

## 已执行验证

- 新读取层测试先记录缺少实现的失败，再实施；完整 unittest 49项通过（退出0）。
- 新API真实复核既有两条精确Trace：Supervisor 3次LLM / 1,473 tokens；Handoff 2次 / 515 tokens；均PASS。
- 与本地保存的旧接口证据逐字段比较：run数量、全部父子调用树、消息配对、路径、owner、LLM次数及Token完全一致。新报告不再生成Trace URL。
- 通用报告真实执行：`scripts/langsmith_report.py --chapter 09 --since-hours 168 --strict --output-dir .runs/smithdb-report`，退出0。读取到本章2条Graph / 5条LLM / 1,988 tokens，但完整验收状态为REVIEW。
- REVIEW原因：整个项目的根扫描超过默认100条；5条LLM未提供reasoning Token计量。没有失败项；不将缺失数据补0或宣称全窗口PASS。
- 新接口日志无旧方法DeprecationWarning。仅进行只读LangSmith查询，新增DeepSeek调用为0。
- 独立代码审查未发现阻断问题；审查者复跑31项LangSmith相关测试通过。

## 边界

`traces.list_runs` 一次返回items；max_spans限制本地分析规模，不限制服务端响应体大小。超大Trace需另行设计读取方式。
调用树结构检查不能独立证明服务端没有遗漏孤立叶节点，完整性仍依赖接口返回。
新服务接口可能要求自托管LangSmith版本支持；本次验证针对当前已配置服务，不代表所有部署均兼容。

真实项目ID、Trace UUID/URL和原始响应只保留在Git忽略目录。本报告不包含这些追踪元数据，历史第09章报告保留原验收时间与结果。

## 官方依据

- [Run读取迁移](https://docs.langchain.com/langsmith/smithdb-sdk-migration-runs)
- [Run查询迁移](https://docs.langchain.com/langsmith/smithdb-sdk-migration-query-runs)


## 本轮收尾复核

- 重新检查本地现有改造，未重建已完成实现；完整 unittest 49项再次通过，退出0。
- 新接口只读复核已有两条精确Trace再次PASS；调用树、消息配对、路径、owner、LLM次数和Token与旧证据逐字段一致。
- 本轮没有重复执行通用七天窗口查询，也没有重复运行DeepSeek；上文REVIEW为上一轮保留的真实结果。
- 第09章README已更新为“本地迁移已完成”，历史验收日志保留，不将旧日志中的弃用提示改写为从未出现。
- 推荐 commit message：`refactor(langsmith): 迁移 Trace 读取接口并统一验收适配`。
- 本地实施与验证已完成，用户已人工确认提交远程；提交前再次运行49项测试通过。
