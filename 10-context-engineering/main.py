"""第10章：上下文工程 A—D 教学实验。默认离线，显式 --online 才访问 DeepSeek。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from threading import RLock

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt, wrap_model_call
from langchain.tools import ToolRuntime, tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import tracing_context

from context_policy import AppContext, MemoryRepository, select_history, validate_protocol
from offline_model import OfflineChatModel, offline_summary

# 入口脚本位于章节目录内，因此向上两级定位仓库根目录；只有在线分支才读取其 .env。
ROOT = Path(__file__).resolve().parent.parent
BASE_PROMPT = "你是星河工单助手。仅使用获准事实；调用工具取得结果后再回复，不重复调用。不要索取密码。"


# 【观测层】回调收到的是实际模型调用输入，可用来检查裁剪和专家隔离是否生效。
# 原文只留在进程内；evidence() 对外导出调用标识、计数和用量观测值。
class Capture(BaseCallbackHandler):
    """在内存中捕获实际模型输入用于验收；导出的证据只含计数，不含原文。"""
    # 每个 Demo 单独创建观测容器，避免不同实验的调用记录混算。
    def __init__(self):
        self.inputs = []
        self.tags = []
        self.usage = []
        self.run_ids = []
        # 专家工具可能并行执行，共用回调的写操作需避免交错破坏 inputs / tags 对应关系。
        self._lock = RLock()

    # 框架在模型调用开始时触发；messages 是批次列表，每个批次是一组消息。
    # tags 与 inputs 按相同顺序保存，D 场景据此区分账号专家和物流专家。
    def on_chat_model_start(self, serialized, messages, *, run_id, tags=None, **kwargs):
        with self._lock:
            # 复制每个批次的列表容器；保留消息对象，供运行结束后审查实际可见内容。
            self.inputs.extend([list(batch) for batch in messages])
            self.tags.extend([list(tags or []) for _ in messages])
            # 本地模型运行 ID 可辅助关联事件，不等于已成功上传到 LangSmith。
            self.run_ids.append(str(run_id))

    # 只读取模型响应附带的 usage_metadata；没有该字段就不虚构 Token 数。
    # 每个 generation 的用量单独保存，后续汇总应留意批量生成等情况。
    def on_llm_end(self, response, **kwargs):
        with self._lock:
            for group in response.generations:
                for generation in group:
                    message = getattr(generation, "message", None)
                    usage = getattr(message, "usage_metadata", None)
                    if usage is not None:
                        self.usage.append(dict(usage))

    # 导出验收摘要；model_calls 是捕获的输入批次数，当前实验每次只发一批。
    # provider_reported 表示观测到了提供商字段，不保证所有调用的用量都齐全。
    def evidence(self):
        return {
            "model_calls": len(self.inputs),
            "observed_model_run_ids": self.run_ids,
            "usage_observed_calls": len(self.usage),
            "usage_metadata": self.usage or None,
            "token_usage_status": "provider_reported" if self.usage else "unavailable",
        }


# 【运行配置】thread_id 选择 Checkpoint；callbacks 收集证据；tags 标记章节。
# recursion_limit 限制图执行步数，不是模型调用次数或总费用的精确上限。
def config(thread_id, capture):
    return {"configurable": {"thread_id": thread_id}, "callbacks": [capture],
            "recursion_limit": 12, "tags": ["chapter10"]}


# 【应用封装】把 Agent、记忆仓库、观测器与线程归属表放在一起供实验调用。
# 每个 Demo 的 owners 独立；该表仅在进程内存在，不是生产级用户登录系统。
@dataclass
class Demo:
    agent: object
    memory: MemoryRepository
    capture: Capture
    # default_factory 为每个 Demo 创建独立字典，避免多个实例共享一个可变默认值。
    owners: dict = field(default_factory=dict)

    # 先检查应用层线程归属，再传入最新 HumanMessage。
    # 配置中的 thread_id 用于恢复历史，context=ctx 注入本次可信运行依赖。
    def invoke(self, text, thread_id, ctx):
        """先验证线程归属，再让框架读取 Checkpoint；不是仅凭 thread_id 放行。"""
        if not text.strip():
            raise ValueError("问题不能为空")
        identity = (ctx.tenant_id, ctx.user_id)
        # 新线程登记首次访问者；已有线程继续使用原归属，不能被后来的调用覆盖。
        owner = self.owners.setdefault(thread_id, identity)
        if owner != identity:
            raise PermissionError("线程不属于当前身份")
        return self.agent.invoke({"messages": [HumanMessage(content=text)]},
                                 config(thread_id, self.capture), context=ctx)


# 【组装入口】model_factory=None 表示离线；传入工厂表示使用调用方提供的模型。
# summarize 控制是否加入摘要策略，experts 控制工具集合是否改为专家委派。
# 这里创建一次 Saver 并交给 Agent；同一 Demo 多次 invoke 才能接续线程状态。
def build_demo(*, model_factory=None, budget=400, summarize=False, experts=False, memory=None):
    """同一套真实 Agent 循环；仅模型工厂与摘要服务在离线时使用确定性替身。"""
    # 模型差异集中在工厂入口，Agent 编排、工具与记忆逻辑在两种模式中保持一致。
    factory = model_factory or (lambda role: OfflineChatModel(role=role))
    capture = Capture()
    memory = memory if memory is not None else MemoryRepository()

    # 每次模型调用前读取该用户的最新有效偏好，映射成受控的系统提示。
    # 读取的是 Runtime Context 中的身份，不从模型生成的工具参数取得用户 ID。
    @dynamic_prompt
    def user_prompt(request: ModelRequest) -> str:
        # 每次重新读有效偏好，避免将过期或已删除条目继续当作当前系统设置。
        value = memory.get(request.runtime.context)
        style = value["style"] if value else "默认"
        # 仅枚举映射可以进入系统提示，任意 Store 文本不得升级为指令。
        suffix = {"简洁": "用中文简洁回答。", "详细": "用中文解释必要步骤。"}.get(style, "用中文回答。")
        return BASE_PROMPT + suffix

    # handler 表示后续模型调用链；override 仅替换本次请求视图。
    # 没有返回 State 更新，因此 State 中的旧消息不会因这次裁剪自动被删除。
    @wrap_model_call
    def bounded_context(request: ModelRequest, handler):
        # 限制历史消息估算，不冒充整个提供商窗口的精确硬上限。
        selected = select_history(request.messages, budget)
        return handler(request.override(messages=selected))

    # ToolRuntime 由框架注入，不作为需要模型填写的工具参数。
    # 模型只提供订单号；身份与订单归属检查仍在程序侧执行。
    @tool
    def lookup_order(order_id: str, runtime: ToolRuntime[AppContext]) -> str:
        """查询当前用户订单的物流；必须检查归属，只返回必要字段。"""
        return lookup_owned_order(order_id, runtime.context)

    # 工具展示显式 Store 读取；它与动态 Prompt 是两种不同的记忆使用入口。
    @tool
    def read_preference(runtime: ToolRuntime[AppContext]) -> str:
        """读取当前已认证用户的回复风格。"""
        value = memory.get(runtime.context)
        return "默认" if value is None else value["style"]

    tools = [lookup_order, read_preference]
    if experts:
        # 专家无共享 Checkpointer；各自输入由包装工具重新构造。
        account_agent = create_agent(factory("account"), tools=[], system_prompt=
            "你是账号专家。只处理账号问题，给出简短建议，不索取密码。", name="account-expert")
        logistics_agent = create_agent(factory("logistics"), tools=[], system_prompt=
            "你是物流专家。只基于收到的物流事实给出简短建议。", name="logistics-expert")

        # 【专家输入边界】工具无自由任务参数，只传本例允许的账号故障描述。
        # 沿用父级回调并添加专家标签，使验收能观察到真实子 Agent 的模型输入。
        @tool
        def account_expert(runtime: ToolRuntime[AppContext]) -> str:
            """请账号专家处理登录失败，不传递物流信息。"""
            child_config = {**runtime.config, "tags": ["expert:account"]}
            result = account_agent.invoke({"messages": [HumanMessage(content="用户账号登录失败，请给排查建议。") ]},
                                          child_config)
            return json.dumps({"domain": "account", "advice": str(result["messages"][-1].content)[:180],
                               "source": "account-expert", "truncated": len(str(result["messages"][-1].content)) > 180},
                              ensure_ascii=False)

        # 先做订单授权和字段筛选，再把物流事实传入独立专家上下文。
        # 授权失败立即返回，不调用专家；账号历史和无关支付内容不随父线程传入。
        @tool
        def logistics_expert(order_id: str, runtime: ToolRuntime[AppContext]) -> str:
            """请物流专家处理订单物流，不传递账号资料。"""
            facts = lookup_owned_order(order_id, runtime.context)
            if "拒绝" in facts:
                return facts
            child_config = {**runtime.config, "tags": ["expert:logistics"]}
            result = logistics_agent.invoke({"messages": [HumanMessage(content=facts)]}, child_config)
            advice = str(result["messages"][-1].content)
            return json.dumps({"domain": "logistics", "advice": advice[:180],
                               "source": "logistics-expert", "truncated": len(advice) > 180}, ensure_ascii=False)

        # 专家模式替换主 Agent 可调用的工具集合；主 Agent 通过工具委派，不直接共享专家消息历史。
        tools = [account_expert, logistics_expert]
    middleware = [user_prompt]
    if summarize:
        summary_model = factory("summary")

        # 在线摘要服务只调用一次模型；空响应或异常直接交给外层失败处理。
        # 选择显式 summarizer 服务，以免误把内置摘要模型包装器的重试当作 SDK 零重试。
        def summary_once(messages):
            # 显式服务替身路径：避开内置摘要模型包装器的默认重试。
            # 在线沿用 SDK max_retries=0；失败直接终止本场景。
            response = summary_model.invoke([SystemMessage(content=
                "提取继续任务所需的摘要，只返回 JSON 对象，不加 Markdown 围栏。"
                "字段：superseded_order_ids 为已作废订单号数组，active_order_id 为当前有效订单号，"
                "constraints 为明确约束的字符串数组。正确保留更正关系，不编造或复活作废订单。"), *messages])
            if not str(response.content).strip():
                raise ValueError("摘要为空")
            return str(response.content)

        # trigger / keep 的消息条数仅为便于观察的教学阈值，不代表通用 Token 窗口配置。
        middleware.append(SummarizationMiddleware(
            model=summary_model, trigger=("messages", 10), keep=("messages", 2),
            summarizer=summary_once if model_factory is not None else offline_summary))
        # 摘要先处理 State，再将压缩后的消息交给预算包装层。
    middleware.append(bounded_context)
    # 同时注入短期 Checkpointer 和长期 Store；两者用途不同，不会自动相互复制数据。
    agent = create_agent(factory("assistant"), tools=tools, middleware=middleware,
                         context_schema=AppContext, store=memory.store,
                         checkpointer=InMemorySaver(), name="chapter10-agent")
    return Demo(agent, memory, capture)


# 【虚构业务接口】先限制身份和可查询订单，再构造用于演示筛选的原始数据。
# PRIVATE_PHONE_MARKER 等标记用于测试泄漏，没有真实用户电话或实际物流查询。
def lookup_owned_order(order_id, ctx):
    """虚构业务数据源；先按用户校验，再构造模型可见的白名单字段。"""
    if ctx.user_id != "u1" or ctx.tenant_id != "demo" or order_id not in {"A100", "B200"}:
        return "拒绝：当前身份无权查询此订单"
    raw = {"order_id": order_id, "status": "运输中", "eta": "明天", "source": "fixture-logistics",
           "private_phone": "PRIVATE_PHONE_MARKER", "debug_blob": "INTERNAL_DEBUG_MARKER" * 1000}
    # 使用字段白名单重新构造输出，电话和超长调试字段均不进入模型可见结果。
    selected = {key: raw[key] for key in ("order_id", "status", "eta", "source")}
    # 显式标注做过字段筛选；这里不是对真实物流系统的查询。
    return json.dumps({**selected, "filtered": True}, ensure_ascii=False)


# 实验断言同时返回 True，方便把通过结果直接放进 checks 字典。
# 显式 raise 不会像 Python assert 语句那样随优化选项被关闭。
def require(condition, label):
    if not condition:
        raise AssertionError(label)
    return True


# 【A—D 验收入口】运行夹具、捕获实际输入并执行断言，全部满足才返回 PASS。
# model_factory 控制模型来源；在线仍使用相同规则，不以预期答案替换真实模型结果。
def run_case(case, *, model_factory=None):
    """返回实际断言的结果。在线回答有随机性，失败必须保留为失败。"""
    ctx = AppContext("demo", "u1")
    online = model_factory is not None
    # 【实验 A】注入历史 → 请求订单 → 对照 State 与实际输入 → 新线程隔离检查。
    if case == "A":
        demo = build_demo(model_factory=model_factory, budget=220)
        # 旧无关对话足够长，以触发历史筛选；关键订单放在最新完整轮次。
        history = [HumanMessage(content="旧套餐讨论" * 300), AIMessage(content="旧问题已结束"),
                   HumanMessage(content="我的有效订单是 A100，请只记住这个订单"), AIMessage(content="收到")]
        cfg = config("a", demo.capture)
        demo.owners["a"] = (ctx.tenant_id, ctx.user_id)
        # 直接为本次实验准备初始状态；这是夹具装载，不是假装这些历史刚刚真实发生。
        demo.agent.update_state(cfg, {"messages": history})
        result = demo.invoke("请说出当前订单号", "a", ctx)
        # 读取 Middleware 之后的实际模型输入，而不是只依据最终回答猜测裁剪是否发生。
        actual = demo.capture.inputs[-1]
        checks = {
            "state_kept_history": require(any("旧套餐" in str(x.content) for x in result["messages"]), "A State 历史丢失"),
            "model_input_trimmed": require(not any("旧套餐" in str(x.content) for x in actual), "A 未裁剪模型输入"),
        }
        answer = str(result["messages"][-1].content)
        require("A100" in answer, "A 关键订单遗漏")
        other = demo.invoke("当前订单号是什么？不知道就说不知道", "other", AppContext("demo", "u2"))
        checks["other_thread_isolated"] = require("A100" not in str(other["messages"][-1].content), "A 跨线程泄漏")
    # 【实验 B】把旧订单与更正放入足够长的历史，触发摘要后确认有效事实仍然保留。
    elif case == "B":
        demo = build_demo(model_factory=model_factory, budget=700, summarize=True)
        history = [HumanMessage(content="旧订单 A100"), AIMessage(content="记录旧订单")]
        for i in range(6):
            history.extend([HumanMessage(content=f"无关历史 {i} " + "背景" * 80), AIMessage(content="已处理")])
        history.extend([HumanMessage(content="更正：A100 已作废，当前有效订单 B200"), AIMessage(content="以新订单为准")])
        cfg = config("b", demo.capture)
        demo.owners["b"] = (ctx.tenant_id, ctx.user_id)
        demo.agent.update_state(cfg, {"messages": history})
        result = demo.invoke("当前有效订单是什么？只答有效订单号", "b", ctx)
        # 从 Checkpoint 读取执行后的 State，确认摘要实际写回，而非只改变本次模型输入。
        saved = demo.agent.get_state(cfg).values["messages"]
        summaries = [message for message in saved
                     if message.additional_kwargs.get("lc_source") == "summarization"]
        require(len(summaries) == 1, "B 没有唯一的实际摘要消息")
        # 当前版本在摘要 JSON 前附一段说明，用空行分隔；这是版本相关的展示格式。
        summary_text = str(summaries[0].content).split("\n\n", 1)[-1]
        try:
            facts = json.loads(summary_text)
        except (ValueError, TypeError) as exc:
            raise AssertionError("B 摘要不符合 JSON 事实格式") from exc
        corrected = (isinstance(facts, dict) and facts.get("active_order_id") == "B200"
                     and isinstance(facts.get("superseded_order_ids"), list)
                     and "A100" in facts["superseded_order_ids"]
                     and "B200" not in facts["superseded_order_ids"])
        checks = {
            "summary_written_to_state": require("B200" in summary_text, "B 摘要未写回状态"),
            "correction_relation_preserved": require(corrected, "B 摘要未明确保留有效与作废关系"),
            "old_messages_removed": require(not any(str(x.content) == "旧订单 A100" for x in saved), "B 未压缩状态"),
        }
        answer = str(result["messages"][-1].content)
        require("B200" in answer and "A100" not in answer, "B 未遵从更正")
    # 【实验 C】Store 共享由 namespace 决定；换 thread 并不需要复制偏好文档。
    elif case == "C":
        # 使用可推进的虚拟时钟，精确检查到期边界，不消耗真实等待时间。
        clock = [100.0]
        memory = MemoryRepository(clock=lambda: clock[0])
        demo = build_demo(model_factory=model_factory, memory=memory)
        memory.put(ctx, "简洁", expected_version=0)
        first = demo.invoke("调用工具读取我的回复风格，只答读取结果", "c1", ctx)
        result = demo.invoke("调用工具读取我的回复风格，只答读取结果", "c2", ctx)
        answer = str(result["messages"][-1].content)
        checks = {"cross_thread_shared": require("简洁" in answer and "简洁" in str(first["messages"][-1].content), "C 跨线程偏好未共享")}
        denied = demo.invoke("调用工具读取我的回复风格，只答读取结果", "c3", AppContext("other-tenant", "u1"))
        checks["other_user_isolated"] = require("简洁" not in str(denied["messages"][-1].content), "C 租户隔离失效")
        # 动态 Prompt 也含偏好，因此正确回答不足以证明工具读取确实发生。
        # 逐次检查标准 ToolMessage、调用名称和实际返回值，防止无工具的假阳性。
        for observed, expected in ((first, "简洁"), (result, "简洁"), (denied, "默认")):
            tool_results = [message for message in observed["messages"] if isinstance(message, ToolMessage)]
            require(validate_protocol(observed["messages"]) == 1 and len(tool_results) == 1
                    and tool_results[0].name == "read_preference"
                    and str(tool_results[0].content).strip() == expected, "C 偏好读取工具未按预期执行")
        checks["preference_tool_reads_verified"] = True
        try:
            memory.put(ctx, "详细", expected_version=0)
        except ValueError:
            checks["conflict_rejected"] = True
        else:
            raise AssertionError("C 没有拒绝旧版本更新")
        # 确认旧版本更新已被拒绝后，再以正确版本更新；随后模拟五秒到期。
        memory.put(ctx, "详细", expected_version=1, ttl=5)
        clock[0] = 105.0
        checks["expired_hidden"] = require(memory.get(ctx) is None and memory.search(ctx) == [], "C 过期记忆仍可读")
        memory.delete(ctx)
        checks["deleted_hidden"] = require(memory.get(ctx) is None and memory.search(ctx) == [], "C 删除未生效")
        # Store 删除验收不等于旧 Checkpoint / Trace 的物理清除。
    # 【实验 D】在父请求中故意加入无关标记，检查专家与工具输入是否遵守最小披露。
    elif case == "D":
        demo = build_demo(model_factory=model_factory, budget=700, experts=True)
        result = demo.invoke("专家协作：分别调用账号专家处理登录失败，物流专家核查 A100；最后合并建议。"
                             "无关支付信息：PAYMENT_PRIVATE_MARKER", "d", ctx)
        answer = str(result["messages"][-1].content)
        # 按回调标签筛选专家真实输入，不能只检查包装工具的参数就宣称专家隔离成功。
        account = [batch for batch, tags in zip(demo.capture.inputs, demo.capture.tags) if "expert:account" in tags]
        logistics = [batch for batch, tags in zip(demo.capture.inputs, demo.capture.tags) if "expert:logistics" in tags]
        require(account and logistics, "D 未实际调用两个专家")
        checks = {
            "expert_calls": len(account) + len(logistics),
            "expert_context_isolated": require("A100" not in str(account) and "PAYMENT_PRIVATE_MARKER" not in str(account + logistics)
                                                and "登录失败" not in str(logistics), "D 专家上下文污染"),
            "paired_tools": validate_protocol(result["messages"]),
            "tool_results_filtered": require("PRIVATE_PHONE_MARKER" not in str(demo.capture.inputs)
                                              and "INTERNAL_DEBUG_MARKER" not in str(demo.capture.inputs), "D 工具结果未筛选"),
        }
        require(checks["paired_tools"] == 2, "D 预期两个专家调用")
    else:
        raise ValueError("未知实验")
    # 每组实验最后都检查完整消息协议；通过后才组装 PASS 结果。
    validate_protocol(result["messages"])
    return {"case": case, "mode": "online" if online else "offline", "status": "PASS", "answer": answer,
            "checks": checks, "telemetry": demo.capture.evidence(),
            "trace_acceptance": "pending" if online else "not_applicable"}


# 【CLI 入口】参数校验 → 选择离线或在线模型 → 顺序执行 → 保存脱敏证据。
# argv 可注入，便于测试直接调用；脚本运行时默认读取命令行参数。
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # 离线与在线参数互斥；未指定时 factory 仍为 None，因此默认离线。
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--offline", action="store_true", help="默认模式：只替换模型，真实执行 Agent 与存储")
    modes.add_argument("--online", action="store_true", help="显式读取本地 .env 并调用 DeepSeek")
    parser.add_argument("--case", choices=["A", "B", "C", "D", "all"], default="all")
    parser.add_argument("--self-test", action="store_true", help="强制执行全部离线断言，不读取 .env")
    parser.add_argument("--trace", action="store_true", help="仅在线：发送虚构实验数据到 LangSmith")
    parser.add_argument("--evidence", type=Path, help="新建不含对话原文的 JSON 证据文件；拒绝覆盖")
    args = parser.parse_args(argv)
    # 先拒绝相互矛盾的参数，再创建任何客户端，防止自检意外联网。
    if args.self_test and args.online:
        parser.error("--self-test 不能与 --online 同用")
    if args.trace and not args.online:
        parser.error("--trace 要求显式 --online")
    if args.evidence and args.evidence.exists():
        parser.error("证据文件已存在，请使用新路径")
    factory = None
    # 在线请求与 Trace 上传是两个独立开关，必须显式同时启用才追踪。
    trace_enabled = bool(args.online and args.trace)
    try:
        if args.online:
            # 默认及 --self-test 路径不读 .env、不创建真实模型客户端。
            from dotenv import load_dotenv
            from langchain_deepseek import ChatDeepSeek
            load_dotenv(ROOT / ".env")
            if not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("DEEPSEEK_MODEL"):
                raise ValueError("缺少在线模型配置")
            if trace_enabled and not os.getenv("LANGSMITH_API_KEY"):
                raise ValueError("缺少 Trace 配置")
            # 角色通过各 Agent 的提示与工具区分；在线工厂统一使用现有环境配置，不打印密钥。
            factory = lambda role: ChatDeepSeek(model=os.environ["DEEPSEEK_MODEL"], temperature=0,
                                                timeout=45, max_retries=0, max_tokens=600)
        # 自检固定执行全部场景；普通运行可以只选择一组，减少排查成本与在线请求。
        cases = "ABCD" if args.self_test or args.case == "all" else args.case
        records = []
        # 显式关闭离线 tracing，即使父进程设置 LANGSMITH_TRACING=true。
        succeeded = True
        # 局部追踪上下文显式覆盖环境默认值，离线即使继承 tracing 环境变量也不上传。
        with tracing_context(enabled=trace_enabled):
            for index, case in enumerate(cases):
                try:
                    item = run_case(case, model_factory=factory)
                    # 证据只保留结构、计数与提供商用量，不落盘模型回答或请求正文。
                    record = {key: value for key, value in item.items() if key != "answer"}
                    record["trace_requested"] = trace_enabled
                # 场景级失败只保存异常类型；保留此前通过结果，但不继续运行后续场景。
                except Exception as exc:
                    record = {"case": case, "mode": "online" if args.online else "offline",
                              "status": "FAILED", "error_type": type(exc).__name__,
                              "trace_requested": trace_enabled, "trace_acceptance": "pending"}
                    succeeded = False
                records.append(record)
                print(json.dumps(record, ensure_ascii=False))
                if not succeeded:
                    # 未执行项必须显式标记，不能因批次提前结束而被误认为通过。
                    records.extend({"case": pending, "status": "NOT_RUN"} for pending in cases[index + 1:])
                    break
        # 实际启用追踪时等待已排队回调；等待结束仍不代表 Trace 内容已经人工验收。
        if trace_enabled:
            from langchain_core.tracers.langchain import wait_for_all_tracers
            wait_for_all_tracers()
        if args.evidence:
            args.evidence.parent.mkdir(parents=True, exist_ok=True)
            # x 模式同时防止 exists 检查后出现的并发覆盖。
            with args.evidence.open("x", encoding="utf-8") as stream:
                json.dump(records, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        # 命令退出码与场景结果一致，便于终端、测试或 CI 判断成功与失败。
        return 0 if succeeded else 1
    except Exception as exc:
        # 不直接输出 SDK 异常，避免请求信息或密钥进入终端日志。
        print(f"FAIL: {type(exc).__name__}；实验未通过，请检查配置、预算或协议。")
        return 1


# 脚本直接运行才解析 CLI；被测试模块导入时不会自动执行实验。
if __name__ == "__main__":
    raise SystemExit(main())
