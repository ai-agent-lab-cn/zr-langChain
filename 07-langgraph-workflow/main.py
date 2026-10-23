"""LangChain 入门实战（七）：LangGraph 工作流编排。

在 zr-langChain 仓库根目录运行：
    uv add 'langgraph>=1.0,<2'
    uv run python 07-langgraph-workflow/main.py --self-test
    uv run python 07-langgraph-workflow/main.py --case all --offline
    uv run python 07-langgraph-workflow/main.py --case all
    uv run python 07-langgraph-workflow/main.py --case account --offline --stream

--self-test / --offline：实际执行 StateGraph，但不调用 Gemini。
在线模式：Gemini 只对工单分类；分支和处理逻辑由 LangGraph 显式控制。
"""

import argparse
import asyncio
import operator
import os
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# 【运行环境】所有章节共用仓库根目录的 .env，避免每章复制密钥文件。
# __file__ 位于 07-langgraph-workflow/，其 parent.parent 才是 zr-langChain/。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# 【业务类型】Literal 限制 Python 类型检查阶段的可选值；运行时的模型输出约束由 Pydantic 负责。
# manual 代表无法明确归类，需要进入模拟人工复核分支。
Category = Literal["account", "payment", "logistics", "manual"]
Status = Literal["triaged", "needs_human"]

# 【验收数据】四个固定样例对应四条条件路由，避免在线测试只看模型回答是否流畅。
# 每条用例保存（用户工单，预期类别）；这里使用的是虚构的教学数据。
CASES = {
    "account": ("我的账号无法登录，一直提示密码错误。", "account"),
    "payment": ("订单被重复扣款了，需要查询退款进度。", "payment"),
    "logistics": ("包裹已发货，但是物流信息三天没有更新。", "logistics"),
    "manual": ("我想申请定制企业报表并升级 VIP 套餐。", "manual"),
}
# 【条件边映射】键是 select_branch 的返回值，值是 add_node 注册的节点名称。
# 增加类别时必须同步维护：类别定义、处理节点、映射关系和测试用例。
BRANCH_NODE = {
    "account": "handle_account",
    "payment": "handle_payment",
    "logistics": "handle_logistics",
    "manual": "handle_manual",
}

# 【核心 API：State】StateGraph(TicketState) 将此结构作为各节点共享的状态契约。
# 节点通常返回状态的“增量字典”，而不是自己修改、重建整份 State。
# total=False 仅表示类型层面字段可缺省，不等同于运行时输入校验。
class TicketState(TypedDict, total=False):
    """Graph 各节点共享的数据；steps 通过 reducer 追加而非覆盖。"""

    ticket: str  # 输入：用户原始工单；classify 节点读取它。
    category: Category  # 分类结果，决定从 classify 走向哪条条件边。
    reason: str  # Gemini / 离线规则提供的分类依据，方便解释决策。
    reply: str  # 处理节点提供的教学建议，不代表实际业务已执行。
    status: Status  # finalize 设置 triaged 或 needs_human。
    # 【Reducer】没有 reducer 的字段默认被新值覆盖；这里使用 operator.add 合并列表。
    # 例如 ["classify"] + ["handle_account"] + ["finalize"] 得到完整执行轨迹。
    # 教学示例是串行单路径；并行节点不应依赖该列表的稳定顺序。
    steps: Annotated[list[str], operator.add]

# 【结构化输出】Pydantic 用于约束模型在 classify 节点的输出格式。
# 格式合法不代表语义一定正确，因此在线测试仍需检验预期分类。
class RouteDecision(BaseModel):
    """在线分类节点的模型输出约束。"""

    category: Category = Field(description="账户 account、支付 payment、物流 logistics，否则 manual")
    reason: str = Field(description="简短的工单分类依据；未能确定时说明为何转人工")

# 【模型职责边界】Gemini 只返回 category / reason；不会直接执行付款、退款或物流操作。
# 分支可达性由 Graph 的 add_conditional_edges 控制，而不是由 Prompt 单独保证。
SYSTEM_PROMPT = (
    "你是教学版工单分类器，只返回要求的结构化数据。"
    "分类限定为 account（登录/账户）、payment（支付/退款）、logistics（物流/配送）和 manual。"
    "输入缺少明确依据、多类重叠或不属于这三类时，必须归入 manual。"
    "不执行支付或订单操作，不编造工单状态；不要将回复模板当作真实的业务处理结果。"
)

# 【离线分类替身】为了在不请求 Gemini API 的条件下测试真正的 StateGraph 执行。
# 该规则只做关键词匹配，不能代替语义分类，也不是生产级意图识别。
def rule_route(text: str) -> tuple[Category, str]:
    """确定性替身：只用于离线教学测试，不冒充真实语义分类模型。"""

    keys = {
        "account": ("账号", "账户", "登录", "密码", "注册"),
        "payment": ("支付", "退款", "扣款", "付款", "账单"),
        "logistics": ("物流", "快递", "包裹", "配送", "收货"),
    }
    # any(...)：某一类至少命中一个关键词；matched 同时有多类时拒绝强行单选。
    matched = [name for name, terms in keys.items() if any(word in text for word in terms)]
    if len(matched) == 1:
        return matched[0], "离线关键词规则命中单一类别"
    if len(matched) > 1:
        return "manual", "离线关键词规则命中多个类别，转人工复核"
    return "manual", "离线规则未发现明确类别，转人工复核"

# 【工作流构建入口】将第七章所有 State、Node、Edge 与条件路由组织到同一张图。
# model=None 时走规则分类；传入 Gemini 时仅替换 classify 内部的分类方式，图结构不变。
def build_workflow(model: ChatGoogleGenerativeAI | None = None):
    """构建并编译同一张图：仅分类节点根据 model 是否提供而变化。"""
    # 【LangChain API】with_structured_output(RouteDecision) 让 Gemini 输出可解析的 Pydantic 对象。
    # 注意：这里还没有请求模型；真实请求发生在 classify 中的 classifier.ainvoke(...)。
    classifier = model.with_structured_output(RouteDecision) if model is not None else None
    # 【Node 1：智能分类】读取 ticket，写入 category / reason / steps。
    # config 是 LangGraph 传入的运行配置：将 tags / metadata 等传给子模型调用，有利于 LangSmith 跟踪。
    async def classify(state: TicketState, config: RunnableConfig) -> dict:
        ticket = state["ticket"].strip()
        if not ticket:
            raise ValueError("工单内容不可为空")
        # 离线/在线共用此节点和后续条件路由，所以能对比“分类方法变化”而不改变 Graph。
        if classifier is None:
            category, reason = rule_route(ticket)
        else:
            # 【模型实际调用】ainvoke 是异步请求；系统提示词限定四种类别和人工兜底规则。
            # config 向下传递，使这次模型调用仍在父 Graph Trace 中。
            answer = await classifier.ainvoke(
                [("system", SYSTEM_PROMPT), ("human", f"请分类该工单：{ticket}")],
                config=config,
            )
            # 结构化解析若意外返回别的类型，立即报错，避免用错误数据继续路由。
            if not isinstance(answer, RouteDecision):
                raise TypeError(f"分类结果不是 RouteDecision：{type(answer).__name__}")
            category, reason = answer.category, answer.reason
        # 返回 State 增量：category/reason 更新普通字段，steps 使用 Reducer 追加。
        return {"category": category, "reason": reason, "steps": ["classify"]}
    # 【条件边函数】不是 Graph Node，不会追加 steps 或写业务 State。
    # add_conditional_edges 将它返回的 category 作为 BRANCH_NODE 的键，得到下一节点名称。
    def select_branch(state: TicketState) -> Category:
        return state["category"]
    # 【处理节点】四个分支只做教学建议；不写入真实账户、订单或支付系统。
    # 当前示例的处理函数无需读取 state，仍保留相同函数签名以符合 Node 输入约定。
    def handle_account(state: TicketState) -> dict:
        return {
            "reply": "已分类为账户问题。建议先核实登录提示和账户信息，由客服确认后处理。",
            "steps": ["handle_account"],
        }
    # 支付分支只输出客服核查建议，不进行扣款、退款等真实业务写操作。
    def handle_payment(state: TicketState) -> dict:
        return {
            "reply": "已分类为支付问题。建议由客服核对订单与支付记录，不收集完整银行卡信息。",
            "steps": ["handle_payment"],
        }
    # 物流分支仅返回下一步建议，不执行真实物流查询。
    def handle_logistics(state: TicketState) -> dict:
        return {
            "reply": "已分类为物流问题。建议核对订单号与运单状态，再由客服跟进。",
            "steps": ["handle_logistics"],
        }
    # 【重要边界】handle_manual 不是 LangGraph 的 interrupt()；该分支不会暂停执行。
    # 它只记录模拟人工复核建议，随后照常进入 finalize；真正 HITL 留待第八章。
    def handle_manual(state: TicketState) -> dict:
        return {
            "reply": "无法可靠归入账户、支付或物流类别，已标记为待人工复核（演示流程，未创建真实工单）。",
            "steps": ["handle_manual"],
        }
    # 【末端校验节点】验证本轮确实经过了唯一、匹配 category 的处理分支。
    # 对于 manual，将状态标记 needs_human；其余类别只是已分类 triaged，不等于工单完成。
    def finalize(state: TicketState) -> dict:
        category = state["category"]
        selected = BRANCH_NODE[category]
        steps = state.get("steps", [])
        if not state.get("reply"):
            raise ValueError("处理节点没有产出回复")
        # 检查两种错误：应该走的节点没走到 / 意外走到了其他处理分支。
        if selected not in steps or any(
            node in steps for node in set(BRANCH_NODE.values()) - {selected}
        ):
            raise ValueError("Graph 经过了错误的处理分支")
        status: Status = "needs_human" if category == "manual" else "triaged"
        return {"status": status, "steps": ["finalize"]}

    # 【LangGraph API ①】StateGraph 定义图的 State Schema；目前只是构建器，不会运行。
    builder = StateGraph(TicketState)
    # 【LangGraph API ②】add_node(name, callable) 注册可执行节点，名称必须唯一。
    # 注意：注册节点并不决定执行顺序，执行路径要通过 add_edge / add_conditional_edges 指定。
    builder.add_node("classify", classify)
    builder.add_node("handle_account", handle_account)
    builder.add_node("handle_payment", handle_payment)
    builder.add_node("handle_logistics", handle_logistics)
    builder.add_node("handle_manual", handle_manual)
    builder.add_node("finalize", finalize)

    # 【LangGraph API ③】add_edge(START, "classify")：每次执行固定从 classify 开始。
    builder.add_edge(START, "classify")
    # 【LangGraph API ④】条件边：分类完成后读取 State.category，只选择 BRANCH_NODE 映射中的一个分支。
    # select_branch 本身是路由函数，不是有独立 State 更新的业务 Node。
    builder.add_conditional_edges("classify", select_branch, BRANCH_NODE)
    # 【LangGraph API ⑤】四个分支完成后统一汇入 finalize，然后到 END。
    for node_name in BRANCH_NODE.values():
        builder.add_edge(node_name, "finalize")
    builder.add_edge("finalize", END)
    # 【LangGraph API ⑥】compile() 生成可通过 ainvoke / astream 运行的 CompiledStateGraph。
    # 本章没有传 checkpointer，因此不会提供跨请求暂停/恢复能力。
    return builder.compile()

# 【验收逻辑】验证真实 Graph 返回的最终 State，而非只核对打印出来的文本。
# 只测试固定教学样例的已知预期，不代表生产分类准确率。
def verify(case: str, result: TicketState, expected: Category) -> list[str]:
    issues = []
    category = result.get("category")
    if category != expected:
        issues.append(f"分类错误：预期 {expected}，实际 {category}")
    expected_status = "needs_human" if expected == "manual" else "triaged"
    if result.get("status") != expected_status:
        issues.append(f"状态错误：预期 {expected_status}，实际 {result.get('status')}")
    path = result.get("steps", [])
    # 必须严格匹配完整路径：分类 → 唯一的处理分支 → finalize。
    if path != ["classify", BRANCH_NODE.get(category, "INVALID"), "finalize"]:
        issues.append(f"实际节点路径不正确：{path}")
    if not result.get("reply"):
        issues.append("回复为空")
    if case == "manual" and "人工" not in result.get("reply", ""):
        issues.append("兜底分支未标明人工复核")
    return issues

# 【统一实验入口】同一个 Graph 同时支持 ainvoke 一次性执行、astream 分阶段查看 State。
# 两条路径使用相同验收函数，因此 Stream 模式不是“另一套模拟运行”。
async def execute_case(graph, case: str, *, stream: bool = False) -> tuple[TicketState, list[str]]:
    ticket, expected = CASES[case]
    # 【LangSmith】run_name 便于按 chapter07-account 等名称查找根 Trace；tags/metadata 标记场景。
    config = {
        "run_name": f"chapter07-{case}",
        "tags": ["zrLangChain", "chapter-07", "langgraph", case],
        "metadata": {"chapter": "07-langgraph-workflow", "scenario": case},
    }
    # 初始化状态。步骤列表从空开始，随后由各节点的 Reducer 累加。
    initial: TicketState = {"ticket": ticket, "steps": []}
    print(f"\n============ workflow / {case} ============")
    print("工单:", ticket)
    # 【LangGraph Streaming】values 表示每个阶段后的完整 State 快照，不是模型逐字输出。
    if stream:
        final = None
        # astream 按节点执行节奏产出状态；最后一次快照作为实验结果进行统一验收。
        async for value in graph.astream(initial, config=config, stream_mode="values"):
            final = value
            print("State Stream:", value.get("steps", []))
        if final is None:
            raise RuntimeError("Graph 没有发出状态事件")
        result: TicketState = final
    else:
        # 【LangGraph 执行 API】ainvoke 异步运行到 END 并返回最终合并后的 State。
        result = await graph.ainvoke(initial, config=config)

    print("分类:", result.get("category"))
    print("依据:", result.get("reason"))
    print("执行路径:", " → ".join(result.get("steps", [])))
    print("状态:", result.get("status"))
    print("处理建议:", result.get("reply"))
    issues = verify(case, result, expected)
    print("验收:", "✅ 通过" if not issues else "❌ " + "; ".join(issues))
    return result, issues

# 【离线自检】真正编译并运行 StateGraph，但只使用确定性分类替身，不消耗 Gemini Token。
# 验证：四条路由、跨类别歧义兜底、values 流式状态合并。
async def self_test() -> bool:
    """实际运行 LangGraph 的 4 条路由和状态流；0 次 Gemini API 调用。"""

    graph = build_workflow(model=None)
    ok = True
    for case in CASES:
        _, issues = await execute_case(graph, case)
        ok = ok and not issues

    # 额外反例：同一输入同时触发账户、支付，离线规则必须走 manual，避免误选单一路径。
    ambiguous = "账号无法登录，同时支付失败，还要咨询退款。"
    ambiguous_result = await graph.ainvoke({"ticket": ambiguous, "steps": []})
    ambiguous_ok = (
        ambiguous_result["category"] == "manual"
        and ambiguous_result["status"] == "needs_human"
        and ambiguous_result["steps"] == ["classify", "handle_manual", "finalize"]
    )
    print("\n多类别歧义转人工:", "✅" if ambiguous_ok else "❌")
    ok = ok and ambiguous_ok

    # 检查 Reducer 追加和 State Streaming 的最终结果，而不是仅打印流事件。
    seen = []
    async for value in graph.astream(
        {"ticket": CASES["account"][0], "steps": []}, stream_mode="values"
    ):
        seen.append(value.get("steps", []))
    stream_ok = len(seen) >= 3 and seen[-1] == ["classify", "handle_account", "finalize"]
    print("State 流式观察:", "✅" if stream_ok else "❌", "最终路径:", seen[-1] if seen else [])
    ok = ok and stream_ok
    print("\nself-test:", "✅ 全部通过（真实 LangGraph，0 次 Gemini 调用）" if ok else "❌ 未通过")
    return ok

# 【运行模式选择】self-test/offline 完全不初始化模型；在线模式才从根 .env 读取密钥。
async def main_async(args: argparse.Namespace) -> int:
    if args.self_test:
        return 0 if await self_test() else 1

    model = None
    if not args.offline:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("缺少 GEMINI_API_KEY / GOOGLE_API_KEY，请检查项目根目录 .env")
        model = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            api_key=api_key,
        )
    # 离线与在线都通过同一 build_workflow 建图；唯一变化是 classify 的内部实现。
    graph = build_workflow(model)
    print("运行模式:", "离线规则分类（不调用 Gemini）" if args.offline else "Gemini 结构化分类")
    targets = list(CASES) if args.case == "all" else [args.case]
    success = True
    for case in targets:
        try:
            _, issues = await execute_case(graph, case, stream=args.stream)
            success = success and not issues
        except Exception as exc:
            success = False
            print(f"❌ {case} 运行异常：{type(exc).__name__}: {exc}")
    print("\n========== 总体验收 ==========")
    print("✅ 所选实验全部通过" if success else "❌ 存在未通过的实验")
    return 0 if success else 1

# 【CLI】解析命令行，统一打印错误并用退出码 0/1 表示验收是否通过。
def main() -> int:
    parser = argparse.ArgumentParser(description="Chapter 07: LangGraph workflow")
    parser.add_argument("--self-test", action="store_true", help="4 个分支 + 歧义 + 流式状态；不调用 Gemini")
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument("--offline", action="store_true", help="使用关键词分类替身，但仍执行实际 LangGraph")
    parser.add_argument("--stream", action="store_true", help="使用 LangGraph stream_mode=values 输出完整 State 演进")
    args = parser.parse_args()
    try:
        return asyncio.run(main_async(args))
    except Exception as exc:
        print(f"❌ 运行失败：{type(exc).__name__}: {exc}")
        return 1
    finally:
        # 在进程退出前刷新 LangSmith 异步追踪，减少 Trace 因提前退出而丢失。
        wait_for_all_tracers()


if __name__ == "__main__":
    raise SystemExit(main())
