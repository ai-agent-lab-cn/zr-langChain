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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

Category = Literal["account", "payment", "logistics", "manual"]
Status = Literal["triaged", "needs_human"]

CASES = {
    "account": ("我的账号无法登录，一直提示密码错误。", "account"),
    "payment": ("订单被重复扣款了，需要查询退款进度。", "payment"),
    "logistics": ("包裹已发货，但是物流信息三天没有更新。", "logistics"),
    "manual": ("我想申请定制企业报表并升级 VIP 套餐。", "manual"),
}

BRANCH_NODE = {
    "account": "handle_account",
    "payment": "handle_payment",
    "logistics": "handle_logistics",
    "manual": "handle_manual",
}


class TicketState(TypedDict, total=False):
    """Graph 各节点共享的数据；steps 通过 reducer 追加而非覆盖。"""

    ticket: str
    category: Category
    reason: str
    reply: str
    status: Status
    steps: Annotated[list[str], operator.add]


class RouteDecision(BaseModel):
    """在线分类节点的模型输出约束。"""

    category: Category = Field(description="账户 account、支付 payment、物流 logistics，否则 manual")
    reason: str = Field(description="简短的工单分类依据；未能确定时说明为何转人工")


SYSTEM_PROMPT = (
    "你是教学版工单分类器，只返回要求的结构化数据。"
    "分类限定为 account（登录/账户）、payment（支付/退款）、logistics（物流/配送）和 manual。"
    "输入缺少明确依据、多类重叠或不属于这三类时，必须归入 manual。"
    "不执行支付或订单操作，不编造工单状态；不要将回复模板当作真实的业务处理结果。"
)


def rule_route(text: str) -> tuple[Category, str]:
    """确定性替身：只用于离线教学测试，不冒充真实语义分类模型。"""

    keys = {
        "account": ("账号", "账户", "登录", "密码", "注册"),
        "payment": ("支付", "退款", "扣款", "付款", "账单"),
        "logistics": ("物流", "快递", "包裹", "配送", "收货"),
    }
    matched = [name for name, terms in keys.items() if any(word in text for word in terms)]
    if len(matched) == 1:
        return matched[0], "离线关键词规则命中单一类别"
    if len(matched) > 1:
        return "manual", "离线关键词规则命中多个类别，转人工复核"
    return "manual", "离线规则未发现明确类别，转人工复核"


def build_workflow(model: ChatGoogleGenerativeAI | None = None):
    """构建并编译同一张图：仅分类节点根据 model 是否提供而变化。"""

    classifier = model.with_structured_output(RouteDecision) if model is not None else None

    async def classify(state: TicketState, config: RunnableConfig) -> dict:
        ticket = state["ticket"].strip()
        if not ticket:
            raise ValueError("工单内容不可为空")
        if classifier is None:
            category, reason = rule_route(ticket)
        else:
            answer = await classifier.ainvoke(
                [("system", SYSTEM_PROMPT), ("human", f"请分类该工单：{ticket}")],
                config=config,
            )
            # with_structured_output(Pydantic) 应返回 RouteDecision；失败要显式报告。
            if not isinstance(answer, RouteDecision):
                raise TypeError(f"分类结果不是 RouteDecision：{type(answer).__name__}")
            category, reason = answer.category, answer.reason
        return {"category": category, "reason": reason, "steps": ["classify"]}

    def select_branch(state: TicketState) -> Category:
        return state["category"]

    def handle_account(state: TicketState) -> dict:
        return {
            "reply": "已分类为账户问题。建议先核实登录提示和账户信息，由客服确认后处理。",
            "steps": ["handle_account"],
        }

    def handle_payment(state: TicketState) -> dict:
        return {
            "reply": "已分类为支付问题。建议由客服核对订单与支付记录，不收集完整银行卡信息。",
            "steps": ["handle_payment"],
        }

    def handle_logistics(state: TicketState) -> dict:
        return {
            "reply": "已分类为物流问题。建议核对订单号与运单状态，再由客服跟进。",
            "steps": ["handle_logistics"],
        }

    def handle_manual(state: TicketState) -> dict:
        return {
            "reply": "无法可靠归入账户、支付或物流类别，已标记为待人工复核（演示流程，未创建真实工单）。",
            "steps": ["handle_manual"],
        }

    def finalize(state: TicketState) -> dict:
        category = state["category"]
        selected = BRANCH_NODE[category]
        steps = state.get("steps", [])
        if not state.get("reply"):
            raise ValueError("处理节点没有产出回复")
        if selected not in steps or any(
            node in steps for node in set(BRANCH_NODE.values()) - {selected}
        ):
            raise ValueError("Graph 经过了错误的处理分支")
        status: Status = "needs_human" if category == "manual" else "triaged"
        return {"status": status, "steps": ["finalize"]}

    builder = StateGraph(TicketState)
    builder.add_node("classify", classify)
    builder.add_node("handle_account", handle_account)
    builder.add_node("handle_payment", handle_payment)
    builder.add_node("handle_logistics", handle_logistics)
    builder.add_node("handle_manual", handle_manual)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", select_branch, BRANCH_NODE)
    for node_name in BRANCH_NODE.values():
        builder.add_edge(node_name, "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


def verify(case: str, result: TicketState, expected: Category) -> list[str]:
    issues = []
    category = result.get("category")
    if category != expected:
        issues.append(f"分类错误：预期 {expected}，实际 {category}")
    expected_status = "needs_human" if expected == "manual" else "triaged"
    if result.get("status") != expected_status:
        issues.append(f"状态错误：预期 {expected_status}，实际 {result.get('status')}")
    path = result.get("steps", [])
    if path != ["classify", BRANCH_NODE.get(category, "INVALID"), "finalize"]:
        issues.append(f"实际节点路径不正确：{path}")
    if not result.get("reply"):
        issues.append("回复为空")
    if case == "manual" and "人工" not in result.get("reply", ""):
        issues.append("兜底分支未标明人工复核")
    return issues


async def execute_case(graph, case: str, *, stream: bool = False) -> tuple[TicketState, list[str]]:
    ticket, expected = CASES[case]
    config = {
        "run_name": f"chapter07-{case}",
        "tags": ["zrLangChain", "chapter-07", "langgraph", case],
        "metadata": {"chapter": "07-langgraph-workflow", "scenario": case},
    }
    initial: TicketState = {"ticket": ticket, "steps": []}
    print(f"\n============ workflow / {case} ============")
    print("工单:", ticket)
    if stream:
        final = None
        async for value in graph.astream(initial, config=config, stream_mode="values"):
            final = value
            print("State Stream:", value.get("steps", []))
        if final is None:
            raise RuntimeError("Graph 没有发出状态事件")
        result: TicketState = final
    else:
        result = await graph.ainvoke(initial, config=config)

    print("分类:", result.get("category"))
    print("依据:", result.get("reason"))
    print("执行路径:", " → ".join(result.get("steps", [])))
    print("状态:", result.get("status"))
    print("处理建议:", result.get("reply"))
    issues = verify(case, result, expected)
    print("验收:", "✅ 通过" if not issues else "❌ " + "; ".join(issues))
    return result, issues


async def self_test() -> bool:
    """实际运行 LangGraph 的 4 条路由和状态流；0 次 Gemini API 调用。"""

    graph = build_workflow(model=None)
    ok = True
    for case in CASES:
        _, issues = await execute_case(graph, case)
        ok = ok and not issues

    ambiguous = "账号无法登录，同时支付失败，还要咨询退款。"
    ambiguous_result = await graph.ainvoke({"ticket": ambiguous, "steps": []})
    ambiguous_ok = (
        ambiguous_result["category"] == "manual"
        and ambiguous_result["status"] == "needs_human"
        and ambiguous_result["steps"] == ["classify", "handle_manual", "finalize"]
    )
    print("\n多类别歧义转人工:", "✅" if ambiguous_ok else "❌")
    ok = ok and ambiguous_ok

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
        wait_for_all_tracers()


if __name__ == "__main__":
    raise SystemExit(main())
