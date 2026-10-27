"""A/B/D：真实 LangGraph 状态编排 + 规则路由 + Fake 业务工具。

真实的是 StateGraph、ToolNode、消息 reducer、节点路径；不是 LLM 推理或多 Agent 子模型。
所有工具只返回固定教学结果，不调用业务服务、LLM 或 HTTP API。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from evaluation_core import Observation


class TicketState(TypedDict, total=False):
    """图状态：plan/cursor 为调度信息，messages 为真实 AI/Tool 消息。"""

    case_id: str
    ticket: str
    version: str
    plan: list[str]
    cursor: int
    messages: Annotated[list[BaseMessage], add_messages]
    steps: Annotated[list[str], operator.add]
    status: str
    final_reply: str


@tool
def lookup_account(ticket: str) -> str:
    """Fake Account 工具：返回固定的账号核验建议；不读取真实账号。"""
    return "账号专员：先使用官方密码重置渠道核验身份，勿索取完整密码。"


@tool
def lookup_logistics(ticket: str) -> str:
    """Fake Logistics 工具：返回固定的物流建议；不查询真实运单。"""
    return "物流专员：检查运单和最后一次轨迹，再由客服跟进。"


@tool
def lookup_business(ticket: str) -> str:
    """Fake Business 工具：返回固定的套餐建议；不收费或升级套餐。"""
    return "商务专员：先核对企业套餐要求，人工确认报价和权限。"


TOOLS = (lookup_account, lookup_logistics, lookup_business)


def choose_plan(ticket: str, version: str) -> list[str]:
    """模拟 Supervisor 的确定性路由策略，baseline 故意遗漏第二类需求。

    这里不是 LLM Agent；故意弱化的 baseline 仅为 D 组回归对照。
    """
    plan: list[str] = []
    if any(word in ticket for word in ("账号", "登录", "密码")):
        plan.append("lookup_account")
    if any(word in ticket for word in ("物流", "包裹", "运单")):
        plan.append("lookup_logistics")
    if any(word in ticket for word in ("VIP", "套餐", "商务")):
        plan.append("lookup_business")
    if version == "baseline":
        return plan[:1]
    if version == "candidate":
        return plan
    raise ValueError(f"未知路由版本：{version}")


def build_ticket_graph():
    """构建真实的 LangGraph 调度图（无 LLM 请求）。

    返回编译后的 Graph；路由由规则替身执行，工具由正式 ToolNode 执行。
    异常：工具执行失败或消息协议不满足会让 Graph 失败，而不会吞掉错误。
    关键边界：ToolMessage 必须由 ToolNode 产生，不能把 Fake 工具回复当成 Agent 模型能力。
    """
    builder = StateGraph(TicketState)

    def route(state: TicketState) -> dict[str, Any]:
        plan = choose_plan(state["ticket"], state["version"])
        return {"plan": plan, "cursor": 0, "steps": ["route"]}

    def supervisor(state: TicketState) -> dict[str, Any]:
        """确定性调度替身。发起真实 AIMessage.tool_calls，让 ToolNode 执行工具。"""
        cursor = state["cursor"]
        name = state["plan"][cursor]
        call_id = f"{state['case_id']}-tool-{cursor}"
        message = AIMessage(
            content="", tool_calls=[{
                "name": name,
                "args": {"ticket": state["ticket"]},
                "id": call_id,
                "type": "tool_call",
            }],
        )
        return {"messages": [message], "steps": [f"supervisor:{name}"]}

    def advance(state: TicketState) -> dict[str, Any]:
        """检查最后一条工具结果，并让游标前进；绝不只凭回答推断工具成功。"""
        last = state["messages"][-1]
        if not isinstance(last, ToolMessage) or last.status == "error":
            raise RuntimeError("缺失有效 ToolMessage 或工具执行失败")
        return {"cursor": state["cursor"] + 1, "steps": ["tool_result"]}

    def next_step(state: TicketState) -> str:
        return "supervisor" if state["cursor"] < len(state["plan"]) else "finalize"

    def finalize(state: TicketState) -> dict[str, Any]:
        """只生成教学结果；无付款、修改账号等业务副作用。"""
        replies = [m.content for m in state.get("messages", []) if isinstance(m, ToolMessage)]
        return {
            "status": "advice_ready" if replies else "needs_review",
            "final_reply": "；".join(str(x) for x in replies),
            "steps": ["finalize"],
        }

    builder.add_node("route", route)
    builder.add_node("supervisor", supervisor)
    builder.add_node("execute_tool", ToolNode(list(TOOLS), handle_tool_errors=False))
    builder.add_node("advance", advance)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", lambda s: "supervisor" if s["plan"] else "finalize")
    builder.add_edge("supervisor", "execute_tool")
    builder.add_edge("execute_tool", "advance")
    builder.add_conditional_edges("advance", next_step)
    builder.add_edge("finalize", END)
    return builder.compile()


def run_ticket(graph: Any, case_id: str, ticket: str, version: str) -> Observation:
    """实际 invoke 图并提取观测，不输入 GoldenCase 的参考标签。

    参数：graph 为已编译的真实图；case_id/ticket 为被测样本输入；version
    是应用路由策略版本。返回 Observation（无原始工单/回复），模型用量置 None。
    ToolNode 失败时异常向外传播，不能填造一次“正常执行”。
    """
    state = graph.invoke(
        {"case_id": case_id, "ticket": ticket, "version": version,
         "messages": [], "steps": [], "cursor": 0},
        config={"recursion_limit": 24},
    )
    calls: list[tuple[str, str]] = []
    results: list[str] = []
    pending: dict[str, str] = {}
    seen: set[str] = set()
    protocol_valid = True
    for message in state["messages"]:
        if isinstance(message, AIMessage):
            if pending:
                protocol_valid = False
            for call in message.tool_calls:
                call_id, name = call.get("id") or "", call["name"]
                if not call_id or call_id in seen:
                    protocol_valid = False
                seen.add(call_id)
                pending[call_id] = name
                calls.append((name, call_id))
        elif isinstance(message, ToolMessage):
            if (message.tool_call_id not in pending or message.status != "success"
                    or (message.name and pending.get(message.tool_call_id) != message.name)):
                protocol_valid = False
            pending.pop(message.tool_call_id, None)
            results.append(str(message.tool_call_id))
        elif pending:
            protocol_valid = False
    return Observation(
        case_id=case_id, status=state["status"],
        tool_calls=tuple(calls), tool_results=tuple(results),
        steps=tuple(state["steps"]), effect_count=0,
        usage_metadata=None,
        message_protocol_valid=protocol_valid and not pending,
    )
