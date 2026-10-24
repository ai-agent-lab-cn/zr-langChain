"""第09章：Supervisor / Agent as Tool / 真实 LangGraph Handoff。
离线仅替换模型决策，Graph、Tool、Command 和 checkpoint 都真实执行。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import operator
import os
from pathlib import Path
from typing import Annotated, Literal, TypedDict
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ("account_agent", "logistics_agent")
PROMPTS = {
    "account_agent": "你是账号支持专员。简短给出核实登录提示和账号状态的建议，不索取密码，不声称已操作。",
    "logistics_agent": "你是物流支持专员。简短建议核实运单和物流节点，不声称已联系快递或完成操作。",
}

# State TypedDict 允许可选字段，方便在 Graph 中增量更新。
class State(TypedDict, total=False):
    # add_messages 按消息 ID 合并；ToolMessage.tool_call_id 对应 AI tool_calls 的 id。
    messages: Annotated[list[AnyMessage], add_messages]
    steps: Annotated[list[str], operator.add]
    active_agent: str

# 提取用户输入的文本
def request_text(state):
    text = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("请求不能为空")
    return text

# 离线路由器：根据关键词确定领域，或返回 None 让用户澄清。
def offline_route(text):
    """确定性测试替身；含多个领域或未知请求时请用户澄清。"""
    hits = [name for name, words in (("account_agent", ("账号", "登录")), ("logistics_agent", ("物流", "包裹")))
            if any(word in text for word in words)]
    return hits[0] if len(hits) == 1 else None

# 构建子 Agent 的 StateGraph，独立消息上下文，在线由自己的 LLM 生成专业回复。
def build_specialist(name, model=None):
    """子 Agent 有独立消息上下文；在线由自己的 LLM 生成专业回复。"""
    async def respond(state: State, config: RunnableConfig):
        text = request_text(state)
        answer = (await model.ainvoke([SystemMessage(content=PROMPTS[name]), HumanMessage(content=text)], config)
                  if model is not None else AIMessage(content=PROMPTS[name] + "（离线教学回复）"))
        if answer.tool_calls or not answer.content:
            raise ValueError("专员必须返回非空最终回复")
        return {"messages": [answer]}
    graph = StateGraph(State)
    graph.add_node(name, respond)
    graph.add_edge(START, name)
    graph.add_edge(name, END)
    return graph.compile(name=name)

# 构建委派工具，Supervisor 调用子 Agent 的 StateGraph 处理任务。
def specialist_tool(name, model=None):
    agent = build_specialist(name, model)

    @tool(name, description=f"将对应领域的任务交给{name}专业 Agent，返回处理建议。")
    async def delegate(task: str, config: RunnableConfig) -> str:
        # 只传必要任务，不把 Supervisor 的全部对话泄漏给子 Agent。
        result = await agent.ainvoke({"messages": [HumanMessage(content=task)]}, config)
        return result["messages"][-1].content
    return delegate

# Supervisor 作为顶层 Agent，调用子 Agent 工具处理任务，整合回复。
def build_supervisor(model=None):
    tools = [specialist_tool(name, model) for name in AGENTS]
    router = model.bind_tools(tools, parallel_tool_calls=False) if model is not None else None

    async def supervisor(state: State, config: RunnableConfig):
        text = request_text(state)
        if router is not None:
            answer = await router.ainvoke([SystemMessage(content=
                "你是 Supervisor。账号或物流问题必须调用对应专业Agent工具；工具返回后直接整合回复，不重复调用。"
                "每次最多一个工具调用；未知或多领域歧义请澄清。"), *state["messages"]], config)
        elif isinstance(state["messages"][-1], ToolMessage):
            answer = AIMessage(content="Supervisor 汇总：" + state["messages"][-1].content)
        else:
            name = offline_route(text)
            answer = (AIMessage(content="", tool_calls=[{"name": name, "args": {"task": text}, "id": uuid4().hex}])
                      if name else AIMessage(content="请明确账号或物流领域。"))
        return {"messages": [answer], "steps": ["supervisor"], "active_agent": "supervisor"}
    # 路由器：Supervisor 只允许单个子 Agent 工具调用，或直接结束。
    def route(state):
        calls = state["messages"][-1].tool_calls
        if len(calls) > 1 or any(c["name"] not in AGENTS for c in calls):
            raise ValueError("无效或并行委派")
        return "delegate" if calls else END

    tool_node = ToolNode(tools, handle_tool_errors=False)
    # delegate 节点仅调用子 Agent 工具，返回结果给 Supervisor。
    async def delegate(state: State, config: RunnableConfig):
        result = await tool_node.ainvoke(state, config)
        return {"messages": result["messages"], "steps": [state["messages"][-1].tool_calls[0]["name"]]}
    # Supervisor 节点和 delegate 节点共享同一 StateGraph，便于在同一上下文中整合消息。
    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor)
    graph.add_node("delegate", delegate)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route)
    graph.add_edge("delegate", "supervisor")
    return graph.compile(name="chapter09-supervisor")

# 构建移交工具，triage 节点调用 transfer 工具将控制权移交给子 Agent。
def build_handoff(model=None):
    # transfer 工具声明提供给模型；控制权移交由 triage 节点返回的 Command 执行。
    @tool
    def transfer_to_account() -> str:
        """将当前会话移交账号专员。"""
        return "移交账号专员"

    @tool
    def transfer_to_logistics() -> str:
        """将当前会话移交物流专员。"""
        return "移交物流专员"

    targets = {"transfer_to_account": "account_agent", "transfer_to_logistics": "logistics_agent"}
    router = model.bind_tools([transfer_to_account, transfer_to_logistics], parallel_tool_calls=False) if model is not None else None
    # triage 节点根据用户请求调用 transfer 工具，返回 Command 改变下一节点。
    async def triage(state: State, config: RunnableConfig) -> Command[Literal["account_agent", "logistics_agent", "__end__"]]:
        text = request_text(state)
        if router is not None:
            answer = await router.ainvoke([SystemMessage(content=
                "你是分诊Agent。账号问题必须调用transfer_to_account，物流问题必须调用transfer_to_logistics。"
                "每次仅调用一个工具。未知或歧义请求澄清。移交后由专员直接回复。"), *state["messages"]], config)
        else:
            name = offline_route(text)
            transfer = next((k for k, v in targets.items() if v == name), None)
            answer = (AIMessage(content="", tool_calls=[{"name": transfer, "args": {}, "id": uuid4().hex}])
                      if transfer else AIMessage(content="请明确账号或物流领域。"))
        if not answer.tool_calls:
            return Command(goto=END, update={"messages": [answer], "steps": ["triage"], "active_agent": "triage"})
        calls = answer.tool_calls
        if len(calls) != 1 or calls[0]["name"] not in targets or calls[0]["args"]:
            raise ValueError("无效移交调用")
        call = calls[0]
        target = targets[call["name"]]
        # 在同一 StateGraph 内移交，不用 Command.PARENT；goto 真实改变下一节点。
        # AIMessage 和对应 ToolMessage 同时写回，防止下个模型收到悬空 tool_call。
        return Command(goto=target, update={"messages": [answer, ToolMessage(
            content=f"控制权已移交 {target}", name=call["name"], tool_call_id=call["id"])],
            "active_agent": target, "steps": ["triage"]})
    # triage 节点和子 Agent 节点共享同一 StateGraph，便于在同一上下文中整合消息。
    def node(name):
        async def respond(state: State, config: RunnableConfig):
            if state["active_agent"] != name:
                raise ValueError("当前 Agent 与目标节点不一致")
            # 共享移交历史，让专员直接面对用户，不再返回分诊节点。
            answer = (await model.ainvoke([SystemMessage(content=PROMPTS[name]), *state["messages"]], config)
                      if model is not None else AIMessage(content=PROMPTS[name] + "（离线教学回复）"))
            if answer.tool_calls or not answer.content:
                raise ValueError("专员未返回最终回复")
            return {"messages": [answer], "steps": [name]}
        return respond

    graph = StateGraph(State)
    graph.add_node("triage", triage)
    graph.add_edge(START, "triage")
    for name in AGENTS:
        graph.add_node(name, node(name))
        graph.add_edge(name, END)
    return graph.compile(checkpointer=InMemorySaver(), name="chapter09-handoff")

# 验证实际输出的消息协议与路径，而非只判断模型文字。
def validate(state, case):
    """验证实际输出的消息协议与路径，而非只判断模型文字。"""
    pending = {}
    count = 0
    for message in state["messages"]:
        if isinstance(message, AIMessage):
            assert not pending, "上一轮工具调用未配对"
            for call in message.tool_calls:
                assert call["id"] not in pending
                pending[call["id"]] = call["name"]
        elif isinstance(message, ToolMessage):
            assert message.tool_call_id in pending, "ToolMessage 无对应调用"
            assert message.name == pending.pop(message.tool_call_id)
            count += 1
    assert not pending, "tool_call_id 未配对"
    assert count == 1, f"预期一次委派/移交，实际 {count}"
    expected = (["supervisor", "account_agent", "supervisor"] if case == "supervisor"
                else ["triage", "logistics_agent"])
    assert state["steps"] == expected, state["steps"]
    assert state["active_agent"] == ("supervisor" if case == "supervisor" else "logistics_agent")
    assert isinstance(state["messages"][-1], AIMessage) and state["messages"][-1].content

# 主函数：根据命令行参数选择测试用例，在线或离线运行 Graph，打印结果并保存验收摘要。
async def run(args):
    load_dotenv(ROOT / ".env")
    offline = args.offline or args.self_test
    if offline:
        # 自检不上传 Trace，不把离线替身误标为在线 LLM。
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    model = None
    if not offline:
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise RuntimeError("缺少 DEEPSEEK_API_KEY")
        model = ChatDeepSeek(model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"), max_tokens=512,
                             max_retries=0, timeout=45, extra_body={"thinking": {"type": "disabled"}})
    cases = ["supervisor", "handoff"] if args.self_test or args.case == "all" else [args.case]
    evidence = []
    for case in cases:
        run_id = uuid4()
        graph = build_supervisor(model) if case == "supervisor" else build_handoff(model)
        config = {"run_id": run_id, "run_name": f"chapter09-{case}", "recursion_limit": 6,
                  "configurable": {"thread_id": str(run_id)},
                  "metadata": {"chapter": "09", "case": case, "offline": offline}}
        text = "账号无法登录，请提供核实建议。" if case == "supervisor" else "包裹物流三天未更新，请提供核实建议。"
        result = await graph.ainvoke({"messages": [HumanMessage(content=text)], "steps": []}, config)
        validate(result, case)
        item = {"case": case, "run_id": str(run_id), "offline": offline, "steps": result["steps"],
                "active_agent": result["active_agent"], "protocol": "PASS"}
        if case == "handoff":
            snapshot = await graph.aget_state(config)
            assert snapshot.next == () and snapshot.values["active_agent"] == "logistics_agent"
            item["checkpoint"] = "PASS"
        evidence.append(item)
        print(json.dumps(item, ensure_ascii=False))
    if args.evidence:
        Path(args.evidence).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["supervisor", "handoff", "all"], default="all")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--evidence", help="保存不含密钥/原始对话的验收摘要")
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except Exception as exc:
        # 不输出可能包含请求头或服务端响应的原始异常。
        print(f"FAIL: {type(exc).__name__}；请核查配置、配额或协议，未自动重试。")
        return 1
    finally:
        wait_for_all_tracers()


if __name__ == "__main__":
    raise SystemExit(main())
