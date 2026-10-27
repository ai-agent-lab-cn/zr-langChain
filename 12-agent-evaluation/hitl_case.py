"""C：真实 LangGraph interrupt/Command(resume)；所有业务写入为进程内 Fake。"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ReviewState(TypedDict, total=False):
    """审核状态：reviewer 为外部决定里的教学字段，不是真实身份认证。"""

    case_id: str
    decision: str
    status: str
    steps: Annotated[list[str], operator.add]


class FakeLedger:
    """内存 Fake 账本：同操作键只记一次，不代表数据库级 exactly-once。"""

    def __init__(self) -> None:
        self._effects: set[str] = set()

    def apply(self, key: str) -> bool:
        """若操作键已存在则返回 False，否则记录一次模拟副作用并返回 True。"""
        if key in self._effects:
            return False
        self._effects.add(key)
        return True

    @property
    def count(self) -> int:
        """已发生的唯一模拟写入数。"""
        return len(self._effects)


def build_review_graph(ledger: FakeLedger):
    """创建 HITL 图，使用 InMemorySaver 保存进程内暂停位置。

    参数 ledger 为进程内 FakeLedger；返回可同步 invoke 的已编译 Graph。
    在 review 节点执行 interrupt，恢复之后再做应用级角色判断；真实鉴权、
    审批有效期、跨进程幂等和数据库事务均不在教学实现中。
    """
    flow = StateGraph(ReviewState)

    def prepare(state: ReviewState) -> dict:
        return {"status": "awaiting_review", "steps": ["prepare"]}

    def review(state: ReviewState) -> dict:
        # interrupt 所在节点恢复时会重新进入，因此在此之前不得写入外部副作用。
        answer = interrupt({"kind": "approval", "case_id": state["case_id"]})
        if not isinstance(answer, dict):
            return {"decision": "reject", "steps": ["review"]}
        decision = "approve" if (answer.get("decision") == "approve" and answer.get("reviewer") == "approver") else "reject"
        return {"decision": decision, "steps": ["review"]}

    def route_decision(state: ReviewState) -> str:
        return "write" if state["decision"] == "approve" else "reject"

    def write(state: ReviewState) -> dict:
        ledger.apply(f"{state['case_id']}:write")
        return {"status": "approved", "steps": ["write"]}

    def reject(state: ReviewState) -> dict:
        return {"status": "rejected", "steps": ["reject"]}

    flow.add_node("prepare", prepare)
    flow.add_node("review", review)
    flow.add_node("write", write)
    flow.add_node("reject", reject)
    flow.add_edge(START, "prepare")
    flow.add_edge("prepare", "review")
    flow.add_conditional_edges("review", route_decision)
    flow.add_edge("write", END)
    flow.add_edge("reject", END)
    return flow.compile(checkpointer=InMemorySaver())


def run_review(graph: Any, case_id: str, decision: str, reviewer: str) -> dict:
    """检查暂停、Checkpoint 与恢复终态，并返回不含原文的观测。

    graph 为带 checkpointer 的图；decision/reviewer 是预设教学输入。
    每次生成独立 thread_id，初始调用必须出现 __interrupt__，恢复后必须 END；
    未达到状态约束将抛 AssertionError，绝不在报告里记作 PASS。
    """
    thread_id = f"ch12-{case_id}-{uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}
    paused = graph.invoke({"case_id": case_id, "steps": []}, config=config)
    checkpoint = graph.get_state(config)
    if "__interrupt__" not in paused or "review" not in checkpoint.next:
        raise AssertionError("图未在 review 节点真实暂停")
    resumed = graph.invoke(Command(resume={"decision": decision, "reviewer": reviewer}), config=config)
    if graph.get_state(config).next:
        raise AssertionError("恢复后图未结束")
    return {
        "case_id": case_id,
        "paused": True,
        "checkpoint_next": tuple(checkpoint.next),
        "status": resumed["status"],
        "steps": tuple(resumed["steps"]),
        "resumed": True,
    }
