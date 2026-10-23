"""LangChain 入门实战（八）：LangGraph HITL 暂停、人工审核与恢复。

从仓库根目录运行：
    uv run python 08-langgraph-hitl/main.py --self-test
    uv run python 08-langgraph-hitl/main.py --case all --offline
    uv run python 08-langgraph-hitl/main.py --case all
    uv run python 08-langgraph-hitl/main.py --case edit --offline --interactive

注意：InMemorySaver 仅在当前 Python 进程内保留 Checkpoint；不是跨重启持久化。
人工决策仅用于教学演示，不操作真实订单、账户、资金或工单系统。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import operator
import os
import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

# 【环境配置】所有章节复用仓库根目录 .env，不在每个章节复制密钥。
# 本文件在 08-langgraph-hitl/，其上两级目录就是仓库根目录。
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# 【实验可靠性】通用工具放在根目录 scripts/，HITL 图本身不依赖 Runner 的实现。
# 直接执行 `python 08-langgraph-hitl/main.py` 时，把兄弟目录 scripts/ 加入模块搜索路径。
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_errors import classify_error  # noqa: E402
from experiment_runner import ExperimentBatch, CASE_NAMES  # noqa: E402

# 【教学中的三组类型】类别是 LLM 的草稿分类结果；审核动作由人提供；
# status 则是 Graph 执行后的业务演示状态，三者不能混淆。
Category = Literal["account", "payment", "logistics", "manual"]
Action = Literal["approve", "reject", "edit"]
Status = Literal["awaiting_review", "approved", "rejected", "edited"]

# 【固定实验用例】每个场景指定一条虚构工单和预期审核动作。
# 自动实验使用脚本填入人工决定；interactive 模式才接收用户实时输入。
CASES: dict[str, tuple[str, Action]] = {
    "approve": ("我的账号无法登录，一直提示密码错误。", "approve"),
    "reject": ("我想申请定制企业报表并升级 VIP 套餐。", "reject"),
    "edit": ("包裹已发货，但是物流信息三天没有更新。", "edit"),
}
# edit 分支的可复现修订文本；注意不是 DeepSeek 二次生成的回复。
EDITED_REPLY = "人工修订：请先核对运单号与最后一次物流更新，再由客服确认处理进展。（教学示例）"


# 【LangGraph State】所有节点共享这份数据结构（TypedDict 是类型声明，不做运行时业务校验）。
# 节点只返回需要更新的字段；Reducer 决定这些增量如何合并。
class TicketState(TypedDict, total=False):
    ticket: str  # 用户输入的原始工单；prepare_draft 和 human_review 读取。
    category: Category  # DeepSeek 或离线规则提出的工单类别。
    draft: str  # prepare_draft 生成的待人工审核草稿。
    final_reply: str  # 审核分支选择后才能写入的最终教学回复。
    review_action: Action  # interrupt 恢复后的人类动作：approve/reject/edit。
    review_note: str  # 人工审核的附加备注。
    edited_reply: str  # edit 动作提交的修订内容。
    status: Status  # awaiting_review（等待审核）/ approved / rejected / edited。
    # 【Reducer】operator.add 让每个完成的节点追加自己的执行路径，而不覆盖先前节点。
    # 注意 human_review 在中断尚未返回增量，因此暂停快照中没有该步骤。
    steps: Annotated[list[str], operator.add]


# 【结构化输出】Pydantic 负责校验模型生成的类别和草稿字段。
# 它与 TicketState 不同：前者约束单次模型输出，后者存储整个 Graph 的状态。
class DraftProposal(BaseModel):
    """模型只提供待审草稿，不获准执行真实业务操作。"""

    category: Category = Field(description="账户、支付、物流三类，否则 manual")
    suggestion: str = Field(description="给客服的简短处理建议；不要声称已完成实际操作")


# 【安全边界】模型仅起草待审建议，不进行真实客服操作、支付操作或消息发送。
SYSTEM_PROMPT = (
    "你是工单处理建议起草助手。仅输出结构化草稿，不执行任何真实业务操作。"
    "分类只能是 account/payment/logistics/manual，范围不明或多类歧义用 manual。"
    "建议必须简短、准确、可供人工审核；不得声称已经退款、修改账户或处理订单。"
)


# 【确定性测试替身】只有草稿生成方式不同；HITL 仍然使用真实 LangGraph。
# 多类别关键词同时命中或者完全不命中时进入 manual，避免随意选一个分类。
def offline_proposal(ticket: str) -> DraftProposal:
    """离线规则替身；可复现、零 API 费用，不代表真实模型语义能力。"""
    keywords = {
        "account": ("账号", "账户", "登录", "密码"),
        "payment": ("付款", "支付", "退款", "扣款"),
        "logistics": ("物流", "包裹", "快递", "配送", "运单"),
    }
    # any(...) 判断一类是否命中至少一个词；多类命中则归为 manual。
    matched = [k for k, v in keywords.items() if any(term in ticket for term in v)]
    category: Category = matched[0] if len(matched) == 1 else "manual"
    suggestions = {
        "account": "建议核实登录提示与账户信息，交由客服确认。（教学示例）",
        "payment": "建议核实订单与支付记录，不收集完整银行卡信息。（教学示例）",
        "logistics": "建议核对运单状态并由客服跟进。（教学示例）",
        "manual": "需求范围不明，建议人工确认业务归属。（教学示例）",
    }
    return DraftProposal(category=category, suggestion=suggestions[category])


# 【Graph 构建入口】将草稿生成、真正的暂停/恢复和审核条件分支连接到一张图。
# model=None 使用离线草稿，否则只在 prepare_draft 节点调用 DeepSeek。
def build_graph(model: ChatDeepSeek | None = None):
    """与第七章一样，离线和在线使用同一张真实 StateGraph。"""
    # 【LangChain API】with_structured_output 只是在包装模型；这行代码不会发起请求。
    # 真正的 LLM 调用在 prepare_draft 内的 classifier.ainvoke(...)。
    classifier = model.with_structured_output(DraftProposal, method="function_calling") if model is not None else None

    # 【Node 1：准备草稿】读取原始工单，返回 category/draft/status/steps 的状态增量。
    # 先把草稿节点完整执行，再进入 human_review；恢复审核节点时不会重跑本节点。
    async def prepare_draft(state: TicketState, config: RunnableConfig) -> dict:
        ticket = state["ticket"].strip()
        if not ticket:
            raise ValueError("工单内容不能是空字符串")
        if classifier is None:
            proposal = offline_proposal(ticket)
        else:
            # 【异步模型调用】把上层 config 传给子调用，便于 LangSmith 关联 Graph 与 LLM Trace。
            proposal = await classifier.ainvoke(
                [("system", SYSTEM_PROMPT), ("human", f"请为以下工单起草待人工审核建议：{ticket}")],
                config=config,
            )
            if not isinstance(proposal, DraftProposal):
                raise TypeError("DeepSeek 未返回预期的 DraftProposal 结构化结果")
        # 先完成草稿并写入 Checkpoint，状态成为 awaiting_review。
        # 这里没有实际人工审批；只有下一个节点调用 interrupt() 时才真正暂停。
        return {
            "category": proposal.category,
            "draft": proposal.suggestion,
            "status": "awaiting_review",
            "steps": ["prepare_draft"],
        }

    # 【Node 2：HITL 核心】本节点负责“发出审核请求 → 暂停 → 收到人工决定”。
    # 首次执行：interrupt(payload) 暂停，节点不会返回 State 增量。
    # 恢复执行：同一 thread_id 的 Command(resume=...) 成为 interrupt() 的返回值。
    # 重要：恢复从 human_review 节点开头重跑，所以 interrupt 前只构造无副作用的负载。
    def human_review(state: TicketState) -> dict:
        # 不要在 interrupt 之前执行数据库写入/发消息等副作用：节点恢复会从头执行。
        # 【核心 API ①】payload 应当是 JSON 可序列化的审核请求，供 CLI/UI 展示。
        # 它不是人工的审核结果；结果将在 resume 后通过 decision 取得。
        decision = interrupt({
            "kind": "ticket_review",
            "ticket": state["ticket"],
            "category": state["category"],
            "draft": state["draft"],
            "actions": ["approve", "reject", "edit"],
            "instruction": "请选择批准、拒绝或修改；这是模拟审核，不会产生真实业务操作。",
        })
        # 恢复后要主动验证人工输入：Python Type Hint 不能代替运行时校验。
        if not isinstance(decision, dict):
            raise ValueError("人工返回值必须为字典，例如 {'action': 'approve'}")
        action = decision.get("action")
        if action not in ("approve", "reject", "edit"):
            raise ValueError("action 只能为 approve/reject/edit")
        edited_reply = decision.get("edited_reply", "")
        if action == "edit" and (not isinstance(edited_reply, str) or not edited_reply.strip()):
            raise ValueError("edit 决策必须提供非空 edited_reply")
        note = decision.get("note", "")
        if not isinstance(note, str):
            raise ValueError("note 必须为字符串")
        # 【State 更新】只有恢复且校验成功才追加 human_review 步骤；
        # 这样就不会把中断中的节点误判为“已经完成”。
        return {
            "review_action": action,
            "review_note": note,
            "edited_reply": edited_reply.strip() if isinstance(edited_reply, str) else "",
            "steps": ["human_review"],
        }

    # 【条件边路由函数】读取 review_action 选择后续节点，本身不写入 State。
    # 与第七章 select_branch 的 API 相同，只是判断依据变成人工审核结果。
    def choose_review_branch(state: TicketState) -> Action:
        return state["review_action"]

    # 【审核分支】批准 = 使用原草稿作为教学用最终回复；不代表“已发送给客户”。
    def approve(state: TicketState) -> dict:
        return {"final_reply": state["draft"], "status": "approved", "steps": ["approve"]}

    # 拒绝 = 不将原草稿设为最终回复；此处只写入模拟拒绝的文字结果。
    def reject(state: TicketState) -> dict:
        return {
            "final_reply": "人工拒绝了本次草稿，请重新确认需求。（教学示例；未执行业务操作）",
            "status": "rejected",
            "steps": ["reject"],
        }

    # 修改 = 用人工输入的 edited_reply 替换最终回复，不再次调用 DeepSeek。
    def edit(state: TicketState) -> dict:
        return {
            "final_reply": state["edited_reply"],
            "status": "edited",
            "steps": ["edit"],
        }

    # 【最终校验节点】断言实际路径仅走一个审核分支，status 与 action 一致。
    # 校验逻辑用于教学验收；不会进行真实工单写入或消息发送。
    def finalize(state: TicketState) -> dict:
        selected = state["review_action"]
        if not state.get("final_reply"):
            raise ValueError("缺少最终处理建议")
        if state.get("steps") != ["prepare_draft", "human_review", selected]:
            raise ValueError(f"异常执行路径：{state.get('steps')}")
        if state["status"] != {"approve": "approved", "reject": "rejected", "edit": "edited"}[selected]:
            raise ValueError("审核动作与最终状态不一致")
        return {"steps": ["finalize"]}

    # 【LangGraph API ②】StateGraph 声明共享 State 的 Graph 构建器；此时尚不能执行。
    builder = StateGraph(TicketState)
    # 【LangGraph API ③】add_node(name, fn) 注册 6 个可执行节点；
    # human_review 是唯一调用 interrupt() 的业务节点。
    for name, node in [
        ("prepare_draft", prepare_draft),
        ("human_review", human_review),
        ("approve", approve), ("reject", reject), ("edit", edit),
        ("finalize", finalize),
    ]:
        builder.add_node(name, node)
    # 【LangGraph API ④】固定边决定先起草、再审核。
    builder.add_edge(START, "prepare_draft")
    builder.add_edge("prepare_draft", "human_review")
    # 【LangGraph API ⑤】审核结束后按 review_action 三选一。
    # 注意 route 函数不是独立处理节点，也不会触发另一次模型调用。
    builder.add_conditional_edges("human_review", choose_review_branch,
                                  {"approve": "approve", "reject": "reject", "edit": "edit"})
    # 【LangGraph API ⑥】三个分支汇合后执行 finalize，再到 END。
    for name in ("approve", "reject", "edit"):
        builder.add_edge(name, "finalize")
    builder.add_edge("finalize", END)
    # 【LangGraph API ⑦】compile(checkpointer=...) 启用 Checkpoint 保存和恢复。
    # InMemorySaver 仅支持当前 Python 进程生命周期；跨重启需持久化 Checkpointer。
    return builder.compile(checkpointer=InMemorySaver())


# 【运行身份与可观察性】thread_id 用来检索同一 Graph 执行线程的 Checkpoint；
# pause / resume 使用同一个 thread_id，但 LangSmith run_name 区分两次 Graph 调用。
# thread_id 不是登录凭据：生产环境需要额外鉴权与资源授权。
def trace_config(case: str, thread_id: str, phase: str, *, batch_id: str | None = None) -> dict:
    # 【关联 Trace】batch_id 关联本轮实验；thread_fingerprint 用于脱敏配对 pause/resume。
    # 原始 thread_id 仅用于 configurable 查找 Checkpoint，不放入公开报告。
    trace_meta = {"chapter": "08-langgraph-hitl", "scenario": case, "phase": phase}
    if batch_id:
        trace_meta["batch_id"] = batch_id
        trace_meta["thread_fingerprint"] = hashlib.sha256(thread_id.encode()).hexdigest()[:12]
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": f"chapter08-{case}-{phase}",
        "tags": ["zrLangChain", "chapter-08", "langgraph", "hitl", case, phase],
        "metadata": trace_meta,
    }


# 【执行阶段一：暂停】真正调用 ainvoke()，观察 __interrupt__ 和已保存的 State。
async def begin(graph, case: str, thread_id: str, ticket: str, *, batch_id: str | None = None) -> dict:
    """第一段只负责暂停；检查真实 checkpoint，而非模拟暂停标记。"""
    cfg = trace_config(case, thread_id, "pause", batch_id=batch_id)
    # 【核心 API ⑧】第一次 ainvoke 从 START 执行至 human_review 的 interrupt()，并暂停。
    # 此时不会走 approve/reject/edit，也不会到 END。
    output = await graph.ainvoke({"ticket": ticket, "steps": []}, config=cfg)
    # 默认 invoke/ainvoke 调用将中断负载放在 __interrupt__ 中（不是最终业务回复）。
    interrupts = output.get("__interrupt__", ())
    # 【核心 API ⑨】aget_state(config) 读取相同 thread_id 当前 Checkpoint。
    # next 表示待执行的节点；values 是已成功写入的 State。
    snap = await graph.aget_state(cfg)
    if len(interrupts) != 1 or not snap.next or snap.next[0] != "human_review":
        raise AssertionError("期望在 human_review 发生真实 interrupt，但实际没有暂停")
    payload = interrupts[0].value
    if not isinstance(payload, dict) or payload.get("kind") != "ticket_review":
        raise AssertionError("interrupt 负载异常")
    # 验证暂停节点还没完成，因此步骤日志只有已完成的 prepare_draft。
    if snap.values.get("steps") != ["prepare_draft"]:
        raise AssertionError(f"暂停前 State 异常：{snap.values.get('steps')}")
    if snap.values.get("status") != "awaiting_review":
        raise AssertionError("暂停时并非 awaiting_review 状态")
    if snap.values.get("draft") != payload.get("draft"):
        raise AssertionError("审核负载与 Checkpoint 草稿不一致")
    print("⏸ 已暂停: human_review；线程:", thread_id)
    print("Checkpoint:", snap.values.get("steps"), "/ status=awaiting_review")
    print("待审核草稿:", payload["draft"])
    return payload


# 【执行阶段二：恢复】用同一个 thread_id 提交人工决定；resume 值传给 interrupt()。
async def resume(graph, case: str, thread_id: str, action: Action, *, edited_reply: str = "", batch_id: str | None = None) -> TicketState:
    cfg = trace_config(case, thread_id, "resume", batch_id=batch_id)
    response = {"action": action, "note": f"{case} 教学审核"}
    if action == "edit":
        response["edited_reply"] = edited_reply
    # 【核心 API ⑩】Command(resume=response) 并非新工单输入，而是恢复旧 Checkpoint。
    # human_review 将从节点开头重跑；其前面已经完成的 prepare_draft 不会重跑。
    output = await graph.ainvoke(Command(resume=response), config=cfg)
    if "__interrupt__" in output:
        raise AssertionError("恢复后不应再次暂停")
    # 再读同一个线程的 Checkpoint，确认没有剩余待执行节点。
    snap = await graph.aget_state(cfg)
    if snap.next:
        raise AssertionError(f"工作流尚未结束：{snap.next}")
    # 最终 Graph 输出应与最后一份 Checkpoint 的 State 一致。
    if output != snap.values:
        raise AssertionError("最终输出与 checkpoint 中的 State 不匹配")
    # 【Reducer 验收】验证中断前后 steps 正确追加，且只经过一个审核分支。
    required = ["prepare_draft", "human_review", action, "finalize"]
    if output.get("steps") != required:
        raise AssertionError(f"恢复后路径不正确：{output.get('steps')}")
    expected_status = {"approve": "approved", "reject": "rejected", "edit": "edited"}[action]
    if output.get("status") != expected_status:
        raise AssertionError(f"最终状态错误：{output.get('status')}")
    if action == "approve" and output["final_reply"] != output["draft"]:
        raise AssertionError("批准时应保留审核前草稿")
    if action == "edit" and output["final_reply"] != edited_reply:
        raise AssertionError("修改结果没有写回最终 State")
    if action == "reject" and output["final_reply"] == output["draft"]:
        raise AssertionError("拒绝后不能直接发布原草稿")
    print("▶ 已恢复:", " → ".join(output["steps"]))
    print("最终状态:", output["status"])
    print("处理结果:", output["final_reply"])
    print("验收: ✅ 通过")
    return output


# 【单场景实验】先暂停、后审核恢复；默认脚本提供审核结果，interactive 由人输入。
# UUID 避免多轮测试重复复用同一个线程的旧 Checkpoint。
async def run_case(graph, case: str, *, interactive: bool = False,
                   batch_id: str | None = None, on_phase=None) -> None:
    ticket, expected = CASES[case]
    thread_id = f"ch08-{case}-{uuid4().hex[:10]}"
    print(f"\n============ hitl / {case} ============")
    print("工单:", ticket)
    if on_phase:
        on_phase("pause")
    await begin(graph, case, thread_id, ticket, batch_id=batch_id)
    action = expected
    edited = EDITED_REPLY if action == "edit" else ""
    # interactive=True 才让操作者现场输入，否则使用固定实验决定便于自动验收。
    if interactive:
        print("请选择动作 approve/reject/edit（回车默认", expected, "）:")
        candidate = input("审核动作: ").strip()
        action = candidate or expected
        if action not in ("approve", "reject", "edit"):
            raise ValueError("无效审核动作")
        if action == "edit":
            edited = input("人工修订后的建议: ").strip()
            if not edited:
                raise ValueError("人工修订不能为空")
    if on_phase:
        on_phase("resume")
    result = await resume(graph, case, thread_id, action, edited_reply=edited, batch_id=batch_id)
    if not interactive and result["review_action"] != expected:
        raise AssertionError("自动验收决策不符合预期")


# 【隔离实验】在同一 compiled graph + InMemorySaver 中，让两个 thread_id 分别保存状态。
# 先恢复 B，再检查 A 仍待审核；最终各自恢复并校验业务数据没有串线。
async def run_isolation(graph, *, batch_id: str | None = None, on_phase=None) -> None:
    """一个 graph、两个不同的 thread_id；交叉恢复验证 Checkpoint 隔离。"""
    print("\n============ hitl / isolation ============")
    a_id = f"ch08-A-{uuid4().hex[:10]}"
    b_id = f"ch08-B-{uuid4().hex[:10]}"
    a_ticket = CASES["approve"][0]
    b_ticket = CASES["edit"][0]
    # A、B 并发推进到各自 interrupt()，两个 Checkpoint 按 thread_id 区分。
    if on_phase:
        on_phase("pause-isolation")
    a, b = await asyncio.gather(
        begin(graph, "isolation-A", a_id, a_ticket, batch_id=batch_id),
        begin(graph, "isolation-B", b_id, b_ticket, batch_id=batch_id),
    )
    # 【隔离验收】比较“工单是否归属于正确线程”，而不是强制要求草稿文本不同。
    # 在线 DeepSeek 可能对不同输入给出相似甚至相同的建议，不能因此判定 Checkpoint 串线。
    if a["ticket"] != a_ticket or b["ticket"] != b_ticket:
        raise AssertionError("暂停负载的工单与对应 thread_id 不匹配")
    # 再从两个不同 thread_id 读取真实 Checkpoint；验证二者都对应原始工单。
    a_before = await graph.aget_state(trace_config("isolation-A", a_id, "inspect-before"))
    b_before = await graph.aget_state(trace_config("isolation-B", b_id, "inspect-before"))
    if a_before.values.get("ticket") != a_ticket or b_before.values.get("ticket") != b_ticket:
        raise AssertionError("两个线程的 Checkpoint 工单归属错误")
    # 交叉恢复：先 B，检查 A 仍待审核；再恢复 A。
    # 交错顺序是：B 完成 → A 仍暂停 → A 完成；重点验证线程隔离。
    if on_phase:
        on_phase("resume-isolation")
    b_result = await resume(graph, "isolation-B", b_id, "edit", edited_reply=EDITED_REPLY, batch_id=batch_id)
    a_cfg = trace_config("isolation-A", a_id, "inspect")
    a_snapshot = await graph.aget_state(a_cfg)
    if a_snapshot.next != ("human_review",) or a_snapshot.values.get("status") != "awaiting_review":
        raise AssertionError("B 的恢复不应改变 A 的暂停状态")
    a_result = await resume(graph, "isolation-A", a_id, "approve", batch_id=batch_id)
    if a_result["ticket"] != a_ticket or b_result["ticket"] != b_ticket:
        raise AssertionError("不同 thread_id 的工单数据发生串线")
    print("thread_id / Checkpoint 隔离: ✅ 通过")


# 【命令行调度】--self-test 和 --offline 都不创建 DeepSeek 模型，但真正执行 LangGraph。
# 在线模式才加载根目录的 DEEPSEEK_API_KEY。
async def main_async(args) -> int:
    if args.resume_batch and (args.self_test or args.offline or args.interactive):
        raise ValueError("--resume-batch 只支持非交互的在线实验；离线自检不使用批次状态")
    if args.interactive and (args.case in ("all", "isolation") or args.self_test):
        raise ValueError("--interactive 只能与 approve/reject/edit 中单个场景使用")
    model = None
    offline = args.self_test or args.offline
    if not offline:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY，请检查根目录 .env")
        # 对教学型短草稿关闭思考模式，减少延迟和推理 Token；禁用底层 SDK 自动重试，
        # 避免余额不足 / 限流时多次无效请求，交给实验可靠性模块明确分类处理。
        model = ChatDeepSeek(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            api_key=api_key,
            extra_body={"thinking": {"type": "disabled"}},
            max_tokens=1024,
            max_retries=0,
        )
    # 始终使用同一个图结构；离线与在线仅影响 prepare_draft 的起草方式。
    graph = build_graph(model)
    print("运行模式:", "真实 LangGraph / 离线规则草稿 / 0 次 DeepSeek" if offline else "真实 LangGraph / DeepSeek 结构化草稿")
    cases = ["approve", "reject", "edit", "isolation"] if args.self_test or args.case == "all" else [args.case]
    # 【在线批次】运行时写在 .runs/（必须 Git 忽略），保留 PASS 供后续只重跑未通过用例。
    # source_hash 可阻止脚本修改后复用旧 PASS；隔离实验作为整体原子场景。
    batch = None
    if not offline and not args.interactive:
        source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
        path = ROOT / ".runs" / "chapter08" / "latest.json"
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        batch = (ExperimentBatch.resume(path, chapter="08", model=model_name, source_hash=source_hash)
                 if args.resume_batch else
                 ExperimentBatch.create(path, chapter="08", model=model_name, source_hash=source_hash))
        print("实验批次:", batch.batch_id, "|", "恢复未通过场景" if args.resume_batch else "新建批次")

    all_good = True
    for case in cases:
        if batch and batch.status(case) == "PASS":
            print(f"⏭ {case}：已在本批次 PASS，跳过重复 API 调用")
            continue
        phase = {"value": "startup"}
        on_phase = lambda value: phase.__setitem__("value", value)
        if batch:
            batch.start(case)   # 在 API 请求前记录 RUNNING，异常退出也不会被误判 PASS
        try:
            if case == "isolation":
                await run_isolation(graph, batch_id=batch.batch_id if batch else None,
                                    on_phase=on_phase)
            else:
                await run_case(graph, case, interactive=args.interactive,
                               batch_id=batch.batch_id if batch else None, on_phase=on_phase)
            if batch:
                batch.pass_case(case)
        except Exception as exc:
            all_good = False
            failure = classify_error(exc, phase=phase["value"])
            if batch:
                batch.fail_case(case, failure)
            # 不把含有服务端配额详情 / API 响应 / 工单内容的完整异常写到批次文件。
            retry = f"；建议最早重试间隔约 {failure.retry_after_seconds}s" if failure.retry_after_seconds else ""
            print(f"❌ {case} 失败：{failure.code.value}（阶段：{failure.phase}）{retry}")
            if failure.stop_batch:
                print("⛔ 检测到配额耗尽或账户余额不足：终止本批后续在线用例；保留剩余 PENDING")
                break
    if batch:
        print("批次状态:", batch.overall_status(), "|", batch.path)
    print("\n========== 总体验收 ==========")
    if args.self_test:
        print("self-test:", "✅ 全部通过（真实 LangGraph Interrupt；0 次 DeepSeek 调用）" if all_good else "❌ 存在失败")
    else:
        print("✅ 所选实验全部通过" if all_good else "❌ 存在失败实验")
    return 0 if all_good else 1


# 【程序入口】解析实验参数并将是否通过映射为 CLI 退出码（0=通过，1=失败）。
def main() -> int:
    parser = argparse.ArgumentParser(description="Chapter 08: LangGraph Human-in-the-loop")
    parser.add_argument("--self-test", action="store_true", help="离线执行暂停、恢复、编辑、拒绝及双线程隔离")
    parser.add_argument("--case", choices=["approve", "reject", "edit", "isolation", "all"], default="all")
    parser.add_argument("--offline", action="store_true", help="离线规则代替 DeepSeek 起草，但真正执行 HITL Graph")
    parser.add_argument("--interactive", action="store_true", help="在同一 CLI 进程中手动输入审核动作")
    parser.add_argument("--resume-batch", action="store_true", help="在线批次恢复：跳过上次已 PASS 的用例，重跑其余场景")
    args = parser.parse_args()
    try:
        return asyncio.run(main_async(args))
    except Exception as exc:
        print(f"❌ 运行失败：{type(exc).__name__}: {exc}")
        return 1
    finally:
        # 退出前等待 LangSmith 异步 Trace 尽量写完，减少因进程退出而丢记录。
        wait_for_all_tracers()


if __name__ == "__main__":
    raise SystemExit(main())
