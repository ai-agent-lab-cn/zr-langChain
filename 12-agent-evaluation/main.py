"""LangChain 入门实战（十二）：本地 Agent Evaluation A—D。

从仓库根目录运行：
    uv run --locked --offline python 12-agent-evaluation/main.py --self-test
    uv run --locked --offline python 12-agent-evaluation/main.py --case B

本程序真实执行 LangGraph，但专家回答是 Fake Tool，路由为固定规则。
不读取 .env，不创建 LLM / HTTP 客户端，也不发送 LangSmith Trace。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

# 必须早于 LangGraph / LangSmith 的导入：此 CLI 默认不可上传追踪数据。
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from evaluation_core import (  # noqa: E402
    DATASET_VERSION, EVALUATOR_VERSION, GOLDEN_DATASET,
    compare_versions, evaluate_case, summarize,
)

ROOT = Path(__file__).resolve().parent.parent
CHAPTER = Path(__file__).resolve().parent
SOURCE_FILES = ("main.py", "evaluation_core.py", "graph_cases.py", "hitl_case.py")
CASES = ("A", "B", "C", "D")


def source_sha256() -> str:
    """本次四个代码文件内容指纹；以文件名分隔，避免拼接歧义。"""
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        data = (CHAPTER / name).read_bytes()
        digest.update(name.encode("utf-8") + b"\0" + len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


def run_a() -> dict[str, Any]:
    """A：在真实图上执行冻结 Golden Dataset，用 Code Evaluator 逐案评分。"""
    from graph_cases import build_ticket_graph, run_ticket

    graph = build_ticket_graph()
    scores = [evaluate_case(c, run_ticket(graph, c.case_id, c.ticket, "candidate")) for c in GOLDEN_DATASET]
    result = summarize(scores, len(GOLDEN_DATASET))
    if result["status"] != "PASS":
        raise AssertionError(f"A 预期全部 PASS，实际为 {result['status']}")
    return {"status": "PASS", "summary": result, "graph_invokes": len(GOLDEN_DATASET)}


def run_b() -> dict[str, Any]:
    """B：真实 ToolNode/ToolMessage 轨迹 + 注入缺失/错配证据的负例。"""
    from graph_cases import build_ticket_graph, run_ticket

    combined = next(c for c in GOLDEN_DATASET if c.case_id == "combined")
    graph = build_ticket_graph()
    observed = run_ticket(graph, combined.case_id, combined.ticket, "candidate")
    correct = evaluate_case(combined, observed)
    # 将 ToolMessage id 替换为孤儿 id，必须判 FAIL 而不能照样 PASS。
    broken = evaluate_case(combined, replace(observed, tool_results=("orphan-id",)))
    # 不包含中间 Tool Call 的记录无法证明真实工具执行，应判 REVIEW。
    incomplete = evaluate_case(combined, replace(observed, tool_calls=(), tool_results=()))
    if (correct.outcome, broken.outcome, incomplete.outcome) != ("PASS", "FAIL", "REVIEW"):
        raise AssertionError("工具消息协议/缺失证据判定不符合预期")
    if tuple(name for name, _ in observed.tool_calls) != combined.expected_tools:
        raise AssertionError("真实 ToolNode 调用顺序错误")
    return {
        "status": "PASS", "graph_invokes": 1,
        "actual_tool_calls": len(observed.tool_calls),
        "actual_tool_results": len(observed.tool_results),
        "actual_steps": list(observed.steps),
        "good": correct.outcome, "orphan_id_injection": broken.outcome,
        "missing_trajectory_injection": incomplete.outcome,
    }


def run_c() -> dict[str, Any]:
    """C：真实 interrupt + Checkpoint + Command 恢复，验证拒绝与 Fake 幂等。"""
    from hitl_case import FakeLedger, build_review_graph, run_review

    ledger = FakeLedger()
    graph = build_review_graph(ledger)
    declined = run_review(graph, "denied", "reject", "approver")
    unauthorized = run_review(graph, "unauthorized", "approve", "guest")
    accepted = run_review(graph, "approved", "approve", "approver")
    before_replay = ledger.count
    replay_applied = ledger.apply("approved:write")
    if not (
        declined["status"] == "rejected"
        and unauthorized["status"] == "rejected"
        and accepted["status"] == "approved"
        and all(x["paused"] and x["resumed"] for x in (declined, unauthorized, accepted))
        and ledger.count == before_replay == 1
        and replay_applied is False
    ):
        raise AssertionError("审批/权限/真实暂停/模拟幂等验收失败")
    return {
        "status": "PASS", "graph_invokes": 6,
        "interrupts": 3, "resumes": 3, "fake_unique_effects": ledger.count,
        "fake_idempotency_replay_allowed": replay_applied,
        "decision_status": {"reject": declined["status"],
                            "unauthorized": unauthorized["status"],
                            "approve": accepted["status"]},
        "resume_steps": list(accepted["steps"]),
    }


def run_d() -> dict[str, Any]:
    """D：同一 Dataset 在真实图上重放两种固定策略，产生可复核回归对比。"""
    from graph_cases import build_ticket_graph, run_ticket

    graph = build_ticket_graph()
    baseline = [evaluate_case(c, run_ticket(graph, c.case_id, c.ticket, "baseline")) for c in GOLDEN_DATASET]
    candidate = [evaluate_case(c, run_ticket(graph, c.case_id, c.ticket, "candidate")) for c in GOLDEN_DATASET]
    comparison = compare_versions(baseline, candidate)
    if not (comparison["status"] == "PASS" and comparison["improvements"] == ["combined"]):
        raise AssertionError("相同数据集的预期回归改进未观察到")
    return {"status": "PASS", "comparison": comparison,
            "graph_invokes": 2 * len(GOLDEN_DATASET), "usage_metadata": None,
            "provider_calls": 0, "langsmith_traces_uploaded": 0}


RUNNERS = {"A": run_a, "B": run_b, "C": run_c, "D": run_d}


def main() -> int:
    """串行执行实验；失败后保留失败类别并标记后续 NOT_RUN，不制造 PASS。"""
    parser = argparse.ArgumentParser(description="第12章离线评测：真实 LangGraph + Fake Tool，不联网")
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--self-test", action="store_true", help="执行 A—D 全部场景")
    parser.add_argument("--evidence", type=Path, help="排他创建脱敏 JSON；已有文件拒绝覆盖")
    args = parser.parse_args()
    if args.self_test and args.case != "all":
        parser.error("--self-test 不应同时指定单个 --case")
    selected = CASES if args.self_test or args.case == "all" else (args.case,)

    evidence_file = None
    if args.evidence:
        try:
            args.evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence_file = args.evidence.open("x", encoding="utf-8")
        except OSError as exc:
            print(f"证据文件创建失败：{type(exc).__name__}（不会覆盖历史结果）", file=sys.stderr)
            return 1

    record = {
        "batch_id": f"ch12-offline-{uuid4().hex[:12]}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": DATASET_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "source_sha256": source_sha256(),
        "mode": "local-real-langgraph-with-fake-tools",
        "provider_calls": 0, "langsmith_traces_uploaded": 0,
        "usage_metadata": None, "status": "RUNNING", "cases": {},
    }
    failed = False
    try:
        for case in selected:
            if failed:
                record["cases"][case] = {"status": "NOT_RUN"}
                continue
            try:
                observed = RUNNERS[case]()
                record["cases"][case] = observed
                print(f"[chapter12 / {case}] PASS")
            except Exception as exc:
                failed = True
                record["cases"][case] = {"status": "FAILED", "error_type": type(exc).__name__}
                print(f"[chapter12 / {case}] FAILED: {type(exc).__name__}", file=sys.stderr)
        record["status"] = "FAILED" if failed else "PASS"
        print(f"总体验收：{record['status']}；执行场景：{','.join(selected)}")
        return 1 if failed else 0
    finally:
        if evidence_file is not None:
            json.dump(record, evidence_file, ensure_ascii=False, indent=2)
            evidence_file.write("\n")
            evidence_file.close()
            print(f"本地证据：{args.evidence}")


if __name__ == "__main__":
    raise SystemExit(main())
