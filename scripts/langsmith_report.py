#!/usr/bin/env python3
"""zrLangChain｜LangSmith Trace Analyzer（只读、默认脱敏）。

本工具只读取 LangSmith，不发送额外的 LLM 请求，也不上传 Trace 到 GitHub。
推荐从仓库根目录运行：
    uv run python scripts/langsmith_report.py --chapter 08 --since-hours 2 --expect-online --expect-all
    uv run python -m unittest discover -s tests -p 'test_langsmith_report.py' -v

设计：一个根 Run 代表一次 Graph 调用，子 Run 中 run_type='llm' 才算 LLM 调用；
避免把 Graph / RunnableSequence / Parser 的聚合 token 再加一次。
使用新异步 runs.query 扫描根 Run，traces.list_runs 读取精确调用树。
字段归一化由 langsmith_reader 完成；不调用旧版 read_run/list_runs。
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from itertools import islice
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable

from langsmith_reader import TraceReadLimitExceeded, fetch_roots, fetch_trace

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS = ROOT / "reports"
EXPECTED_HITL = {"approve", "reject", "edit", "isolation-A", "isolation-B"}
EXPECTED_HITL_STATUS = {"approve": "approved", "reject": "rejected", "edit": "edited", "isolation-A": "approved", "isolation-B": "edited"}
CH08_NAME = re.compile(r"^chapter08-(.+)-(pause|resume)$")


def value(obj: Any, name: str, default: Any = None) -> Any:
    """兼容 LangSmith SDK 的 Run 模型和测试中的普通字典。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def mapping(obj: Any) -> dict[str, Any]:
    return obj if isinstance(obj, dict) else {}


def metadata(run: Any) -> dict[str, Any]:
    """LangSmith SDK 的 metadata 通常存放在 extra.metadata，亦兼容导出文件结构。"""
    extra = mapping(value(run, "extra"))
    return {**mapping(extra.get("metadata")), **mapping(value(run, "metadata"))}


def extract_thread_id(run: Any) -> str | None:
    """只接受 Trace 明确给出的 thread_id；不根据业务样例或运行时间推断。"""
    info = metadata(run)
    direct = info.get("thread_id")
    if direct:
        return str(direct)
    configurable = mapping(info.get("configurable"))
    if configurable.get("thread_id"):
        return str(configurable["thread_id"])
    extra = mapping(value(run, "extra"))
    extras = mapping(extra.get("configurable"))
    if extras.get("thread_id"):
        return str(extras["thread_id"])
    return None


def safe_thread_label(thread_id: str | None) -> str | None:
    """报告中用稳定不可逆的短摘要做匹配，不写出完整 thread_id。"""
    return hashlib.sha256(thread_id.encode()).hexdigest()[:12] if thread_id else None


def when(run: Any, field: str = "start_time") -> datetime | None:
    dt = value(run, field)
    if isinstance(dt, datetime):
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    if isinstance(dt, str):
        try:
            parsed = datetime.fromisoformat(dt.replace("Z", "+00:00"))
            return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
        except ValueError:
            return None
    return None


def natural_int(obj: Any) -> int | None:
    if isinstance(obj, bool):
        return None
    if isinstance(obj, int) and obj >= 0:
        return obj
    return None


def first_int(*items: Any) -> int | None:
    return next((v for item in items if (v := natural_int(item)) is not None), None)


def token_usage(run: Any) -> dict[str, int | None]:
    """仅供 llm 子 Run 使用；缺失计量以 None 表示，绝不伪装为 0。"""
    extra = mapping(value(run, "extra"))
    outputs = mapping(value(run, "outputs"))
    m = metadata(run)
    response = mapping(outputs.get("response_metadata"))
    generation = mapping(outputs.get("generations"))
    # 兼容 SDK 的 prompt_tokens/completion_tokens 与模型 usage_metadata。
    usage = mapping(m.get("usage_metadata")) or mapping(response.get("usage_metadata"))
    if not usage:
        usage = mapping(outputs.get("usage_metadata"))
    if not usage:
        usage = mapping(extra.get("usage_metadata"))
    token_info = mapping(outputs.get("token_usage")) or mapping(response.get("token_usage"))
    prompt = first_int(value(run, "prompt_tokens"), usage.get("input_tokens"), token_info.get("prompt_tokens"))
    completion = first_int(value(run, "completion_tokens"), usage.get("output_tokens"), token_info.get("completion_tokens"))
    total = first_int(value(run, "total_tokens"), usage.get("total_tokens"), token_info.get("total_tokens"))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    details = (
        mapping(value(run, "completion_token_details"))
        or mapping(m.get("output_token_details"))
        or mapping(usage.get("output_token_details"))
        or mapping(response.get("output_token_details"))
    )
    reasoning = first_int(details.get("reasoning"), details.get("reasoning_tokens"), details.get("reasoning_output_tokens"))
    return {"input": prompt, "output": completion, "total": total, "reasoning": reasoning}


def state_output(run: Any) -> dict[str, Any]:
    """只取 State 里的非敏感结构字段；不会在报告中记录原始 draft/ticket。"""
    data = mapping(value(run, "outputs"))
    if "output" in data and isinstance(data["output"], dict) and "status" in data["output"]:
        return data["output"]
    return data


def error_present(run: Any) -> bool:
    return bool(value(run, "error"))


def summarize_trace(root: Any, runs: list[Any], chapter: str, *, truncated: bool = False) -> dict[str, Any]:
    name = str(value(root, "name") or "")
    phase, case = None, name.removeprefix(f"chapter{chapter}-")
    if chapter == "08":
        match = CH08_NAME.fullmatch(name)
        if match:
            case, phase = match.group(1), match.group(2)
    # API 分页可能包含根 Run：按 id 去重，且确保根 Run 一定在列表中。
    unique: dict[str, Any] = {}
    for obj in [root, *runs]:
        run_id = str(value(obj, "id") or value(obj, "run_id") or "")
        if run_id:
            unique[run_id] = obj
    all_runs = list(unique.values())
    llms = [r for r in all_runs if value(r, "run_type") == "llm"]
    used = [token_usage(r) for r in llms]
    counts = {
        "input": sum(item["input"] or 0 for item in used),
        "output": sum(item["output"] or 0 for item in used),
        "total": sum(item["total"] or 0 for item in used),
        "reasoning": sum(item["reasoning"] or 0 for item in used),
    }
    missing = {key: sum(item[key] is None for item in used) for key in counts}
    errors = sum(error_present(r) for r in all_runs)
    # Only the known direct human_review control-flow exception is a candidate.
    # Never evaluate the repr or retain its potentially sensitive payload.
    candidates = sum(
        value(r, "name") == "human_review"
        and value(r, "run_type") == "chain"
        and str(value(r, "parent_run_id")) == str(value(root, "id"))
        and str(value(r, "trace_id")) == str(value(root, "trace_id") or value(root, "id"))
        and isinstance(value(r, "error"), str)
        and value(r, "error").startswith("GraphInterrupt(")
        and "Interrupt(value=" in value(r, "error")
        for r in all_runs
    ) if chapter == "08" and phase == "pause" and not error_present(root) else 0
    outputs = state_output(root)
    step_names = outputs.get("steps")
    steps = step_names if isinstance(step_names, list) and all(isinstance(n, str) for n in step_names) else []
    interrupt = outputs.get("__interrupt__")
    has_interrupt = bool(interrupt)
    status = outputs.get("status") if isinstance(outputs.get("status"), str) else None
    action = outputs.get("review_action") if isinstance(outputs.get("review_action"), str) else None
    thread = extract_thread_id(root)
    info = metadata(root)
    fingerprint = info.get("thread_fingerprint")
    # 新增的客户端脱敏指纹优先；旧版 Trace 仍回退到明确记录的 thread_id。
    thread_hash = fingerprint if isinstance(fingerprint, str) and re.fullmatch(r"[0-9a-f]{12}", fingerprint) else safe_thread_label(thread)
    batch_id = info.get("batch_id") if isinstance(info.get("batch_id"), str) else None
    return {
        "name": name,
        "root_id": str(value(root, "id") or ""),
        "trace_id": str(value(root, "trace_id") or value(root, "id") or ""),
        "start_time": when(root).isoformat() if when(root) else None,
        "end_time": when(root, "end_time").isoformat() if when(root, "end_time") else None,
        "case": case,
        "phase": phase,
        "run_count": len(all_runs),
        "llm_calls": len(llms),
        "nodes": sorted({str(value(r, "name")) for r in all_runs if str(value(r, "name")) in {"prepare_draft", "human_review", "classify", "finalize", "approve", "reject", "edit", "handle_account", "handle_payment", "handle_logistics", "handle_manual"}}),
        "tokens": counts,
        "token_missing": missing,
        "errors": errors,
        "raw_errors": errors,
        "interrupt_candidates": candidates,
        "expected_interrupts": 0,
        "completed": when(root, "end_time") is not None,
        "unfinished_runs": sum(when(r, "end_time") is None for r in all_runs),
        "truncated": truncated,
        "interrupt_observed": has_interrupt,
        "steps": steps,
        "status": status,
        "review_action": action,
        "thread_hash": thread_hash,
        "batch_id": batch_id,
    }


def analyze_hitl(traces: list[dict[str, Any]], *, expect_online: bool, expect_all: bool) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """按 case 关联 Pause 与 Resume；不把没有 thread_id 的配对冒充已验证。"""
    by_case: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for trace in traces:
        if trace["phase"] in ("pause", "resume"):
            by_case[trace["case"]][trace["phase"]].append(trace)
    warnings: list[str] = []
    failures: list[str] = []
    pairs: list[dict[str, Any]] = []
    if expect_all:
        for missing in sorted(EXPECTED_HITL - set(by_case)):
            failures.append(f"缺少预期场景：{missing}")
    selected: dict[str, dict[str, Any]] = {}
    for case, phases in sorted(by_case.items()):
        selected[case] = {}
        for phase in ("pause", "resume"):
            candidates = sorted(phases.get(phase, []), key=lambda x: x["start_time"] or "", reverse=True)
            selected[case][phase] = candidates[0] if candidates else None
            if len(candidates) > 1:
                warnings.append(f"{case}/{phase} 存在 {len(candidates)} 条候选，仅检查最近一条；建议缩小 --since-hours 避免混批。")
    isolation_verified = False
    if "isolation-A" in selected and "isolation-B" in selected:
        hashes = [{t["thread_hash"] for t in selected[c].values() if t and t["thread_hash"]}
                  for c in ("isolation-A", "isolation-B")]
        if hashes[0] & hashes[1]:
            failures.append("isolation-A/isolation-B 复用了同一线程，未通过线程隔离验收")
        elif all(len(h) == 1 for h in hashes):
            isolation_verified = True
        else:
            warnings.append("isolation-A/isolation-B 线程证据不完整，无法核实线程隔离")
    for case, ordered in sorted(selected.items()):
        pause, resume = ordered["pause"], ordered["resume"]
        checks: list[str] = []
        if not pause or not resume:
            failures.append(f"{case} 缺少 pause 或 resume Run")
        else:
            if pause["truncated"] or resume["truncated"]:
                warnings.append(f"{case} 有 Trace 被 --max-spans 截断，节点/LLM 计数可能不完整")
            exp_action = "approve" if case == "isolation-A" else "edit" if case == "isolation-B" else case
            exp_status = EXPECTED_HITL_STATUS.get(case)
            wanted = ["prepare_draft", "human_review", exp_action, "finalize"]
            pairing_verified = True
            if pause["thread_hash"] and resume["thread_hash"]:
                if pause["thread_hash"] == resume["thread_hash"]:
                    checks.append("same_thread_id_verified")
                else:
                    failures.append(f"{case} Pause/Resume 的 thread_id 不一致")
                    pairing_verified = False
            else:
                warnings.append(f"{case} 的 Trace 未提供双方 thread_id；不能声称线程配对已证实")
                pairing_verified = False
            if pause.get("batch_id") and resume.get("batch_id"):
                if pause["batch_id"] == resume["batch_id"]:
                    checks.append("same_batch_verified")
                else:
                    failures.append(f"{case} Pause/Resume 的 batch_id 不一致")
                    pairing_verified = False
            else:
                warnings.append(f"{case} 缺少双方 batch_id，不能核实同批次配对")
                pairing_verified = False
            pause_start, pause_end = when(pause), when(pause, "end_time")
            resume_start, resume_end = when(resume), when(resume, "end_time")
            if all(t is not None for t in (pause_start, pause_end, resume_start, resume_end)):
                if pause_start <= pause_end <= resume_start <= resume_end and pause_start < resume_start:
                    checks.append("pause_before_resume_verified")
                else:
                    failures.append(f"{case} Pause/Resume 时间顺序异常或执行区间重叠")
                    pairing_verified = False
            else:
                warnings.append(f"{case} 缺少有效的起止时间，不能核实暂停恢复顺序")
                pairing_verified = False
            if case in ("isolation-A", "isolation-B"):
                if isolation_verified:
                    checks.append("isolation_threads_distinct_verified")
                else:
                    warnings.append(f"{case} 尚未核实与另一隔离场景使用不同线程")
                    pairing_verified = False
            # A candidate alone is insufficient: require a later completed resume
            # with the same explicit thread and batch, validated final state,
            # no errors, no truncation and no repeated generation.
            verified_interrupt = (
                case in EXPECTED_HITL and pairing_verified
                and not pause.get("unfinished_runs", 1) and not resume.get("unfinished_runs", 1)
                and pause.get("interrupt_candidates", 0) == pause["errors"] == 1
                and pause["status"] == "awaiting_review"
                and pause["steps"] == ["prepare_draft"]
                and not any(n in pause["nodes"] for n in ("approve", "reject", "edit", "finalize"))
                and pause.get("completed") and resume.get("completed")
                and not pause["truncated"] and not resume["truncated"]
                and not resume["errors"] and not resume["interrupt_observed"]
                and "human_review" in resume["nodes"]
                and "finalize" in resume["nodes"]
                and "prepare_draft" not in resume["nodes"]
                and not resume["llm_calls"]
                and (not expect_online or pause["llm_calls"] == 1)
                and resume["steps"] == wanted and resume["status"] == exp_status
                and resume["review_action"] == exp_action
            )
            if verified_interrupt:
                pause["expected_interrupts"] = 1
                pause["errors"] = pause["raw_errors"] - 1
                checks.append("expected_interrupt_verified")
            if pause["errors"] or resume["errors"]:
                failures.append(f"{case} 存在 Run 错误，需打开 LangSmith 检查")
            if not pause.get("completed", True) or not resume.get("completed", True):
                warnings.append(f"{case} 有根 Run 尚未结束，需等待 Trace 完整写入")
            if pause["interrupt_observed"] or verified_interrupt:
                checks.append("interrupt_observed")
            else:
                warnings.append(f"{case}/pause 的根输出未观察到 __interrupt__；仅凭 Trace 无法确认已暂停")
            if resume["interrupt_observed"]:
                failures.append(f"{case}/resume 仍有中断负载，未完成执行")
            if "human_review" in resume["nodes"]:
                checks.append("resume_review_node_observed")
            else:
                warnings.append(f"{case}/resume 未看到 human_review 子 Run")
            if resume["steps"] == wanted and resume["status"] == exp_status and resume["review_action"] == exp_action:
                checks.append("state_and_route_validated")
            else:
                failures.append(f"{case}/resume State 路径、审核动作或最终状态不符合预期（或 Trace 缺少这些字段）")
            if any(n in resume["nodes"] for n in ("prepare_draft",)):
                failures.append(f"{case}/resume 再次执行 prepare_draft；检查是否重复生成")
            if resume["llm_calls"]:
                failures.append(f"{case}/resume 出现 {resume['llm_calls']} 次 LLM 调用，检查重复生成")
            if expect_online and pause["llm_calls"] != 1:
                failures.append(f"{case}/pause 预期 1 次 LLM，实际 {pause['llm_calls']}")
        pairs.append({"case": case, "pause_run_id": pause["root_id"] if pause else None, "resume_run_id": resume["root_id"] if resume else None, "checks": checks, "thread_pairing": "verified" if "same_thread_id_verified" in checks else "unverified", "complete": bool(pause and resume)})
    if not pairs:
        failures.append("没有匹配的 chapter08-*-pause/resume 根 Run")
    return pairs, warnings, failures


def summarize(chapter: str, project: str, traces: list[dict[str, Any]], *, expect_online: bool, expect_all: bool, scan_limited: bool, batch_id: str | None = None) -> dict[str, Any]:
    # Work on fresh records: never mutate source evidence or reuse an exemption.
    traces = [{**t, "raw_errors": t.get("raw_errors", t["errors"]),
               "errors": t.get("raw_errors", t["errors"]), "expected_interrupts": 0}
              for t in traces]
    warnings, failures = [], []
    if not traces:
        failures.append("未检索到符合章节条件的根 Run")
    if scan_limited:
        warnings.append("达到 --max-roots 扫描上限，可能有根 Run 未获取；请缩短时间范围")
    pairs = []
    selected_root_ids: set[str] | None = None
    if chapter == "08":
        pairs, warns, fails = analyze_hitl(traces, expect_online=expect_online, expect_all=expect_all)
        warnings.extend(warns)
        failures.extend(fails)
        # 同一批次可能经历额度耗尽后重试。最终验收只看每个场景最近一对 pause/resume；
        # 旧失败仍显示为历史警告，且历史 LLM 用量仍计入本批成本，不会被悄悄删除。
        selected_root_ids = {
            str(x) for p in pairs for x in (p.get("pause_run_id"), p.get("resume_run_id")) if x
        }
    for trace in traces:
        if trace["errors"]:
            if selected_root_ids is not None and trace["root_id"] not in selected_root_ids:
                warnings.append(f"{trace['name']} 含历史重试失败 Run；保留成本统计，但不代表最新验收失败")
            else:
                failures.append(f"{trace['name']} 含 {trace['errors']} 条错误 Run")
        if trace.get("unfinished_runs", 1):
            warnings.append(f"{trace['name']} 有 {trace.get('unfinished_runs', '未知数量')} 条 Run 未记录有效结束时间，Trace 尚不完整")
        if trace["truncated"]:
            warnings.append(f"{trace['name']} 子 Run 达到上限，结果可能不完整")
        for field, count in trace["token_missing"].items():
            if count:
                warnings.append(f"{trace['name']} 有 {count} 个 LLM Run 缺少 {field} Token 计量")
    counts = {key: sum(t["tokens"][key] for t in traces) for key in ("input", "output", "total", "reasoning")}
    return {
        "schema_version": 2,
        "chapter": chapter,
        "project": project,
        "batch_id": batch_id,
        "trace_count": len(traces),
        "llm_calls": sum(t["llm_calls"] for t in traces),
        "token_totals_known": counts,
        "tokens_partial": any(any(v > 0 for v in t["token_missing"].values()) for t in traces),
        "root_runs": [t for t in sorted(traces, key=lambda x: x["start_time"] or "")],
        "hitl_pairs": pairs,
        "warnings": list(dict.fromkeys(warnings)),
        "failures": list(dict.fromkeys(failures)),
        "status": "fail" if failures else "review" if warnings else "pass",
    }


def markdown(report: dict[str, Any]) -> str:
    chapter = report["chapter"]
    header = [f"# LangSmith Trace 自动验收报告｜第 {chapter} 章", "", f"- 项目：`{report['project']}`", f"- 根 Graph Run：**{report['trace_count']}**", f"- 实际 LLM Run：**{report['llm_calls']}**（只统计 `run_type=llm`）", f"- 验收状态：**{report['status'].upper()}**（PASS / REVIEW / FAIL）", ""]
    t = report["token_totals_known"]
    if report.get("batch_id"):
        header.insert(4, f"- 实验批次：`{report['batch_id']}`")
    header.extend(["## Token 汇总", "", "| 输入 | 输出 | 其中 Reasoning | 总 Token |", "|---:|---:|---:|---:|", f"| {t['input']} | {t['output']} | {t['reasoning']} | {t['total']} |", ""])
    if report["tokens_partial"]:
        header.extend(["> **数据不完整：** 至少一条 LLM Run 缺少 Token 计量；各列仅为已知值合计。Reasoning 缺失不等于 0，也不表示 input/output/total 必然缺失；具体缺失字段见待复核项。", ""])
    header.extend(["## 根 Run 明细", "", "| Run | 阶段 | LLM Calls | Tokens（已知） | 原始 error | 预期中断 | 实际错误 |", "|---|---|---:|---:|---:|---:|---:|"])
    for r in report["root_runs"]:
        header.append(f"| `{r['name']}` | {r['phase'] or '—'} | {r['llm_calls']} | {r['tokens']['total']} | {r['raw_errors']} | {r['expected_interrupts']} | {r['errors']} |")
    header.extend(["", "原始 error 保留服务端计数；预期中断只在节点、State、同线程同批次及后续成功恢复全部核实后豁免。", ""])
    if chapter == "08":
        header.extend(["## HITL Pause / Resume 核对", "", "| 案例 | 成对 Trace | 可见的验证项 | thread_id 配对 |", "|---|---|---|---|"])
        for p in report["hitl_pairs"]:
            ck = ", ".join(p["checks"]) if p["checks"] else "—"
            header.append(f"| `{p['case']}` | {'是' if p['complete'] else '否'} | {ck} | {p['thread_pairing']} |")
        header.append("")
    if report["failures"]:
        header.extend(["## 失败项", "", *[f"- {x}" for x in report["failures"]], ""])
    if report["warnings"]:
        header.extend(["## 待复核 / 数据限制", "", *[f"- {x}" for x in report["warnings"]], ""])
    header.extend(["## 报告范围与隐私", "", "- 只生成脱敏摘要：不包含 Prompt、工单正文、草稿、模型回答或 API Key。", "- `thread_id` 只用短哈希匹配；未在 Trace 中出现时，明确标记证据不足。", "- 本工具是只读 Trace 分析，不负责证明真实的跨进程持久化或企业权限控制。", "- 同一时间窗口若含多轮实验，可能出现跨批次结果；建议运行实验后用较短的 `--since-hours`。", ""])
    return "\n".join(header)


def collect_bounded(iterator: Iterable[Any], cap: int) -> tuple[list[Any], bool]:
    """SDK 自动分页；本地多读 1 条，用来判断是否真的超过 cap。

    不传 `limit=cap` 给 LangSmith：API /runs/query 单页 limit 最大为 100，
    但我们的 --max-spans（默认 200）表示允许读取的总 Run 数。
    """
    if cap <= 0:
        raise ValueError("本地读取上限必须大于 0")
    items = list(islice(iter(iterator), cap + 1))
    return items[:cap], len(items) > cap


async def load_remote_async(client: Any, *, project: str, chapter: str, since: datetime, max_roots: int, max_spans: int, batch_id: str | None = None) -> tuple[list[dict[str, Any]], bool]:
    """只读新 API：显式时间窗口扫描根 Run，再按精确 Trace ID 获取完整树。"""
    project_record = await client.aread_project(project_name=project)
    project_id = str(value(project_record, "id"))
    all_roots, scan_limited = await fetch_roots(client, project_id=project_id, since=since, max_roots=max_roots)
    roots = []
    for root in all_roots:
        name = str(value(root, "name") or "")
        tags = value(root, "tags") or []
        meta = metadata(root)
        is_chapter = name.startswith(f"chapter{chapter}-") or f"chapter-{chapter}" in tags or str(meta.get("chapter", "")).startswith(chapter)
        if is_chapter and (batch_id is None or meta.get("batch_id") == batch_id):
            roots.append(root)
    results = []
    for i, root in enumerate(roots, 1):
        print(f"[{i}/{len(roots)}] 查询 {value(root, 'name')}", file=sys.stderr)
        trace_id = str(value(root, "trace_id") or value(root, "id"))
        actual_root, spans = await fetch_trace(client, project_id=project_id, trace_id=trace_id, max_spans=max_spans)
        results.append(summarize_trace(actual_root, spans, chapter, truncated=False))
    return results, scan_limited


def load_remote(client: Any, **kwargs) -> tuple[list[dict[str, Any]], bool]:
    """保留同步 CLI 入口；异步调用方直接使用 load_remote_async。"""
    return asyncio.run(load_remote_async(client, **kwargs))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="自动读取 LangSmith Trace 并生成脱敏的章节验收报告")
    p.add_argument("--chapter", choices=["07", "08", "09", "10", "11", "12", "13", "14"], required=True)
    p.add_argument("--project", help="LangSmith project；默认 LANGSMITH_PROJECT 或 zr-langChain")
    p.add_argument("--since-hours", type=float, default=2, help="查询最近 N 小时；默认 2")
    p.add_argument("--max-roots", type=int, default=100, help="最大扫描根 Run 数；默认 100")
    p.add_argument("--max-spans", type=int, default=200, help="每个 Trace 的最大 Run 数；默认 200")
    p.add_argument("--expect-online", action="store_true", help="第八章：要求每个 Pause 恰有一次 LLM")
    p.add_argument("--expect-all", action="store_true", help="第八章：要求五个工单线程都在时间窗口内出现")
    p.add_argument("--batch-id", help="只统计指定在线实验 batch_id 的根 Run，避免跨批次混算")
    p.add_argument("--batch-latest", action="store_true", help="读取 .runs/chapter08/latest.json 内的 batch_id，仅支持第八章")
    p.add_argument("--strict", action="store_true", help="有失败项则退出码 1；待复核项仍会生成报告")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_id and args.batch_latest:
        print("--batch-id 与 --batch-latest 只能二选一", file=sys.stderr)
        return 2
    if args.batch_latest and args.chapter != "08":
        print("--batch-latest 当前只支持第八章", file=sys.stderr)
        return 2
    batch_id = args.batch_id
    if args.batch_latest:
        try:
            batch = json.loads((ROOT / ".runs" / "chapter08" / "latest.json").read_text(encoding="utf-8"))
            batch_id = batch["batch_id"]
            if batch.get("chapter") != "08" or not isinstance(batch_id, str) or not batch_id.startswith("chapter08-"):
                raise ValueError("batch_id 格式异常")
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            print("无法读取合法的第八章最新在线批次，请先运行在线实验", file=sys.stderr)
            return 2
    if args.since_hours <= 0 or args.max_roots <= 0 or args.max_spans <= 0:
        print("参数错误：--since-hours / --max-roots / --max-spans 必须大于 0", file=sys.stderr)
        return 2
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    project = args.project or os.getenv("LANGSMITH_PROJECT", "zr-langChain")
    if not os.getenv("LANGSMITH_API_KEY"):
        print("缺少 LANGSMITH_API_KEY；请在仓库根目录 .env 配置，勿提交密钥。", file=sys.stderr)
        return 2
    try:
        from langsmith import Client  # 读取 LangSmith 数据；不会发起模型请求
    except ImportError:
        print("缺少 langsmith：在仓库根目录执行 uv add langsmith && uv sync", file=sys.stderr)
        return 2
    try:
        since = datetime.now(timezone.utc) - timedelta(hours=args.since_hours)
        rows, limited = load_remote(Client(), project=project, chapter=args.chapter, since=since,
                                    max_roots=args.max_roots, max_spans=args.max_spans, batch_id=batch_id)
        report = summarize(args.chapter, project, rows, expect_online=args.expect_online,
                           expect_all=args.expect_all, scan_limited=limited, batch_id=batch_id)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        base = f"chapter{args.chapter}-trace"
        md = args.output_dir / f"{base}-report.md"
        js = args.output_dir / f"{base}-summary.json"
        md.write_text(markdown(report), encoding="utf-8")
        js.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"验收状态：{report['status'].upper()}；Graph Runs={report['trace_count']}；LLM Runs={report['llm_calls']}")
        print(f"报告：{md}\n摘要：{js}")
        if report["failures"]:
            print("失败：" + "; ".join(report["failures"]), file=sys.stderr)
        if report["warnings"]:
            print(f"需复核：{len(report['warnings'])} 项（详见报告）", file=sys.stderr)
        return 1 if args.strict and report["failures"] else 0
    except TraceReadLimitExceeded as exc:
        # 只包含 trace_id 与内部上限信息，不包含原始 Prompt / 响应内容。
        print(f"Trace 数据不完整：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # 不直接打印可能包含原始输入、响应或凭据的异常字符串。
        status_code = None
        current: BaseException | None = exc
        for _ in range(5):
            if current is None:
                break
            response = getattr(current, "response", None)
            code = getattr(response, "status_code", None)
            if isinstance(code, int):
                status_code = code
                break
            current = current.__cause__ or current.__context__
        suffix = f"；HTTP {status_code}" if status_code is not None else ""
        print(f"Trace 查询或生成失败：{type(exc).__name__}{suffix}（检查项目名、权限及网络；没有重新请求模型）", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
