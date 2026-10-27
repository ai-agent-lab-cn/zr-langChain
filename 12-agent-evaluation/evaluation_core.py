"""第 12 章的纯 Python 评测契约：不导入 LangGraph、不访问外部服务。

目标：将参考标签与真正执行所得的证据分开；缺失证据不能得 PASS。
本文件里的阈值和字段是本项目定义的 Code Evaluator，并非 LangSmith 官方内置评分器。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

DATASET_VERSION = "ch12-golden-v1"
EVALUATOR_VERSION = "ch12-code-evaluator-v2"
Outcome = Literal["PASS", "FAIL", "REVIEW"]


@dataclass(frozen=True)
class GoldenCase:
    """冻结的虚构工单；参考字段只供 Evaluator 使用，不传给被测图。"""

    case_id: str
    ticket: str
    expected_tools: tuple[str, ...]
    expected_status: str = "advice_ready"


GOLDEN_DATASET: tuple[GoldenCase, ...] = (
    GoldenCase("account", "我的账号无法登录，一直提示密码错误。", ("lookup_account",)),
    GoldenCase("logistics", "包裹已发货，但是物流信息三天没有更新。", ("lookup_logistics",)),
    GoldenCase("business", "我想了解企业 VIP 套餐升级。", ("lookup_business",)),
    GoldenCase(
        "combined", "我的账号无法登录，还需要升级 VIP 套餐。",
        ("lookup_account", "lookup_business"),
    ),
)


@dataclass(frozen=True)
class Observation:
    """从实际 Graph State/AIMessage/ToolMessage 提取的脱敏观测。

    `tool_calls` 与 `tool_results` 必须直接从消息对象提取，不允许以最终回复反推。
    usage_metadata=None 表示没有真实模型数据，绝不能当作真实零 Token。
    """

    case_id: str
    status: str
    tool_calls: tuple[tuple[str, str], ...]
    tool_results: tuple[str, ...]
    steps: tuple[str, ...]
    effect_count: int = 0
    usage_metadata: dict[str, Any] | None = None
    message_protocol_valid: bool | None = None


@dataclass(frozen=True)
class Score:
    """逐样本评分：FAIL、证据不足 REVIEW、或有证据的 PASS。"""

    case_id: str
    outcome: Outcome
    checks: dict[str, bool | None]
    reasons: tuple[str, ...]


def evaluate_case(golden: GoldenCase, observed: Observation) -> Score:
    """确定性评测工具调用、ToolMessage 协议与状态；缺少轨迹时给 REVIEW。

    参数均为值对象；不接收模型 Prompt 或真实用户数据。返回每项检查的结果与原因。
    不会忽略副作用：任何未授权/意外 effect_count 都会判 FAIL。
    """
    if golden.case_id != observed.case_id:
        return Score(golden.case_id, "FAIL", {"case_id": False}, ("case_id 不匹配",))
    calls = observed.tool_calls
    call_ids = [call_id for _, call_id in calls]
    result_ids = list(observed.tool_results)
    protocol_ok = (
        bool(calls)
        and all(name and call_id for name, call_id in calls)
        and len(call_ids) == len(set(call_ids))
        and len(result_ids) == len(set(result_ids))
        and sorted(call_ids) == sorted(result_ids)
    )
    # 有完整工具调用轨迹时，工具选择与顺序按本数据集的明确参考要求评测。
    checks: dict[str, bool | None] = {
        "status": observed.status == golden.expected_status,
        "tool_protocol": (
            False if observed.message_protocol_valid is False or ((calls or result_ids) and not protocol_ok)
            else observed.message_protocol_valid if (calls or result_ids) else None
        ),
        "tool_trajectory": tuple(name for name, _ in calls) == golden.expected_tools if calls else None,
        "state_path": bool(observed.steps) and observed.steps[-1] == "finalize",
        "no_side_effect": observed.effect_count == 0,
    }
    failure = tuple(name for name, ok in checks.items() if ok is False)
    unknown = tuple(name for name, ok in checks.items() if ok is None)
    if failure:
        return Score(golden.case_id, "FAIL", checks, tuple(f"{name} 未通过" for name in failure))
    if unknown:
        return Score(golden.case_id, "REVIEW", checks, tuple(f"{name} 缺少证据" for name in unknown))
    return Score(golden.case_id, "PASS", checks, ())


def summarize(scores: list[Score], planned: int) -> dict[str, Any]:
    """汇总指标，显式区分 planned / executed / NOT_RUN；安全失败不能平均掉。"""
    if planned < 0 or len(scores) > planned:
        raise ValueError("非法计划/执行数量")
    if len({x.case_id for x in scores}) != len(scores):
        raise ValueError("存在重复 case_id，不能重复累计")
    counts = {key: sum(s.outcome == key for s in scores) for key in ("PASS", "FAIL", "REVIEW")}
    counts["NOT_RUN"] = planned - len(scores)
    verdict = "FAIL" if counts["FAIL"] else "REVIEW" if counts["REVIEW"] or counts["NOT_RUN"] else "PASS"
    return {
        "status": verdict,
        "planned": planned,
        "executed": len(scores),
        "counts": counts,
        "dataset_version": DATASET_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "pass_rate_over_planned": (counts["PASS"] / planned) if planned else None,
        "scores": [asdict(s) for s in scores],
    }


def compare_versions(base: list[Score], candidate: list[Score]) -> dict[str, Any]:
    """按同一 case_id 对齐基线和候选，拒绝缺失或重复，输出逐案进退。"""
    def index(items: list[Score]) -> dict[str, Score]:
        result = {item.case_id: item for item in items}
        if len(result) != len(items):
            raise ValueError("版本含重复 case_id")
        return result

    before, after = index(base), index(candidate)
    if not before or before.keys() != after.keys():
        raise ValueError("基线和候选数据集不一致，不能对比")
    rank = {"FAIL": 0, "REVIEW": 1, "PASS": 2}
    improvements = sorted(k for k in before if rank[after[k].outcome] > rank[before[k].outcome])
    regressions = sorted(k for k in before if rank[after[k].outcome] < rank[before[k].outcome])
    return {
        "aligned_cases": len(before),
        "baseline_pass": sum(s.outcome == "PASS" for s in base),
        "candidate_pass": sum(s.outcome == "PASS" for s in candidate),
        "improvements": improvements,
        "regressions": regressions,
        "status": (
            "FAIL" if regressions or any(s.outcome == "FAIL" for s in candidate)
            else "REVIEW" if any(s.outcome == "REVIEW" for s in candidate) else "PASS"
        ),
    }
