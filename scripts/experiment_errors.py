"""实验错误分类（无第三方依赖，不把供应商的原始响应写进报告）。

只对证据明确的配额耗尽或 DeepSeek 402 余额不足触发整批终止；普通 429 仍归为瞬时限流。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class ErrorCode(str, Enum):
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    INSUFFICIENT_BALANCE = "INSUFFICIENT_BALANCE"
    RATE_LIMIT_TRANSIENT = "RATE_LIMIT_TRANSIENT"
    MODEL_ERROR = "MODEL_ERROR"
    WORKFLOW_ERROR = "WORKFLOW_ERROR"
    ASSERTION_FAILED = "ASSERTION_FAILED"


@dataclass(frozen=True)
class ExperimentFailure:
    code: ErrorCode
    phase: str
    retry_after_seconds: int | None = None
    stop_batch: bool = False


_DAILY_EVIDENCE = (
    "generaterequestsperday", "requestsperday", "per_day", "per-day",
    "free-tier daily", "daily quota", "per day quota",
)
_RETRY_DELAY = re.compile(r"[\"']?retryDelay[\"']?\s*[:=]\s*[\"']?(\d+)s", re.I)
_RETRY_TEXT = re.compile(r"please retry in\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+(?:\.\d+)?)s)?", re.I)


def _error_chain(exc: BaseException) -> list[BaseException]:
    """读取少量因果链，兼容 API SDK 包装过的底层错误。"""
    seen: set[int] = set()
    result: list[BaseException] = []
    while exc is not None and len(result) < 4 and id(exc) not in seen:
        seen.add(id(exc))
        result.append(exc)
        exc = exc.__cause__ or exc.__context__
    return result


def _retry_seconds(text: str) -> int | None:
    match = _RETRY_DELAY.search(text)
    if match:
        return int(match.group(1))
    match = _RETRY_TEXT.search(text)
    if match and any(match.groups()):
        hours, minutes, seconds = match.groups()
        return int(int(hours or 0)*3600 + int(minutes or 0)*60 + float(seconds or 0))
    return None


def classify_error(exc: BaseException, *, phase: str = "unknown") -> ExperimentFailure:
    """将异常变为可序列化的安全元数据；不暴露 API 原始错误、Prompt 或业务内容。"""
    chain = _error_chain(exc)
    message = " ".join(f"{type(e).__name__} {str(e)}" for e in chain)
    lower = message.lower()
    is_limit = any(x in lower for x in ("429", "resource_exhausted", "ratelimit", "rate limit", "quota exceeded"))
    # DeepSeek 用 HTTP 402 表示余额不足，不应按普通 429 稍后重试。
    if any(x in lower for x in ("402 payment required", "402 insufficient", "insufficient balance", "insufficient_balance", "not enough balance")):
        return ExperimentFailure(ErrorCode.INSUFFICIENT_BALANCE, phase, stop_batch=True)
    if is_limit and any(x in lower for x in _DAILY_EVIDENCE):
        return ExperimentFailure(ErrorCode.QUOTA_EXHAUSTED, phase, _retry_seconds(message), stop_batch=True)
    if is_limit:
        return ExperimentFailure(ErrorCode.RATE_LIMIT_TRANSIENT, phase, _retry_seconds(message))
    if any(isinstance(e, AssertionError) for e in chain):
        return ExperimentFailure(ErrorCode.ASSERTION_FAILED, phase)
    # SDK / Pydantic 解析错误或 HTTP 服务端错误；具体来源和请求内容不写入状态文件。
    model_names = ("google", "gemini", "deepseek", "openai", "apierror", "validationerror", "outputparser", "model", "httpstatus", "jsondecode")
    if (any(any(n in type(e).__name__.lower() for n in model_names) for e in chain)
            or "deepseek 未返回预期" in lower or "gemini 未返回预期" in lower):
        return ExperimentFailure(ErrorCode.MODEL_ERROR, phase)
    return ExperimentFailure(ErrorCode.WORKFLOW_ERROR, phase)
