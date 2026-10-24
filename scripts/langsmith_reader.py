"""LangSmith SmithDB 只读适配层：新异步 API → 既有验收字段。

不回退旧 read_run/list_runs，不发模型请求，不隐藏缺失 Token。
新 API 默认只选 ID，因此显式选择验收所需字段，且不拉取 inputs。
"""
from datetime import datetime, timezone
from types import SimpleNamespace

FIELDS = ["ID", "NAME", "RUN_TYPE", "STATUS", "START_TIME", "END_TIME", "ERROR",
          "OUTPUTS", "EXTRA", "METADATA", "PARENT_RUN_IDS", "PROJECT_ID", "TRACE_ID",
          "PROMPT_TOKENS", "COMPLETION_TOKENS", "TOTAL_TOKENS", "COMPLETION_TOKEN_DETAILS",
          "TAGS"]


class TraceReadLimitExceeded(RuntimeError):
    """完整 Trace 超出本地上限，不能用部分证据生成 PASS。"""


def field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def normalize_run(raw):
    """隔离 v2 字段变化，让已有章节报告继续使用同一分析规则。"""
    kind = field(raw, "run_type")
    if not isinstance(kind, str) or not kind:
        raise ValueError("Run 缺少显式选取的 RUN_TYPE")
    ancestors = field(raw, "parent_run_ids") or []
    values = {name.lower(): field(raw, name.lower()) for name in FIELDS}
    values["run_type"] = kind.lower()
    values["parent_run_ids"] = [str(x) for x in ancestors]
    values["parent_run_id"] = str(ancestors[-1]) if ancestors else None
    details = field(raw, "completion_token_details")
    # v2 将 token details 包装在 .raw；缺失值仍保留 None。
    values["completion_token_details"] = field(details, "raw", details)
    # 不生成或公开私有 Trace URL；本地验收可用 run_id 在 LangSmith 检索。
    values["url"] = None
    return SimpleNamespace(**values)


async def fetch_roots(client, *, project_id, since, max_roots):
    """SDK AsyncPaginator 自动跨页；多读一条判定根查询是否达到本地上限。"""
    if max_roots <= 0:
        raise ValueError("max_roots 必须大于0")
    rows = []
    async for raw in client.runs.query(project_ids=[project_id], is_root=True,
                                      min_start_time=since, max_start_time=datetime.now(timezone.utc),
                                      selects=FIELDS, page_size=min(100, max_roots + 1)):
        if len(rows) == max_roots:
            return rows, True
        rows.append(normalize_run(raw))
    return rows, False


async def fetch_trace(client, *, project_id, trace_id, max_spans=200):
    """按精确 trace_id 一次读取调用树，不使用默认24小时的 runs.query。

    traces.list_runs 返回该 Trace 的完整 items（非分页接口）。本地上限是分析
    边界，不能限制服务端单次响应体大小。缺根、缺父、跨 Trace、重复 ID 均拒绝。
    """
    if max_spans <= 0:
        raise ValueError("max_spans 必须大于0")
    response = await client.traces.list_runs(str(trace_id), project_id=str(project_id), selects=FIELDS)
    raw = response.items or []
    if len(raw) > max_spans:
        raise TraceReadLimitExceeded(f"Trace Run数超过本地上限 {max_spans}，拒绝部分验收")
    rows = [normalize_run(r) for r in raw]
    by_id = {str(r.id): r for r in rows}
    if len(by_id) != len(rows) or str(trace_id) not in by_id:
        raise ValueError("Trace 缺少根 Run 或存在重复ID")
    root = by_id[str(trace_id)]
    if root.parent_run_ids:
        raise ValueError("指定ID不是根 Run")
    for run in rows:
        if str(run.trace_id) != str(trace_id):
            raise ValueError("Trace ID 不匹配")
        if str(run.id) != str(trace_id) and (not run.parent_run_ids or run.parent_run_ids[0] != str(trace_id)):
            raise ValueError("非根 Run 缺少完整祖先链")
        if str(run.id) in run.parent_run_ids or any(p not in by_id for p in run.parent_run_ids):
            raise ValueError("Trace 祖先链缺失或自引用")
        if run.parent_run_id and by_id[run.parent_run_id].parent_run_ids != run.parent_run_ids[:-1]:
            raise ValueError("父子祖先链不一致")
    return root, rows
