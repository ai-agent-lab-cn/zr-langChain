"""审计第十一章本地脱敏 Trace；不读取 LangSmith，也不替代业务断言。

配套流程图：同目录 basic-knowledge.md，本地证据审计（trace）。
"""

import math


_OBSERVED_STATES = {"SUCCESS", "FAILED", "DEGRADED", "REJECTED", "WAITING_APPROVAL"}
_RUN_TYPES = {"chain", "llm", "tool", "retriever", "prompt", "embedding", "parser"}


def complete_fixture() -> list[dict]:
    """返回独立的三条教学 Run，每次调用新建字典，不共享修改。

    root 的 999 是故意放入的父级聚合干扰项；只有 model-1 的 12 参与计量。
    这些值均为预设夹具，不是在线 Token；id/batch/thread 也都不是服务端标识。"""
    shared = {"batch_id": "fixture-batch", "thread_id": "fixture-thread",
              "status": "SUCCESS", "end_time": 1}
    return [
        dict(shared, id="root", parent_id=None, run_type="chain", total_tokens=999),
        dict(shared, id="model-1", parent_id="root", run_type="llm", total_tokens=12),
        dict(shared, id="tool-1", parent_id="root", run_type="tool"),
    ]


def audit_trace_fixture(records: list[dict], *, batch_id: str, complete: bool = True) -> dict:
    """按本地 schema 校验脱敏 Trace 字典列表，返回证据级审计报告。

    输入 records 可包含重复分页项；batch_id 指定应属于的批次；complete
    仅声明是否拿到了全部分页，不代表 LangSmith 已验收。输入始终不修改。
    处理顺序：容器/标记 → 每项字段 → ID 去重 → LLM 用量 → 父子关系 → 环检测。

    status 优先级为 FAIL > REVIEW > PASS：类型/内容矛盾 FAIL，证据缺口 REVIEW。
    token_usage 只对可识别、唯一的 LLM 求和，父 Graph 聚合值不再相加。
    缺 usage、未知 Run 类型或缺页使总量为 None；missing_usage 仅数已见的
    缺用量模型记录，不能据它为 0 推断缺页中没有模型调用。

    业务 FAILED/DEGRADED/WAITING_APPROVAL 只要证据合法仍可审计 PASS；
    业务是否符合要求须另看场景断言。source 固定 local_fixture，不读取服务端。"""
    issues = []
    failed = False
    # failed 决定 FAIL 优先级；unknown_usage 独立决定能否给出完整用量。
    unknown_usage = False
    missing_usage = 0
    total = 0

    def issue(message: str, *, contradiction: bool = False) -> None:
        """收集固定脱敏诊断；不要在错误中回显任意输入值。"""
        nonlocal failed
        issues.append(message)
        failed = failed or contradiction

    def result() -> dict:
        """统一输出，避免缺证据时误给出零用量或成功。"""
        return {
            "status": "FAIL" if failed else "REVIEW" if issues else "PASS",
            "token_usage": None if unknown_usage else total,
            "missing_usage": missing_usage,
            "issues": issues,
            "source": "local_fixture",
        }

    if type(batch_id) is not str or not batch_id.strip():
        issue("batch_id 参数必须是非空字符串", contradiction=True)
    if type(complete) is not bool:
        issue("complete 参数必须是 bool", contradiction=True)
    elif not complete:
        issue("分页证据不完整")
        unknown_usage = True
    if type(records) is not list:
        issue("records 必须是 list[dict]", contradiction=True)
        unknown_usage = True
        return result()
    if not records:
        issue("没有 Trace 样本或根调用证据")
        unknown_usage = True
        return result()

    # 可识别 ID 的记录才能去重；无法识别的项仍需检查，不能悄悄丢弃。
    unique = {}
    unidentified = []
    for index, record in enumerate(records):
        label = f"记录 {index + 1}"
        if type(record) is not dict:
            issue(f"{label} 必须是 dict", contradiction=True)
            unknown_usage = True
            continue

        # 先验证每份记录，再去重；否则 bool/int 相等可使畸形重复项绕过验证。
        valid_strings = {}
        for field in ("id", "batch_id", "thread_id", "run_type", "status"):
            value = record.get(field)
            if value is None or (type(value) is str and not value.strip()):
                issue(f"{label} 缺少 {field}")
                valid_strings[field] = False
            elif type(value) is not str:
                issue(f"{label} 的 {field} 类型非法", contradiction=True)
                valid_strings[field] = False
            else:
                valid_strings[field] = True
        if not valid_strings["id"] or not valid_strings["run_type"]:
            # 未知种类的 Run 可能就是缺失的模型调用；不能假定它贡献零 Token。
            unknown_usage = True
        if valid_strings["run_type"] and record["run_type"] not in _RUN_TYPES:
            # 类型拼写错误可能掩盖模型调用，不能把未知种类默认为零用量。
            issue(f"{label} 的 run_type 不属于本地夹具 schema", contradiction=True)
            unknown_usage = True
        if valid_strings["batch_id"] and record["batch_id"] != batch_id:
            issue(f"{label} 与请求批次不一致", contradiction=True)
        if valid_strings["status"] and record["status"] not in _OBSERVED_STATES:
            issue(f"{label} 状态非法", contradiction=True)
        if "parent_id" not in record:
            issue(f"{label} 缺少 parent_id")
        elif record["parent_id"] is not None and (
            type(record["parent_id"]) is not str or not record["parent_id"].strip()
        ):
            issue(f"{label} 的 parent_id 类型非法", contradiction=True)
        ended = record.get("end_time")
        if ended is None:
            issue(f"{label} 缺少 end_time")
        elif type(ended) not in (int, float) or (type(ended) is float and not math.isfinite(ended)):
            issue(f"{label} 的 end_time 必须是有限数值", contradiction=True)
        if "total_tokens" in record:
            tokens = record.get("total_tokens")
            if tokens is not None and (type(tokens) is not int or tokens < 0):
                issue(f"{label} 的 Token 必须是非负整数且不能是 bool", contradiction=True)
                if record.get("run_type") == "llm":
                    unknown_usage = True

        if valid_strings["id"]:
            identity = record["id"]
            if identity in unique:
                if unique[identity] != record:
                    issue("相同 Run ID 的重复证据冲突", contradiction=True)
                    unknown_usage = True
                continue
            unique[identity] = record
        else:
            unidentified.append(record)

    # 先做唯一化再计量，避免分页重复；不要将 root.total_tokens 与叶子再相加。
    for record in [*unique.values(), *unidentified]:
        if record.get("run_type") == "llm":
            tokens = record.get("total_tokens")
            if tokens is None:
                issue("LLM 缺少 usage；失败请求也不能按零统计")
                missing_usage += 1
                unknown_usage = True
            elif type(tokens) is int and tokens >= 0:
                total += tokens

    # 即使 LLM 用量都在，缺根/缺父也只能给出受限证据，不能宣布链路完整。
    if not any("parent_id" in record and record["parent_id"] is None
               for record in unique.values()):
        issue("缺少可识别的根调用")
    for record in unique.values():
        parent_id = record.get("parent_id")
        if type(parent_id) is not str or not parent_id:
            continue
        parent = unique.get(parent_id)
        if parent is None:
            issue("非根调用缺少父级证据")
            continue
        for field in ("batch_id", "thread_id"):
            child_value, parent_value = record.get(field), parent.get(field)
            if type(child_value) is str and type(parent_value) is str and child_value != parent_value:
                issue(f"父子调用的 {field} 不一致", contradiction=True)

    # 迭代沿父链检查所有分量，既捕获独立环，也避免深链导致递归溢出。
    visited = set()
    for identity in unique:
        path = set()
        current = identity
        while type(current) is str and current in unique and current not in visited:
            if current in path:
                issue("父子调用形成环", contradiction=True)
                break
            path.add(current)
            current = unique[current].get("parent_id")
        visited.update(path)
    return result()
