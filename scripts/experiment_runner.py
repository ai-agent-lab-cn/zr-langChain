"""项目级在线实验批次记录器（Python 标准库，独立于 LangGraph）。

重要区别：这是“重新执行未通过的用例”，不是恢复 InMemorySaver 内已暂停的节点。
仅存储匿名批次状态；不存储工单原文、草稿、模型回答或 API Key。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from uuid import uuid4

from experiment_errors import ExperimentFailure


CASE_NAMES = ("approve", "reject", "edit", "isolation")
STATE_VALUES = {"PENDING", "RUNNING", "PASS", "FAILED", "BLOCKED"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ExperimentBatch:
    path: Path
    data: dict[str, Any]

    @classmethod
    def create(cls, path: Path, *, chapter: str, model: str, source_hash: str) -> "ExperimentBatch":
        data = {
            "schema_version": 1, "batch_id": f"chapter{chapter}-{uuid4().hex[:16]}",
            "chapter": chapter, "mode": "online", "model": model,
            "source_hash": source_hash, "created_at": now(), "updated_at": now(),
            "cases": {case: {"status": "PENDING", "attempts": 0} for case in CASE_NAMES},
        }
        item = cls(path, data)
        item.save()
        return item

    @classmethod
    def resume(cls, path: Path, *, chapter: str, model: str, source_hash: str) -> "ExperimentBatch":
        if not path.is_file():
            raise ValueError("没有可恢复的实验批次；请先运行一次不带 --resume-batch 的在线实验")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("本地批次文件不可读取或 JSON 无效，请备份后重新创建批次") from exc
        if (data.get("schema_version") != 1 or data.get("chapter") != chapter
                or data.get("mode") != "online" or data.get("model") != model
                or data.get("source_hash") != source_hash):
            raise ValueError("批次的章节/模型/脚本版本不一致，不能安全跳过旧 PASS；请新建批次")
        cases = data.get("cases")
        if not isinstance(cases, dict) or set(cases) != set(CASE_NAMES):
            raise ValueError("批次文件的场景清单异常")
        for case in CASE_NAMES:
            record = cases[case]
            if not isinstance(record, dict) or record.get("status") not in STATE_VALUES or not isinstance(record.get("attempts"), int):
                raise ValueError("批次文件的场景状态异常")
            # 前一进程中断时的 RUNNING 只能重新执行该场景，不能恢复内存 Graph 节点。
            if record["status"] == "RUNNING":
                record["status"] = "PENDING"
                record["interrupted_previous_process"] = True
        item = cls(path, data)
        item.save()
        return item

    @property
    def batch_id(self) -> str:
        return str(self.data["batch_id"])

    def status(self, case: str) -> str:
        return self.data["cases"][case]["status"]

    def start(self, case: str) -> None:
        record = self.data["cases"][case]
        if record["status"] == "PASS":
            raise ValueError("已通过场景不应再次开始；若需重新验证，请创建新批次")
        record["status"] = "RUNNING"
        record["attempts"] += 1
        record["started_at"] = now()
        record.pop("error", None)
        self.save()

    def pass_case(self, case: str) -> None:
        self.data["cases"][case]["status"] = "PASS"
        self.data["cases"][case]["finished_at"] = now()
        self.save()

    def fail_case(self, case: str, failure: ExperimentFailure) -> None:
        record = self.data["cases"][case]
        record["status"] = "BLOCKED" if failure.stop_batch else "FAILED"
        record["finished_at"] = now()
        record["error"] = {"code": failure.code.value, "phase": failure.phase,
                           "retry_after_seconds": failure.retry_after_seconds}
        self.save()

    def overall_status(self) -> str:
        values = [self.status(case) for case in CASE_NAMES]
        if all(x == "PASS" for x in values):
            return "PASS"
        if "BLOCKED" in values:
            return "BLOCKED"
        if "FAILED" in values:
            return "FAILED"
        return "INCOMPLETE"

    def save(self) -> None:
        """写临时文件并原子替换，减少异常退出造成的半写 JSON。"""
        self.data["updated_at"] = now()
        self.data["status"] = self.overall_status()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".batch-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self.data, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.chmod(name, 0o600)
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
