"""离线模拟真实 429、批次保存/恢复、模型更换保护及 main_async 快速终止。
无需 DeepSeek、LangGraph、LangSmith 依赖；不消耗任何 API Token。
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_errors import ErrorCode, classify_error
from experiment_runner import ExperimentBatch

QUOTA_ERROR = (
    "GoogleRateLimitError: 429 RESOURCE_EXHAUSTED "
    "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests "
    "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier; retryDelay: '4676s'"
)


class ExperimentErrorsTests(unittest.TestCase):
    def test_deepseek_insufficient_balance_stops_batch(self):
        for message in ("DeepSeek API 402 Insufficient Balance", "APIStatusError 402 payment required: insufficient_balance"):
            with self.subTest(message=message):
                result = classify_error(RuntimeError(message), phase="pause")
                self.assertEqual(result.code, ErrorCode.INSUFFICIENT_BALANCE)
                self.assertTrue(result.stop_batch)

    def test_deepseek_rate_limit_is_transient(self):
        result = classify_error(RuntimeError("DeepSeek RateLimitError 429 Rate Limit Reached"), phase="pause")
        self.assertEqual(result.code, ErrorCode.RATE_LIMIT_TRANSIENT)
        self.assertFalse(result.stop_batch)

    def test_deepseek_other_error_is_model_failure(self):
        class DeepSeekAPIError(Exception):
            pass
        result = classify_error(DeepSeekAPIError("server unavailable"), phase="pause")
        self.assertEqual(result.code, ErrorCode.MODEL_ERROR)

    def test_daily_quota_requires_immediate_stop(self):
        result = classify_error(RuntimeError(QUOTA_ERROR), phase="pause")
        self.assertEqual(result.code, ErrorCode.QUOTA_EXHAUSTED)
        self.assertTrue(result.stop_batch)
        self.assertEqual(result.retry_after_seconds, 4676)

    def test_plain_429_is_not_assumed_daily(self):
        result = classify_error(RuntimeError("429 rate limit: please retry in 10s"), phase="pause")
        self.assertEqual(result.code, ErrorCode.RATE_LIMIT_TRANSIENT)
        self.assertFalse(result.stop_batch)
        self.assertEqual(result.retry_after_seconds, 10)

    def test_parser_and_assertion_are_distinct(self):
        class OutputParserException(Exception):
            pass
        self.assertEqual(classify_error(OutputParserException("bad JSON"), phase="pause").code, ErrorCode.MODEL_ERROR)
        self.assertEqual(classify_error(ValueError("invalid graph state"), phase="pause").code, ErrorCode.WORKFLOW_ERROR)
        self.assertEqual(classify_error(AssertionError("unexpected result"), phase="resume").code, ErrorCode.ASSERTION_FAILED)
        self.assertEqual(classify_error(RuntimeError("bad state"), phase="resume").code, ErrorCode.WORKFLOW_ERROR)

    def test_never_copy_error_body_into_report(self):
        from experiment_runner import ExperimentBatch
        with tempfile.TemporaryDirectory() as directory:
            batch = ExperimentBatch.create(Path(directory)/"latest.json", chapter="08", model="m", source_hash="s")
            batch.start("approve")
            secret = "PROMPT_AND_API_SECRET_SHOULD_NEVER_BE_PERSISTED"
            batch.fail_case("approve", classify_error(RuntimeError(QUOTA_ERROR+secret), phase="pause"))
            contents = batch.path.read_text(encoding="utf-8")
            self.assertNotIn(secret, contents)
            self.assertNotIn("RESOURCE_EXHAUSTED", contents)
            self.assertIn("QUOTA_EXHAUSTED", contents)


class ExperimentBatchTests(unittest.TestCase):
    def test_resume_skips_pass_and_retries_pending_or_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"latest.json"
            first = ExperimentBatch.create(p, chapter="08", model="m", source_hash="abc")
            first.start("approve")
            first.pass_case("approve")
            first.start("reject")
            first.fail_case("reject", classify_error(RuntimeError(QUOTA_ERROR), phase="pause"))
            second = ExperimentBatch.resume(p, chapter="08", model="m", source_hash="abc")
            self.assertEqual(second.status("approve"), "PASS")
            self.assertEqual(second.status("reject"), "BLOCKED")
            self.assertEqual(second.status("edit"), "PENDING")
            with self.assertRaises(ValueError):
                second.start("approve")
            second.start("reject")
            second.pass_case("reject")
            self.assertEqual(second.data["cases"]["reject"]["attempts"], 2)

    def test_running_after_process_failure_becomes_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"latest.json"
            first = ExperimentBatch.create(p, chapter="08", model="m", source_hash="abc")
            first.start("isolation")
            second = ExperimentBatch.resume(p, chapter="08", model="m", source_hash="abc")
            self.assertEqual(second.status("isolation"), "PENDING")
            self.assertTrue(second.data["cases"]["isolation"]["interrupted_previous_process"])

    def test_different_model_or_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"latest.json"
            ExperimentBatch.create(p, chapter="08", model="m", source_hash="abc")
            for kwargs in [dict(chapter="08", model="other", source_hash="abc"),
                           dict(chapter="08", model="m", source_hash="def")]:
                with self.assertRaises(ValueError):
                    ExperimentBatch.resume(p, **kwargs)

    def test_atomic_file_is_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"latest.json"
            batch = ExperimentBatch.create(p, chapter="08", model="m", source_hash="abc")
            for case in ("approve", "reject", "edit", "isolation"):
                batch.start(case)
                batch.pass_case(case)
            self.assertEqual(json.loads(p.read_text())["status"], "PASS")
            self.assertEqual(batch.overall_status(), "PASS")


class MainOrchestrationTests(unittest.TestCase):
    """从真实 main.py AST 提取调度函数，用模拟 Graph 来测配额停止与断点跳过。"""

    def test_main_stops_on_quota_then_resumes_without_repeating_pass(self):
        source = (ROOT / "08-langgraph-hitl" / "main.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        node = next(x for x in tree.body if isinstance(x, ast.AsyncFunctionDef) and x.name == "main_async")
        mod = ast.Module(body=[node], type_ignores=[])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            namespace = {
                "__file__": str(ROOT / "08-langgraph-hitl" / "main.py"),
                "ROOT": folder, "os": os, "hashlib": __import__("hashlib"),
                "Path": Path, "ExperimentBatch": ExperimentBatch,
                "classify_error": classify_error,
                "ChatDeepSeek": lambda **kw: object(),
                "build_graph": lambda model: object(),
            }
            exec(compile(mod, "main_async", "exec"), namespace)
            executed = []
            async def failing_case(graph, case, **kw):
                executed.append(case)
                if case == "reject":
                    raise RuntimeError(QUOTA_ERROR)
            async def passing_isolation(graph, **kw):
                executed.append("isolation")
            namespace["run_case"] = failing_case
            namespace["run_isolation"] = passing_isolation
            args = argparse.Namespace(interactive=False,case="all",self_test=False,offline=False,resume_batch=False)
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-value", "DEEPSEEK_MODEL": "deepseek-flash"}):
                result = asyncio.run(namespace["main_async"](args))
                self.assertEqual(result, 1)
                self.assertEqual(executed, ["approve", "reject"])
                p = folder / ".runs" / "chapter08" / "latest.json"
                saved = json.loads(p.read_text(encoding="utf-8"))
                self.assertEqual(saved["cases"]["approve"]["status"], "PASS")
                self.assertEqual(saved["cases"]["reject"]["status"], "BLOCKED")
                self.assertEqual(saved["cases"]["edit"]["status"], "PENDING")
                self.assertEqual(saved["cases"]["isolation"]["status"], "PENDING")
                executed.clear()
                async def good_case(graph, case, **kw):
                    executed.append(case)
                namespace["run_case"] = good_case
                args.resume_batch = True
                result = asyncio.run(namespace["main_async"](args))
                self.assertEqual(result, 0)
                self.assertEqual(executed, ["reject", "edit", "isolation"])
                self.assertEqual(json.loads(p.read_text())["status"], "PASS")


if __name__ == "__main__":
    unittest.main()

# 保持 unittest discover 时通过类发现；不要把测试放在 __main__ 块后。
class AdditionalQuotaSafetyTest(unittest.TestCase):
    def test_free_tier_metric_is_not_sufficient_for_daily_quota(self):
        # 同一指标名也可能遭遇短时配额；必须有每日 Quota ID 或其他日限额证据。
        error = RuntimeError("429 RESOURCE_EXHAUSTED; generate_content_free_tier_requests; retryDelay: '15s'")
        classified = classify_error(error, phase="pause")
        self.assertEqual(classified.code, ErrorCode.RATE_LIMIT_TRANSIENT)
        self.assertFalse(classified.stop_batch)
