"""第 12 章：纯 Code Evaluator 可独立测；LangGraph 相关测试需要真实依赖。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from dataclasses import replace
from pathlib import Path

CHAPTER = Path(__file__).resolve().parents[1] / "12-agent-evaluation"
sys.path.insert(0, str(CHAPTER))

# 避免项目其它章节可能同样使用 `main.py` 导致 sys.modules["main"] 冲突。
_ENTRY_SPEC = importlib.util.spec_from_file_location("chapter12_main_entry", CHAPTER / "main.py")
assert _ENTRY_SPEC is not None and _ENTRY_SPEC.loader is not None
chapter_main = importlib.util.module_from_spec(_ENTRY_SPEC)
_ENTRY_SPEC.loader.exec_module(chapter_main)

from evaluation_core import (  # noqa: E402
    GOLDEN_DATASET, Observation, Score, compare_versions, evaluate_case, summarize,
)


def valid_observation():
    return Observation(
        case_id="account", status="advice_ready",
        tool_calls=(("lookup_account", "test-tool-1"),),
        tool_results=("test-tool-1",),
        steps=("route", "supervisor:lookup_account", "tool_result", "finalize"),
        message_protocol_valid=True,
    )


class EvaluatorContractTest(unittest.TestCase):
    def test_valid_trajectory_pass(self):
        self.assertEqual(evaluate_case(GOLDEN_DATASET[0], valid_observation()).outcome, "PASS")

    def test_tool_call_missing_evidence_review(self):
        obs = replace(valid_observation(), tool_calls=(), tool_results=())
        self.assertEqual(evaluate_case(GOLDEN_DATASET[0], obs).outcome, "REVIEW")

    def test_orphan_result_fail(self):
        obs = replace(valid_observation(), tool_results=("bad-id",))
        self.assertEqual(evaluate_case(GOLDEN_DATASET[0], obs).outcome, "FAIL")

    def test_duplicate_call_id_fail(self):
        obs = replace(valid_observation(), tool_calls=(("lookup_account", "same"), ("lookup_account", "same")), tool_results=("same",))
        self.assertEqual(evaluate_case(GOLDEN_DATASET[0], obs).outcome, "FAIL")

    def test_wrong_tool_fail(self):
        obs = replace(valid_observation(), tool_calls=(("lookup_business", "test-tool-1"),))
        self.assertEqual(evaluate_case(GOLDEN_DATASET[0], obs).outcome, "FAIL")

    def test_illegal_side_effect_fail(self):
        self.assertEqual(evaluate_case(GOLDEN_DATASET[0], replace(valid_observation(), effect_count=1)).outcome, "FAIL")

    def test_no_final_path_fail(self):
        self.assertEqual(evaluate_case(GOLDEN_DATASET[0], replace(valid_observation(), steps=("route",))).outcome, "FAIL")

    def test_case_mismatch_fail(self):
        self.assertEqual(evaluate_case(GOLDEN_DATASET[0], replace(valid_observation(), case_id="other")).outcome, "FAIL")

    def test_missing_usage_not_misrepresented(self):
        self.assertIsNone(valid_observation().usage_metadata)

    def test_summarize_coverage(self):
        summary = summarize([Score("account", "PASS", {}, ())], 4)
        self.assertEqual(summary["counts"]["NOT_RUN"], 3)
        self.assertEqual(summary["status"], "REVIEW")

    def test_no_silent_safety_fail(self):
        summary = summarize([Score("account", "FAIL", {"no_side_effect": False}, ())], 1)
        self.assertEqual(summary["status"], "FAIL")

    def test_duplicate_case_rejected(self):
        with self.assertRaises(ValueError):
            summarize([Score("x", "PASS", {}, ()), Score("x", "PASS", {}, ())], 2)

    def test_regression_alignment(self):
        base = [Score("a", "PASS", {}, ()), Score("b", "FAIL", {}, ())]
        new = [Score("a", "PASS", {}, ()), Score("b", "PASS", {}, ())]
        self.assertEqual(compare_versions(base, new)["improvements"], ["b"])

    def test_missing_pair_rejected(self):
        with self.assertRaises(ValueError):
            compare_versions([Score("a", "PASS", {}, ())], [Score("b", "PASS", {}, ())])

    def test_regression_fails(self):
        base = [Score("a", "PASS", {}, ())]
        new = [Score("a", "FAIL", {}, ())]
        self.assertEqual(compare_versions(base, new)["status"], "FAIL")


class CliContractTest(unittest.TestCase):
    """入口行为通过替换场景函数测试；不是 LangGraph 行为验证。"""

    def test_existing_evidence_never_overwritten(self):
        import contextlib
        import io
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "evidence.json"
            file.write_text("original", encoding="utf-8")
            with patch.object(sys, "argv", ["main.py", "--case", "A", "--evidence", str(file)]):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(chapter_main.main(), 1)
            self.assertEqual(file.read_text(encoding="utf-8"), "original")

    def test_failed_case_blocks_later_cases(self):
        import contextlib
        import io
        import json
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "evidence.json"
            called = []

            def fail():
                called.append("A")
                raise ValueError("private-details-must-not-be-printed")

            def should_not_run():
                called.append("other")
                return {"status": "PASS"}

            with patch.object(sys, "argv", ["main.py", "--self-test", "--evidence", str(file)]):
                with patch.dict(chapter_main.RUNNERS, {"A": fail, "B": should_not_run, "C": should_not_run, "D": should_not_run}):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                        self.assertEqual(chapter_main.main(), 1)
            data = json.loads(file.read_text(encoding="utf-8"))
            self.assertEqual(called, ["A"])
            self.assertEqual(data["cases"]["B"]["status"], "NOT_RUN")
            self.assertEqual(data["cases"]["A"]["error_type"], "ValueError")
            self.assertNotIn("private-details", file.read_text(encoding="utf-8") + output.getvalue())

    def test_single_case_no_llm_or_tracing(self):
        import contextlib
        import io
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "new.json"
            with patch.object(sys, "argv", ["main.py", "--case", "A", "--evidence", str(file)]):
                with patch.dict(chapter_main.RUNNERS, {"A": lambda: {"status": "PASS", "graph_invokes": 0}}):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(chapter_main.main(), 0)
            import json
            data = json.loads(file.read_text(encoding="utf-8"))
            self.assertEqual(data["provider_calls"], 0)
            self.assertEqual(data["langsmith_traces_uploaded"], 0)
            self.assertIsNone(data["usage_metadata"])
            self.assertEqual(tuple(data["cases"]), ("A",))


@unittest.skipUnless(importlib.util.find_spec("langgraph") and importlib.util.find_spec("langchain_core"),
                     "当前环境没有 LangGraph；真实图行为测试未执行，不能宣称通过")
class RealLangGraphTest(unittest.TestCase):
    def test_a_golden(self):
        self.assertEqual(chapter_main.run_a()["status"], "PASS")

    def test_b_protocol(self):
        self.assertEqual(chapter_main.run_b()["status"], "PASS")

    def test_c_human_review(self):
        self.assertEqual(chapter_main.run_c()["status"], "PASS")

    def test_d_regression(self):
        self.assertEqual(chapter_main.run_d()["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
