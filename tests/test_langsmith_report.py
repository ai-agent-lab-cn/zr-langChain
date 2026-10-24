"""不需要 API Key、LangSmith SDK 或 DeepSeek：用内存模拟 Run 验证分析逻辑。"""
from datetime import datetime, timezone, timedelta
import importlib.util
from pathlib import Path
import sys
import unittest
from uuid import uuid4
from types import SimpleNamespace

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "langsmith_report.py"
spec = importlib.util.spec_from_file_location("langsmith_report", MODULE_PATH)
reporter = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = reporter
sys.path.insert(0, str(MODULE_PATH.parent))
spec.loader.exec_module(reporter)


def fake_run(name, typ="chain", *, parent=None, trace=None, outputs=None, meta=None,
             tokens=None, run_id=None, error=None):
    rid = run_id or str(uuid4())
    return {"id": rid, "trace_id": trace or rid, "parent_run_id": parent, "name": name,
            "run_type": typ, "outputs": outputs or {}, "extra": {"metadata": meta or {}},
            "error": error, "start_time": datetime(2026, 10, 23, tzinfo=timezone.utc),
            "end_time": datetime(2026, 10, 23, tzinfo=timezone.utc) + timedelta(seconds=1),
            **(tokens or {})}


def fixture(case="approve", *, thread="a-unique-thread", online=True):
    action = "approve" if case == "isolation-A" else "edit" if case == "isolation-B" else case
    status = {"approve": "approved", "reject": "rejected", "edit": "edited"}[action]
    pause = fake_run(f"chapter08-{case}-pause", outputs={"__interrupt__": [{"value": {"actions": ["approve", "reject", "edit"]}}]}, meta={"thread_id": thread, "batch_id": "chapter08-test"})
    pause_children = [fake_run("prepare_draft", trace=pause["id"], parent=pause["id"]),
                      fake_run("human_review", trace=pause["id"], parent=pause["id"])]
    if online:
        pause_children.append(fake_run("ChatDeepSeek", "llm", trace=pause["id"], parent=pause_children[0]["id"],
                                       tokens={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                                               "completion_token_details": {"reasoning": 12}}))
    resume = fake_run(f"chapter08-{case}-resume", outputs={"status": status, "review_action": action,
                      "steps": ["prepare_draft", "human_review", action, "finalize"]}, meta={"thread_id": thread, "batch_id": "chapter08-test"})
    resume["start_time"] += timedelta(seconds=2)
    resume["end_time"] += timedelta(seconds=2)
    resume_children = [fake_run("human_review", trace=resume["id"], parent=resume["id"]),
                       fake_run(action, trace=resume["id"], parent=resume["id"]),
                       fake_run("finalize", trace=resume["id"], parent=resume["id"])]
    return [reporter.summarize_trace(pause, pause_children, "08"), reporter.summarize_trace(resume, resume_children, "08")]


class NewAPIClient:
    def __init__(self, roots, traces):
        self.roots, self.trace_rows = roots, traces
        self.runs = self
        self.traces = self

    async def aread_project(self, **kwargs):
        return SimpleNamespace(id="test-project")

    @staticmethod
    def v2(row):
        parents = [row['parent_run_id']] if row.get('parent_run_id') else []
        return {**row, 'run_type': row['run_type'].upper(), 'parent_run_ids': parents}

    def query(self, **kwargs):
        async def pages():
            for row in self.roots:
                yield self.v2(row)
        return pages()

    async def list_runs(self, trace_id, **kwargs):
        return SimpleNamespace(items=[self.v2(row) for row in self.trace_rows[trace_id]])


class TestTraceAnalyzer(unittest.TestCase):
    def test_online_complete_chapter08(self):
        traces = [r for c in ("approve", "reject", "edit", "isolation-A", "isolation-B") for r in fixture(c, thread=c)]
        result = reporter.summarize("08", "zr-langChain", traces, expect_online=True, expect_all=True, scan_limited=False)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["trace_count"], 10)
        self.assertEqual(result["llm_calls"], 5)
        self.assertEqual(result["token_totals_known"]["total"], 600)
        self.assertEqual(result["token_totals_known"]["reasoning"], 60)
        self.assertTrue(all(x["thread_pairing"] == "verified" for x in result["hitl_pairs"]))

    def test_cannot_verify_thread_id(self):
        rows = fixture("approve")
        rows[1]["thread_hash"] = None
        result = reporter.summarize("08", "demo", rows, expect_online=True, expect_all=False, scan_limited=False)
        self.assertEqual(result["status"], "review")
        self.assertEqual(result["hitl_pairs"][0]["thread_pairing"], "unverified")

    def test_reject_duplicate_llm_on_resume(self):
        rows = fixture("approve")
        rows[1]["llm_calls"] = 1
        result = reporter.summarize("08", "demo", rows, expect_online=True, expect_all=False, scan_limited=False)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("LLM 调用" in issue for issue in result["failures"]))

    def test_missing_case_detected(self):
        rows = fixture("approve")
        result = reporter.summarize("08", "demo", rows, expect_online=True, expect_all=True, scan_limited=False)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("缺少预期场景" in issue for issue in result["failures"]))

    def test_only_llm_is_counted(self):
        root = fake_run("chapter07-account", outputs={"status": "triaged"}, tokens={"total_tokens": 10000})
        child = fake_run("RunnableSequence", trace=root["id"], tokens={"total_tokens": 10000})
        llm = fake_run("ChatDeepSeek", "llm", trace=root["id"], tokens={"total_tokens": 240, "prompt_tokens": 100, "completion_tokens": 140})
        summary = reporter.summarize_trace(root, [root, child, llm], "07")
        self.assertEqual(summary["run_count"], 3)
        self.assertEqual(summary["llm_calls"], 1)
        self.assertEqual(summary["tokens"]["total"], 240)

    def test_partial_tokens_are_not_false_zeros(self):
        root = fake_run("chapter07-account")
        llm = fake_run("ChatDeepSeek", "llm", trace=root["id"])
        summary = reporter.summarize_trace(root, [llm], "07")
        result = reporter.summarize("07", "demo", [summary], expect_online=False, expect_all=False, scan_limited=False)
        self.assertTrue(result["tokens_partial"])
        self.assertEqual(result["status"], "review")

    def test_empty_result_is_fail(self):
        result = reporter.summarize("07", "demo", [], expect_online=False, expect_all=False, scan_limited=False)
        self.assertEqual(result["status"], "fail")

    def test_no_private_text_in_report(self):
        rows = fixture("approve")
        r = reporter.summarize("08", "demo", rows, expect_online=True, expect_all=False, scan_limited=False)
        md = reporter.markdown(r)
        self.assertNotIn("a-unique-thread", md)
        self.assertIn("章", md)


class TestBatchFiltering(unittest.TestCase):
    def test_batch_id_and_thread_fingerprint_are_extracted(self):
        root = fake_run("chapter08-approve-pause", meta={"batch_id": "chapter08-test", "thread_fingerprint": "aabbccddeeff"})
        record = reporter.summarize_trace(root, [], "08")
        self.assertEqual(record["batch_id"], "chapter08-test")
        self.assertEqual(record["thread_hash"], "aabbccddeeff")

    def test_remote_batch_filter_ignores_other_batches(self):
        good = fake_run("chapter08-approve-pause", meta={"batch_id": "chapter08-yes"})
        wrong = fake_run("chapter08-edit-pause", meta={"batch_id": "chapter08-no"})
        client = NewAPIClient([good, wrong], {good['id']: [good], wrong['id']: [wrong]})
        rows, limited = reporter.load_remote(client, project="demo", chapter="08",
            since=datetime(2026, 10, 23, tzinfo=timezone.utc), max_roots=10, max_spans=10,
            batch_id="chapter08-yes")
        self.assertEqual(len(rows), 1)
        self.assertFalse(limited)
        self.assertEqual(rows[0]["batch_id"], "chapter08-yes")


class TestInterruptEvidence(unittest.TestCase):
    def rows(self, error="GraphInterrupt((Interrupt(value={'private': 'SECRET'}, id='x'),))", node="human_review"):
        rows = fixture()
        root = fake_run("chapter08-approve-pause", outputs={"status": "awaiting_review", "steps": ["prepare_draft"]}, meta={"thread_id": "a-unique-thread", "batch_id": "chapter08-test"})
        child = fake_run(node, trace=root["id"], parent=root["id"], error=error)
        llm = fake_run("model", "llm", trace=root["id"], tokens={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "completion_token_details": {"reasoning": 0}})
        rows[0] = reporter.summarize_trace(root, [child, llm], "08")
        return rows

    def report(self, rows):
        return reporter.summarize("08", "demo", rows, expect_online=True, expect_all=False, scan_limited=False)

    def test_confirmed_interrupt_is_exempt_in_trace_and_pair(self):
        rows = self.rows()
        r = self.report(rows)
        self.assertEqual(r["status"], "pass")
        pause = next(x for x in r["root_runs"] if x["phase"] == "pause")
        self.assertEqual((pause["raw_errors"], pause["expected_interrupts"], pause["errors"]), (1, 1, 0))
        self.assertIn("expected_interrupt_verified", r["hitl_pairs"][0]["checks"])
        self.assertNotIn("SECRET", str(r))
        self.assertEqual(rows[0]["errors"], 1)  # reporting must not mutate evidence

    def test_missing_or_invalid_resume_never_exempts(self):
        for change in ("thread_hash", "batch_id", "status", "steps", "llm_calls", "errors", "truncated", "completed", "start_time"):
            with self.subTest(change=change):
                rows = self.rows()
                rows[1][change] = {"thread_hash": None, "batch_id": "other", "status": "bad", "steps": [], "llm_calls": 1, "errors": 1, "truncated": True, "completed": False, "start_time": rows[0]["start_time"]}[change]
                if change == "errors":
                    rows[1]["raw_errors"] = 1
                r = self.report(rows)
                self.assertNotEqual(r["status"], "pass")
                self.assertEqual(next(x for x in r["root_runs"] if x["phase"] == "pause")["errors"], 1)
        self.assertEqual(self.report(self.rows()[:1])["status"], "fail")

    def test_wrong_exception_or_node_never_exempted(self):
        for error, node in (("RuntimeError('GraphInterrupt')", "human_review"), ("GraphInterrupt(())", "human_review"), ("GraphInterrupt((Interrupt(value=1),))", "prepare_draft")):
            with self.subTest(error=error, node=node):
                self.assertEqual(self.report(self.rows(error, node))["status"], "fail")

    def test_other_pause_error_and_bad_state_are_not_hidden(self):
        for change in ("errors", "status", "truncated", "completed"):
            rows = self.rows()
            rows[0][change] = {"errors": 2, "status": "failed", "truncated": True, "completed": False}[change]
            if change == "errors":
                rows[0]["raw_errors"] = 2
            self.assertNotEqual(self.report(rows)["status"], "pass")

    def test_missing_reasoning_remains_review_after_interrupt_fix(self):
        rows = self.rows()
        rows[0]["token_missing"]["reasoning"] = 1
        r = self.report(rows)
        self.assertEqual(r["status"], "review")
        self.assertEqual(len(r["warnings"]), 1)
        self.assertIn("reasoning", r["warnings"][0])

    def test_reanalysis_cannot_reuse_previous_exemption(self):
        first = self.report(self.rows())
        rows = first["root_runs"]
        next(x for x in rows if x["phase"] == "resume")["thread_hash"] = None
        self.assertEqual(self.report(rows)["status"], "fail")


class TestReadBounds(unittest.TestCase):
    def test_exact_cap_is_complete_over_cap_is_limited(self):
        self.assertEqual(reporter.collect_bounded(iter(range(200)), 200), (list(range(200)), False))
        self.assertEqual(reporter.collect_bounded(iter(range(201)), 200), (list(range(200)), True))

    def test_span_overflow_refuses_partial_report(self):
        root = fake_run("chapter08-approve-pause")
        children = [fake_run("node", trace=root['id'], parent=root['id']) for _ in range(200)]
        client = NewAPIClient([root], {root['id']: [root, *children]})
        with self.assertRaises(reporter.TraceReadLimitExceeded):
            reporter.load_remote(client, project="demo", chapter="08", since=datetime.now(timezone.utc), max_roots=100, max_spans=200)



class TestAcceptanceBoundaries(unittest.TestCase):
    def report(self, rows, expect_all=False):
        return reporter.summarize("08", "demo", rows, expect_online=True, expect_all=expect_all, scan_limited=False)

    def test_all_interrupt_paths_reject_wrong_batch_or_time(self):
        for with_error in (False, True):
            for key, val in (("batch_id", "other"), ("start_time", "2020-01-01T00:00:00+00:00")):
                with self.subTest(with_error=with_error, key=key):
                    rows = TestInterruptEvidence().rows() if with_error else fixture()
                    rows[1][key] = val
                    r = self.report(rows)
                    self.assertEqual(r["status"], "fail")
                    self.assertEqual(r["root_runs"][0]["expected_interrupts"], 0)

    def test_missing_batch_or_time_is_not_verified(self):
        for key in ("batch_id", "start_time", "end_time"):
            with self.subTest(key=key):
                rows = fixture()
                rows[0][key] = rows[1][key] = None
                self.assertEqual(self.report(rows)["status"], "review")

    def test_interrupt_error_without_batch_is_not_exempt(self):
        rows = TestInterruptEvidence().rows()
        rows[0]["batch_id"] = rows[1]["batch_id"] = None
        r = self.report(rows)
        self.assertNotEqual(r["status"], "pass")
        self.assertEqual(next(x for x in r["root_runs"] if x["phase"] == "pause")["expected_interrupts"], 0)

    def test_resume_cannot_start_before_pause_finishes(self):
        rows = fixture()
        rows[0]["end_time"] = "2026-10-23T00:00:03+00:00"
        self.assertEqual(self.report(rows)["status"], "fail")

    def test_isolation_cases_must_use_different_threads(self):
        rows = [r for c in ("approve", "reject", "edit", "isolation-A", "isolation-B") for r in fixture(c, thread="shared")]
        r = self.report(rows, expect_all=True)
        self.assertEqual(r["status"], "fail")
        self.assertTrue(any("isolation" in issue for issue in r["failures"]))

    def test_isolation_thread_cannot_be_missing(self):
        rows = fixture("isolation-A", thread="a") + fixture("isolation-B", thread="b")
        rows[0]["thread_hash"] = rows[1]["thread_hash"] = None
        self.assertEqual(self.report(rows)["status"], "review")

    def test_unfinished_child_is_recorded_and_blocks_pass(self):
        rows = fixture()
        root = fake_run("chapter08-approve-resume", outputs={"status": "approved", "review_action": "approve", "steps": ["prepare_draft", "human_review", "approve", "finalize"]}, meta={"thread_id": "a-unique-thread", "batch_id": "chapter08-test"})
        root["start_time"] += timedelta(seconds=2)
        root["end_time"] += timedelta(seconds=2)
        children = [fake_run(n, trace=root["id"], parent=root["id"]) for n in ("human_review", "approve", "finalize")]
        children[-1]["end_time"] = None
        rows[1] = reporter.summarize_trace(root, children, "08")
        r = self.report(rows)
        self.assertEqual(r["status"], "review")
        self.assertEqual(rows[1]["unfinished_runs"], 1)

    def test_unfinished_child_blocks_interrupt_exemption(self):
        rows = TestInterruptEvidence().rows()
        rows[1]["unfinished_runs"] = 1
        r = self.report(rows)
        self.assertNotEqual(r["status"], "pass")
        self.assertEqual(next(x for x in r["root_runs"] if x["phase"] == "pause")["expected_interrupts"], 0)

    def test_unfinished_non_hitl_trace_is_review(self):
        root = fake_run("chapter07-account")
        child = fake_run("worker", parent=root["id"], trace=root["id"])
        child["end_time"] = None
        row = reporter.summarize_trace(root, [child], "07")
        r = reporter.summarize("07", "demo", [row], expect_online=False, expect_all=False, scan_limited=False)
        self.assertEqual(r["status"], "review")


if __name__ == "__main__":
    unittest.main()
