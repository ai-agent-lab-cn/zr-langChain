"""以固定脱敏输入测试 Trace 证据边界，不调用任何远程服务。"""

import copy
import unittest
import sys
from pathlib import Path

CHAPTER = Path(__file__).resolve().parents[1] / '11-agent-reliability'
sys.path.insert(0, str(CHAPTER))

from trace_audit import audit_trace_fixture, complete_fixture


def sample():
    """独立构造可手工核对的三条输入，避免依赖生产夹具生成期望值。"""
    shared = {"batch_id": "fixture-batch", "thread_id": "fixture-thread",
              "status": "SUCCESS", "end_time": 1}
    return [
        dict(shared, id="root", parent_id=None, run_type="chain", total_tokens=999),
        dict(shared, id="model-1", parent_id="root", run_type="llm", total_tokens=12),
        dict(shared, id="tool-1", parent_id="root", run_type="tool"),
    ]


class TraceAuditTests(unittest.TestCase):
    """直接断言审计结果、未知用量和去重行为。"""

    def audit(self, records, **kwargs):
        """给所有用例绑定脱敏批次，保留可覆盖的完整性参数。"""
        return audit_trace_fixture(records, batch_id="fixture-batch", **kwargs)

    def test_complete_counts_only_llm_usage(self):
        """根聚合值不得再次加入模型用量。"""
        self.assertEqual(self.audit(sample()), {
            "status": "PASS", "token_usage": 12, "missing_usage": 0,
            "issues": [], "source": "local_fixture",
        })

    def test_provided_fixture_is_complete_and_returns_fresh_records(self):
        """公开夹具必须可直接审计，调用者改动不得污染下一次实验。"""
        first = complete_fixture()
        self.assertEqual(self.audit(first)["status"], "PASS")
        self.assertEqual(first, sample())
        first[1]["total_tokens"] = 800
        self.assertEqual(self.audit(complete_fixture())["token_usage"], 12)

    def test_identical_duplicate_does_not_inflate_tokens(self):
        """相同 Run 重复翻页出现时只计一次。"""
        records = sample()
        records.append(copy.deepcopy(records[1]))
        result = self.audit(records)
        self.assertEqual((result["status"], result["token_usage"]), ("PASS", 12))

    def test_missing_llm_usage_remains_unknown_even_on_failure(self):
        """失败调用缺 usage 不能被当作零费用。"""
        for state in ("SUCCESS", "FAILED"):
            with self.subTest(state=state):
                records = sample()
                records[1].pop("total_tokens")
                records[1]["status"] = state
                result = self.audit(records)
                self.assertEqual(result["status"], "REVIEW")
                self.assertIsNone(result["token_usage"])
                self.assertEqual(result["missing_usage"], 1)

    def test_one_unknown_llm_invalidates_total_not_known_observation(self):
        """已有一条用量也不能掩盖另一条未知调用。"""
        records = sample()
        records.append(dict(records[1], id="model-2", total_tokens=None))
        result = self.audit(records)
        self.assertIsNone(result["token_usage"])
        self.assertEqual(result["missing_usage"], 1)
        self.assertEqual(result["status"], "REVIEW")

    def test_missing_structure_is_review_with_known_sample_usage(self):
        """结构不完整时保留已知模型样本用量，不能宣布通过。"""
        variants = []
        for field in ("parent_id", "thread_id", "batch_id", "status", "end_time"):
            records = sample()
            records[2].pop(field)
            variants.append((field, records, {}))
        records = sample()
        records[2]["parent_id"] = "missing-parent"
        variants.append(("missing parent", records, {}))
        records = sample()
        records[0]["parent_id"] = "missing-root"
        variants.append(("missing root", records, {}))
        records = sample()
        records[2]["end_time"] = None
        variants.append(("null end", records, {}))
        for name, records, kwargs in variants:
            with self.subTest(name=name):
                result = self.audit(records, **kwargs)
                self.assertEqual(result["status"], "REVIEW")
                self.assertEqual(result["token_usage"], 12)
                self.assertTrue(result["issues"])

    def test_truncated_pages_cannot_claim_complete_token_total(self):
        """截断的分页可能漏掉整个模型调用，已见用量不是完整总量。"""
        result = self.audit(sample(), complete=False)
        self.assertEqual(result["status"], "REVIEW")
        self.assertIsNone(result["token_usage"])
        self.assertEqual(result["missing_usage"], 0)

    def test_unknown_run_type_cannot_hide_a_model_call_as_zero_usage(self):
        """未知类型可能源于模型类型拼写错误，不得按普通非模型忽略。"""
        records = sample()
        records[1]["run_type"] = "bogus"
        result = self.audit(records)
        self.assertEqual(result["status"], "FAIL")
        self.assertIsNone(result["token_usage"])
        self.assertTrue(result["issues"])

    def test_missing_identity_or_run_type_cannot_hide_usage(self):
        """无法辨认或去重的记录不能生成看似完整的 Token 总量。"""
        for field in ("id", "run_type"):
            with self.subTest(field=field):
                records = sample()
                records[1].pop(field)
                result = self.audit(records)
                self.assertEqual(result["status"], "REVIEW")
                self.assertIsNone(result["token_usage"])

    def test_conflicting_id_batch_status_thread_and_cycle_fail(self):
        """明确相互矛盾的证据应 FAIL，不能按缺证据降为 REVIEW。"""
        variants = []
        records = sample()
        records.append(dict(records[1], total_tokens=13))
        variants.append(("duplicate conflict", records))
        for field, value in (("batch_id", "other-batch"),
                             ("status", "MADE_UP"), ("thread_id", "other-thread")):
            records = sample()
            records[1][field] = value
            variants.append((field, records))
        records = sample()
        records[1]["parent_id"] = "tool-1"
        records[2]["parent_id"] = "model-1"
        variants.append(("cycle beside root", records))
        records = sample()
        records[1]["parent_id"] = "model-1"
        variants.append(("self cycle", records))
        for name, records in variants:
            with self.subTest(name=name):
                result = self.audit(records)
                self.assertEqual(result["status"], "FAIL")
                self.assertTrue(result["issues"])

    def test_negative_fractional_boolean_or_string_tokens_fail(self):
        """Python bool 虽是 int 子类，也不能充当 Token 计数。"""
        for value in (-1, 1.5, True, False, "12", [], {}):
            with self.subTest(value=value):
                records = sample()
                records[1]["total_tokens"] = value
                result = self.audit(records)
                self.assertEqual(result["status"], "FAIL")
                self.assertIsNone(result["token_usage"])

    def test_zero_usage_and_multiple_unique_llms_are_valid(self):
        """真实记录的零值可统计，多条唯一模型记录可相加。"""
        records = sample()
        records.append(dict(records[1], id="model-2", total_tokens=0))
        records.append(dict(records[1], id="model-3", total_tokens=8))
        result = self.audit(records)
        self.assertEqual((result["status"], result["token_usage"]), ("PASS", 20))

    def test_invalid_aggregate_usage_is_not_counted_or_silently_accepted(self):
        """根用量不参与求和，但其非法值仍属于相互矛盾的证据。"""
        for value in (-1, True, "999"):
            with self.subTest(value=value):
                records = sample()
                records[0]["total_tokens"] = value
                result = self.audit(records)
                self.assertEqual(result["status"], "FAIL")
                self.assertEqual(result["token_usage"], 12)

    def test_equal_boolean_duplicate_cannot_bypass_validation(self):
        """True 与 1 在 Python 相等，重复记录也必须先通过字段检查。"""
        records = sample()
        records.append(dict(records[2], end_time=True))
        self.assertEqual(self.audit(records)["status"], "FAIL")

    def test_large_integer_time_and_deep_parent_chain_do_not_crash(self):
        """本地有限整数无需浮点转换，深链检查不依赖递归深度。"""
        records = sample()
        records[0]["end_time"] = 10 ** 1000
        previous = "root"
        for index in range(1500):
            identity = f"deep-{index}"
            records.append(dict(records[2], id=identity, parent_id=previous))
            previous = identity
        self.assertEqual(self.audit(records)["status"], "PASS")

    def test_pause_and_business_terminal_states_are_evidence_not_success(self):
        """证据 PASS 不代表业务成功，合法暂停也必须有结束标记。"""
        for state in ("WAITING_APPROVAL", "FAILED", "DEGRADED", "REJECTED"):
            with self.subTest(state=state):
                records = sample()
                records[0]["status"] = state
                self.assertEqual(self.audit(records)["status"], "PASS")
        records[0]["status"] = "WAITING_APPROVAL"
        records[0]["end_time"] = None
        self.assertEqual(self.audit(records)["status"], "REVIEW")

    def test_empty_is_review_with_unknown_usage(self):
        """没有样本不是零用量且不能通过。"""
        result = self.audit([])
        self.assertEqual(result["status"], "REVIEW")
        self.assertIsNone(result["token_usage"])

    def test_malformed_inputs_fail_without_crashing(self):
        """错误容器、错误字段类型和伪完整标记不能绕过审计。"""
        for records in (None, {}, "runs", 1, (sample()[0],), [None], [1], [[]]):
            with self.subTest(records=records):
                self.assertEqual(self.audit(records)["status"], "FAIL")
        for field, value in (("id", []), ("parent_id", {}),
                             ("thread_id", []), ("batch_id", {}),
                             ("run_type", []), ("status", []), ("end_time", []),
                             ("end_time", True), ("end_time", float("nan"))):
            with self.subTest(field=field, value=value):
                records = sample()
                records[1][field] = value
                self.assertEqual(self.audit(records)["status"], "FAIL")
        for value in ("false", 0, None):
            with self.subTest(complete=value):
                self.assertEqual(self.audit(sample(), complete=value)["status"], "FAIL")
        for value in (None, [], ""):
            with self.subTest(batch=value):
                self.assertEqual(audit_trace_fixture(sample(), batch_id=value)["status"], "FAIL")

    def test_audit_does_not_mutate_input(self):
        """审计不得删除调用者原始证据或补写缺失字段。"""
        records = sample()
        before = copy.deepcopy(records)
        self.audit(records)
        self.assertEqual(records, before)


if __name__ == "__main__":
    unittest.main()
