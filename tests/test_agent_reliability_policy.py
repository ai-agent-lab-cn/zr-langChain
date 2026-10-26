"""使用真实策略与本地故障序列核验有限重试和断路器。"""

import math
import threading
import unittest
import sys
from pathlib import Path

CHAPTER = Path(__file__).resolve().parents[1] / '11-agent-reliability'
sys.path.insert(0, str(CHAPTER))
from concurrent.futures import ThreadPoolExecutor

from retry_policy import (
    CircuitBreaker, CircuitOpen, RetryBudget, ServiceFailure,
    VirtualClock, classify, run_with_retry,
)


class ScriptedOperation:
    """只替换外部响应，保留实际请求次数用于独立核对。"""

    def __init__(self, *responses):
        """按顺序保存本地响应，不进行真实请求。"""
        self.responses = iter(responses)
        self.calls = 0

    def __call__(self):
        """消费一次响应；异常项模拟下游失败。"""
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


class RetryPolicyTests(unittest.TestCase):
    """验证重试分类、预算、脱敏和虚拟时间边界。"""

    def test_503_sequence_retries_twice_and_succeeds(self):
        """少重试或错误退避都应使固定夹具失败。"""
        clock = VirtualClock()
        operation = ScriptedOperation(ServiceFailure(503), ServiceFailure(503), "ok")
        report = run_with_retry(operation, clock=clock)
        self.assertEqual(report["status"], "SUCCESS")
        self.assertEqual(report["result"], "ok")
        self.assertEqual(report["requests"], 3)
        self.assertEqual(operation.calls, 3)
        self.assertEqual(report["waits"], [1, 2])
        self.assertEqual(report["elapsed"], 3)

    def test_deadline_refuses_next_wait_before_another_request(self):
        """剩余预算不足时既不等待也不启动下一请求。"""
        operation = ScriptedOperation(*[ServiceFailure(503) for _ in range(3)])
        report = run_with_retry(operation, clock=VirtualClock(), budget=RetryBudget(deadline=2))
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["reason"], "DEADLINE_EXCEEDED")
        self.assertEqual(operation.calls, 2)
        self.assertEqual(report["requests"], 2)
        self.assertEqual(report["waits"], [1])
        self.assertEqual(report["elapsed"], 1)

    def test_wait_reaching_deadline_does_not_start_request(self):
        """等待恰好耗尽预算也不应先睡再检查。"""
        operation = ScriptedOperation(ServiceFailure(503), "too late")
        report = run_with_retry(operation, clock=VirtualClock(), budget=RetryBudget(deadline=1))
        self.assertEqual(operation.calls, 1)
        self.assertEqual(report["waits"], [])
        self.assertEqual(report["reason"], "DEADLINE_EXCEEDED")

    def test_slow_success_does_not_claim_deadline_success(self):
        """同步调用只能在返回后发现超时，不能把晚到结果算成功。"""
        clock = VirtualClock()

        def slow_operation():
            """模拟下游耗时而非退避等待。"""
            clock.advance(3)
            return "late"

        report = run_with_retry(slow_operation, clock=clock, budget=RetryBudget(deadline=2))
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["reason"], "DEADLINE_EXCEEDED")
        self.assertEqual(report["requests"], 1)
        self.assertEqual(report["elapsed"], 3)
        self.assertNotIn("result", report)
        self.assertEqual(clock.waits, [])

    def test_deadline_is_relative_to_each_run(self):
        """复用时钟不会把过去时间计入新一轮预算。"""
        clock = VirtualClock()
        clock.advance(100)
        clock.sleep(4)
        operation = ScriptedOperation(ServiceFailure(503), "ok")
        report = run_with_retry(operation, clock=clock, budget=RetryBudget(deadline=2))
        self.assertEqual(report["status"], "SUCCESS")
        self.assertEqual(report["elapsed"], 1)
        self.assertEqual(report["waits"], [1])

    def test_transient_errors_retry_but_explicit_quota_stops(self):
        """普通429和明确配额耗尽不得被合并为同一语义。"""
        for failure in (ServiceFailure(429), ServiceFailure(500), TimeoutError("secret"), ConnectionError("secret")):
            with self.subTest(error=type(failure).__name__, status=getattr(failure, "status_code", None)):
                operation = ScriptedOperation(failure, "ok")
                report = run_with_retry(operation, clock=VirtualClock())
                self.assertEqual(report["status"], "SUCCESS")
                self.assertEqual(operation.calls, 2)
        operation = ScriptedOperation(ServiceFailure(429, "daily_quota"), "must not run")
        report = run_with_retry(operation, clock=VirtualClock())
        self.assertEqual((report["status"], report["reason"]), ("FAILED", "QUOTA"))
        self.assertEqual(operation.calls, 1)
        self.assertEqual(report["waits"], [])

    def test_nonretryable_service_failures_preserve_safe_reason(self):
        """余额、鉴权、参数及未知服务故障只执行一次。"""
        for status, expected in ((402, "BALANCE"), (401, "AUTH"), (400, "INVALID"), (422, "INVALID"), (418, "UNKNOWN"), (None, "UNKNOWN")):
            with self.subTest(status=status):
                operation = ScriptedOperation(ServiceFailure(status, "FAKE_SECRET_CH11"), "must not run")
                report = run_with_retry(operation, clock=VirtualClock())
                self.assertEqual(report["reason"], expected)
                self.assertEqual(report["status"], "FAILED")
                self.assertEqual(operation.calls, 1)
                self.assertEqual(report["waits"], [])
                self.assertNotIn("FAKE_SECRET_CH11", repr(report))

    def test_programming_errors_propagate_without_retry(self):
        """把程序缺陷包装成业务失败会使本测试失效。"""
        for failure in (ValueError("defect"), TypeError("defect"), RuntimeError("defect")):
            with self.subTest(error=type(failure).__name__):
                operation = ScriptedOperation(failure)
                with self.assertRaises(type(failure)) as caught:
                    run_with_retry(operation, clock=VirtualClock())
                self.assertIs(caught.exception, failure)
                self.assertEqual(operation.calls, 1)

    def test_attempt_limit_and_delay_cap(self):
        """持续故障必须按调用上限停止，指数退避不能超过上限。"""
        operation = ScriptedOperation(*[ServiceFailure(503) for _ in range(6)])
        report = run_with_retry(operation, clock=VirtualClock(), budget=RetryBudget(max_attempts=5, deadline=30))
        self.assertEqual(report["reason"], "MAX_ATTEMPTS")
        self.assertEqual(report["requests"], 5)
        self.assertEqual(operation.calls, 5)
        self.assertEqual(report["waits"], [1, 2, 4, 4])

    def test_full_jitter_is_seeded_and_bounded(self):
        """随机等待可复现且位于每次封顶退避区间。"""
        observations = []
        for seed in (11, 11, 12):
            operation = ScriptedOperation(*[ServiceFailure(503) for _ in range(5)])
            report = run_with_retry(operation, clock=VirtualClock(), budget=RetryBudget(max_attempts=5, deadline=30, jitter=True, seed=seed))
            observations.append(report["waits"])
            for wait, limit in zip(report["waits"], (1, 2, 4, 4)):
                self.assertGreaterEqual(wait, 0)
                self.assertLessEqual(wait, limit)
            self.assertEqual(report["requests"], 5)
        self.assertEqual(observations[0], observations[1])
        self.assertNotEqual(observations[0], observations[2])

    def test_budget_rejects_invalid_or_nonfinite_values(self):
        """无穷等待、非整数尝试次数和歧义布尔值必须在执行前拒绝。"""
        cases = (
            {"max_attempts": 0}, {"max_attempts": 1.5}, {"max_attempts": True},
            {"initial_delay": -1}, {"initial_delay": math.nan}, {"initial_delay": True},
            {"backoff_factor": 0.5}, {"backoff_factor": math.inf},
            {"max_delay": -1}, {"max_delay": 0.5}, {"max_delay": math.inf},
            {"deadline": 0}, {"deadline": math.inf}, {"deadline": math.nan},
            {"jitter": "yes"}, {"seed": 1.5}, {"seed": False},
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    RetryBudget(**arguments)

    def test_virtual_clock_rejects_invalid_time_and_records_only_sleep(self):
        """操作耗时与等待分开记录，非法时间不能倒退时钟。"""
        clock = VirtualClock()
        clock.advance(2)
        clock.sleep(1)
        self.assertEqual(clock.now, 3)
        self.assertEqual(clock.waits, [1])
        for invalid in (-1, math.nan, math.inf, True):
            for method in (clock.sleep, clock.advance):
                with self.subTest(value=invalid, method=method.__name__):
                    with self.assertRaises(ValueError):
                        method(invalid)
        self.assertEqual(clock.now, 3)

    def test_exception_and_classification_do_not_echo_untrusted_text(self):
        """异常字符串及表示形式都不能保留服务端原始敏感文本。"""
        failure = ServiceFailure(503, "FAKE_SECRET_CH11")
        self.assertNotIn("FAKE_SECRET_CH11", str(failure))
        self.assertNotIn("FAKE_SECRET_CH11", repr(failure))
        self.assertNotIn("FAKE_SECRET_CH11", repr(vars(failure)))
        self.assertEqual(classify(failure), "TRANSIENT")
        self.assertEqual(classify(ServiceFailure(429, "daily_quota")), "QUOTA")
        self.assertEqual(classify(KeyError("secret")), "UNKNOWN")


class CircuitBreakerTests(unittest.TestCase):
    """验证实际下游计数及并发半开探测。"""

    def test_opens_after_three_availability_failures_and_short_circuits(self):
        """打开后再次调用不会再进入下游。"""
        clock = VirtualClock()
        breaker = CircuitBreaker(clock)
        operation = ScriptedOperation(*[ServiceFailure(503) for _ in range(3)])
        report = run_with_retry(operation, clock=clock, breaker=breaker)
        self.assertEqual(report["requests"], 3)
        self.assertEqual(breaker.state, "OPEN")
        blocked = ScriptedOperation("must not run")
        report = run_with_retry(blocked, clock=clock, breaker=breaker)
        self.assertEqual(report["reason"], "CIRCUIT_OPEN")
        self.assertEqual(report["requests"], 0)
        self.assertEqual(blocked.calls, 0)

    def test_account_and_parameter_errors_do_not_open_breaker(self):
        """共享可用性计数不受某一账户或请求配置问题污染。"""
        clock = VirtualClock()
        breaker = CircuitBreaker(clock, threshold=1)
        for failure in (ServiceFailure(402), ServiceFailure(401), ServiceFailure(422), ServiceFailure(429, "daily_quota")):
            with self.subTest(status=failure.status_code):
                with self.assertRaises(ServiceFailure):
                    breaker.call(ScriptedOperation(failure))
                self.assertEqual(breaker.state, "CLOSED")

    def test_success_resets_failures_and_half_open_probe_success_closes(self):
        """成功会重新建立连续失败计数，冷却后的成功探测关闭断路器。"""
        clock = VirtualClock()
        breaker = CircuitBreaker(clock, threshold=2)
        for _ in range(2):
            with self.assertRaises(ServiceFailure):
                breaker.call(ScriptedOperation(ServiceFailure(503)))
            self.assertEqual(breaker.call(ScriptedOperation("ok")), "ok")
        self.assertEqual(breaker.state, "CLOSED")
        for _ in range(2):
            with self.assertRaises(ServiceFailure):
                breaker.call(ScriptedOperation(ServiceFailure(503)))
        self.assertEqual(breaker.state, "OPEN")
        clock.advance(5)
        self.assertEqual(breaker.call(ScriptedOperation("recovered")), "recovered")
        self.assertEqual(breaker.state, "CLOSED")
        with self.assertRaises(ServiceFailure):
            breaker.call(ScriptedOperation(ServiceFailure(503)))
        self.assertEqual(breaker.state, "CLOSED")

    def test_failed_half_open_probe_restarts_cooldown(self):
        """探测再次遇到可用性故障时从新失败时刻重新冷却。"""
        clock = VirtualClock()
        breaker = CircuitBreaker(clock, threshold=1)
        with self.assertRaises(ServiceFailure):
            breaker.call(ScriptedOperation(ServiceFailure(503)))
        clock.advance(5)
        with self.assertRaises(TimeoutError):
            breaker.call(ScriptedOperation(TimeoutError()))
        clock.advance(4)
        blocked = ScriptedOperation("early")
        with self.assertRaises(CircuitOpen):
            breaker.call(blocked)
        self.assertEqual(blocked.calls, 0)
        clock.advance(1)
        self.assertEqual(breaker.call(ScriptedOperation("ok")), "ok")

    def test_half_open_allows_only_one_probe_under_contention(self):
        """线程屏障让请求同时竞争，探测阻塞期间其余请求必须快速拒绝。"""
        clock = VirtualClock()
        breaker = CircuitBreaker(clock, threshold=1)
        with self.assertRaises(ServiceFailure):
            breaker.call(ScriptedOperation(ServiceFailure(503)))
        clock.advance(5)
        barrier = threading.Barrier(8)
        release_probe = threading.Event()
        probe_started = threading.Event()
        count_lock = threading.Lock()
        counts = {"downstream": 0, "blocked": 0}
        all_rejected = threading.Event()

        def operation():
            """唯一获准探测保持运行，让其余线程能观察半开状态。"""
            with count_lock:
                counts["downstream"] += 1
            probe_started.set()
            if not release_probe.wait(3):
                raise AssertionError("测试未释放探测线程")
            return "ok"

        def contender():
            """同步进入断路器并统计被拒绝的竞争者。"""
            barrier.wait(timeout=3)
            try:
                return breaker.call(operation)
            except CircuitOpen:
                with count_lock:
                    counts["blocked"] += 1
                    if counts["blocked"] == 7:
                        all_rejected.set()
                return "blocked"

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(contender) for _ in range(8)]
            try:
                self.assertTrue(probe_started.wait(2))
                self.assertTrue(all_rejected.wait(2))
                self.assertEqual(breaker.state, "HALF_OPEN")
            finally:
                release_probe.set()
            outcomes = [future.result(timeout=3) for future in futures]
        self.assertEqual(counts, {"downstream": 1, "blocked": 7})
        self.assertEqual(outcomes.count("ok"), 1)
        self.assertEqual(breaker.state, "CLOSED")

    def test_breaker_configuration_rejects_invalid_limits(self):
        """阈值和冷却时间必须是有限且有效的预算。"""
        for arguments in ({"threshold": 0}, {"threshold": True}, {"threshold": 1.5}, {"cooldown": -1}, {"cooldown": math.inf}, {"cooldown": math.nan}):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    CircuitBreaker(VirtualClock(), **arguments)


if __name__ == "__main__":
    unittest.main()
