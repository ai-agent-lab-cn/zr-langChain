"""第11章离线重试策略：虚拟时间、有限退避与单探测断路器。

仅替换外部响应，不模拟线上可靠性；同步操作无法被虚拟时钟抢占，
因此操作超时在返回后识别，晚到成功会被记为失败而非按时成功。


配套流程图：同目录 basic-knowledge.md，有限重试与断路器（retry / breaker）。
"""

import math
import random
import threading
from dataclasses import dataclass
from typing import Any, Callable


def _finite_number(value: Any, name: str, *, minimum: float = 0) -> None:
    """拒绝布尔值、非数值、无穷和越界值，避免不受控等待。"""
    try:
        valid = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value >= minimum
        )
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} 必须是大于等于 {minimum} 的有限数值")


def _positive_integer(value: Any, name: str) -> None:
    """确保次数预算是正整数，防止把布尔值当一次调用。"""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须是正整数")


class ServiceFailure(Exception):
    """保留安全状态码与白名单原因，丢弃不可信服务端原文。"""

    def __init__(self, status_code: int | None, reason: str = ""):
        """只有明确 daily_quota 标记进入分类，任意原文均不保存。"""
        if status_code is not None and (isinstance(status_code, bool) or not isinstance(status_code, int)):
            raise ValueError("status_code 必须是整数或 None")
        self.status_code = status_code
        self.reason = "daily_quota" if reason == "daily_quota" else ""
        super().__init__(f"下游服务失败：status_code={status_code}，reason={self.reason or 'unspecified'}")


class CircuitOpen(Exception):
    """表示断路器拒绝调用；不包含下游异常或请求原文。"""


class VirtualClock:
    """用有限数值表示虚拟时间，sleep 只推进时间而不真实阻塞。"""

    def __init__(self):
        """从零创建时钟，并单独记录退避等待。"""
        self._now = 0.0
        self.waits: list[float] = []
        self._lock = threading.Lock()

    @property
    def now(self) -> float:
        """读取线程安全的当前虚拟时间。"""
        with self._lock:
            return self._now

    def sleep(self, seconds: float) -> None:
        """推进并记录等待；拒绝非法时间且不产生真实延迟。"""
        self._advance(seconds, record_wait=True)

    def advance(self, seconds: float) -> None:
        """推进操作耗时，不把它混入退避等待列表。"""
        self._advance(seconds, record_wait=False)

    def _advance(self, seconds: float, *, record_wait: bool) -> None:
        """原子推进时间，防止有效单次时长累加成无穷。"""
        _finite_number(seconds, "seconds")
        with self._lock:
            next_time = self._now + seconds
            if not math.isfinite(next_time):
                raise ValueError("虚拟时间累计值必须有限")
            self._now = next_time
            if record_wait:
                self.waits.append(seconds)


@dataclass(frozen=True)
class RetryBudget:
    """单次逻辑调用的不可变预算；所有时间单位都是虚拟秒。

    max_attempts 含首次请求；initial_delay/backoff_factor/max_delay 定义封顶退避；
    deadline 包含操作与等待，按本次开始时间计算，不是时钟的绝对截止时刻。
    jitter=False 用精确等待序列；True 用局部随机源，seed 控制夹具可复现。
    它是本章应用类型，不是 langgraph.types.RetryPolicy 的别名。"""

    # 教学默认值：不是提供商 SLA，也不是生产环境推荐配置。
    max_attempts: int = 3
    initial_delay: float = 1
    backoff_factor: float = 2
    max_delay: float = 4
    deadline: float = 10
    jitter: bool = False
    seed: int = 11

    def __post_init__(self) -> None:
        """在操作前验证全部参数，禁止运行中改变预算。"""
        _positive_integer(self.max_attempts, "max_attempts")
        _finite_number(self.initial_delay, "initial_delay")
        _finite_number(self.backoff_factor, "backoff_factor", minimum=1)
        _finite_number(self.max_delay, "max_delay", minimum=self.initial_delay)
        _finite_number(self.deadline, "deadline")
        if self.deadline <= 0:
            raise ValueError("deadline 必须大于零")
        if not isinstance(self.jitter, bool):
            raise ValueError("jitter 必须是布尔值")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed 必须是整数")


def classify(exc: BaseException) -> str:
    """把明确故障映射为应用分类，不对原始异常文本做猜测。

    优先识别显式 daily_quota，再区分瞬时、余额、鉴权和参数故障。
    普通 429 不推出免费额度用完；未知状态码归 UNKNOWN，不自动重试。
    返回分类本身不会抛异常；是否捕获并重试由调用方边界决定。"""
    if isinstance(exc, ServiceFailure):
        if exc.reason == "daily_quota":
            return "QUOTA"
        if exc.status_code in (429, 500, 503):
            return "TRANSIENT"
        return {402: "BALANCE", 401: "AUTH", 400: "INVALID", 422: "INVALID"}.get(exc.status_code, "UNKNOWN")
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "TRANSIENT"
    return "UNKNOWN"


class CircuitBreaker:
    """同一依赖共享的可用性保护器：CLOSED → OPEN → HALF_OPEN。

    只有明确瞬时故障增加计数；余额、鉴权等账户/输入问题不熔断共享服务。
    锁只保护资格与状态提交，下游操作在锁外执行，避免让所有请求排队。
    _generation 标记一代状态，过时的并发成功不能把新打开的断路器关掉。
    它仅在当前进程生效，不包含生产监控、分布式存储或账户配额服务。"""

    def __init__(self, clock: VirtualClock, threshold: int = 3, cooldown: float = 5):
        """创建独立依赖的断路器，不把余额或鉴权错误计入阈值。"""
        _positive_integer(threshold, "threshold")
        _finite_number(cooldown, "cooldown")
        self.clock = clock
        self.threshold = threshold
        self.cooldown = cooldown
        self._state = "CLOSED"
        self._failures = 0
        self._opened_at = 0.0
        # 每次重要状态变化推进代号；call 记录资格代号，_finish 拒绝旧结果。
        self._generation = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """返回实际状态；冷却完成后由下一次调用领取探测资格。"""
        with self._lock:
            return self._state

    def call(self, operation: Callable[[], Any]) -> Any:
        """领取执行资格并调用 operation()，返回其结果或保留原异常。

        冷却中的 OPEN、已有探测的 HALF_OPEN 会直接抛 CircuitOpen，不调用下游。
        OPEN 冷却到期后，由实际到来的一个请求领取半开资格，没有后台定时器。
        捕获 BaseException 仅为提交探测结果、释放资格，随后原样 raise；
        这不表示取消、程序缺陷或所有异常都可以被 run_with_retry 重试。"""
        with self._lock:
            if self._state == "OPEN":
                if self.clock.now - self._opened_at < self.cooldown:
                    raise CircuitOpen("断路器处于冷却期")
                self._state = "HALF_OPEN"
                self._generation += 1
            elif self._state == "HALF_OPEN":
                raise CircuitOpen("断路器已有探测运行")
            generation = self._generation
            is_probe = self._state == "HALF_OPEN"
        try:
            result = operation()
        except BaseException as exc:
            # 所有退出路径都释放半开资格，但程序缺陷仍原样抛给调用者。
            self._finish(generation, is_probe, classify(exc))
            raise
        self._finish(generation, is_probe, "SUCCESS")
        return result

    def _finish(self, generation: int, is_probe: bool, outcome: str) -> None:
        """只提交本代调用结果，隔离状态变化前就已在途的并发请求。

        TRANSIENT 达阈值或半开失败会重新计冷却；SUCCESS 清零并关闭。
        半开遇到非瞬时异常则释放资格并回 OPEN，但不增加计数或刷新冷却。
        调用方仍须停止余额/权限路径，断路器不能替代账户条件管理。"""
        with self._lock:
            if generation != self._generation:
                return
            if outcome == "TRANSIENT":
                self._failures += 1
                if is_probe or self._failures >= self.threshold:
                    self._state = "OPEN"
                    self._opened_at = self.clock.now
                    self._generation += 1
            elif outcome == "SUCCESS":
                self._failures = 0
                self._state = "CLOSED"
                if is_probe:
                    self._generation += 1
            elif is_probe:
                # 余额等不能证明可用性恢复，也不能增加故障计数或重置冷却。
                # 取消本次资格后下一请求可重新探测，避免半开永久占用。
                self._state = "OPEN"
                self._generation += 1


def run_with_retry(
    operation: Callable[[], Any],
    *,
    clock: VirtualClock,
    budget: RetryBudget | None = None,
    breaker: CircuitBreaker | None = None,
) -> dict[str, Any]:
    """执行一个应用重试层，返回状态、原因、真实下游次数和虚拟耗时。

    operation 是零参数回调；clock 同时观察操作耗时和等待；budget=None
    采用教学默认值，breaker=None 表示不接共享断路器。
    requests 只在真正进入 operation 时增加，CircuitOpen 不计为下游请求。

    只捕获 ServiceFailure、TimeoutError、ConnectionError；未知程序异常向上抛出。
    已知故障按 deadline → 分类 → 次数上限判断；只有瞬时且有预算才退避。
    成功响应也要核对返回时间，不能把晚到结果算作按时成功。
    jitter 使用局部 Random(seed).uniform(0, 封顶退避)，不污染全局随机源。
    同步操作不能被虚拟时钟强制中断，超时只能在返回后识别。

    返回 status=SUCCESS 时附 result；失败仅保存安全 reason，不保存异常原文。
    operation 的成功结果仍须由业务边界做脱敏，通用策略不替代输出 Guardrail。"""
    budget = budget if budget is not None else RetryBudget()
    # 共享时钟可以复用，但每次 run 都从自己的 started_at 计算相对预算。
    started_at = clock.now
    waits: list[float] = []
    requests = 0
    next_delay = budget.initial_delay
    rng = random.Random(budget.seed)

    def report(status: str, reason: str, **extra: Any) -> dict[str, Any]:
        """只输出分类和计数，禁止把下游原始异常带入证据。"""
        return {
            "status": status, "reason": reason, "requests": requests,
            "waits": list(waits), "elapsed": clock.now - started_at, **extra,
        }

    def invoke() -> Any:
        """仅在实际进入下游时计数，断路器短路不增加请求次数。"""
        nonlocal requests
        requests += 1
        return operation()

    while requests < budget.max_attempts:
        if clock.now - started_at >= budget.deadline:
            return report("FAILED", "DEADLINE_EXCEEDED")
        try:
            result = breaker.call(invoke) if breaker is not None else invoke()
        except CircuitOpen:
            return report("FAILED", "CIRCUIT_OPEN")
        # 此处刻意不捕获普通 Exception：编程错误应使实验失败，而不是安静降级。
        except (ServiceFailure, TimeoutError, ConnectionError) as exc:
            if clock.now - started_at >= budget.deadline:
                return report("FAILED", "DEADLINE_EXCEEDED")
            reason = classify(exc)
            if reason != "TRANSIENT":
                return report("FAILED", reason)
            if requests >= budget.max_attempts:
                return report("FAILED", "MAX_ATTEMPTS")
        else:
            if clock.now - started_at > budget.deadline:
                return report("FAILED", "DEADLINE_EXCEEDED")
            return report("SUCCESS", "SUCCESS", result=result)

        delay = rng.uniform(0, next_delay) if budget.jitter else next_delay
        # 等待恰好到达 deadline 也没有下一请求的执行窗口，因此直接停止。
        if clock.now - started_at + delay >= budget.deadline:
            return report("FAILED", "DEADLINE_EXCEEDED")
        clock.sleep(delay)
        waits.append(delay)
        # 先封顶再用于下一轮；full jitter 是在这个上限内取值，并非 SDK 精确算法。
        next_delay = min(budget.max_delay, next_delay * budget.backoff_factor)

    return report("FAILED", "MAX_ATTEMPTS")
