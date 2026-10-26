"""第11章：虚构工单的权限、幂等账本与真实 LangGraph 人工审批。

仅演示同进程行为：Identity 由可信调用方传入，不能充当真实登录系统；
内存锁与假服务不提供跨进程事务。任何外部写入都须另接真实授权和幂等接口。


配套流程图：同目录 basic-knowledge.md，写入对账与审批（write-recovery / approval）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
import uuid
from threading import RLock
from typing import Callable, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langsmith import tracing_context


@dataclass(frozen=True)
class Identity:
    """由可信入口提供的身份快照，模型和工具资料无权替换它。

    tenant_id/user_id 是虚构作用域标识，role 区分业务用户和审核者。
    frozen=True 防止正常代码就地修改，但不等于签名、登录或生产鉴权。
    教学授权只开放 T1/U1 的 A100；不得把传入任意 Identity 当成真实认证。"""
    tenant_id: str
    user_id: str
    role: str = 'user'

    def __post_init__(self):
        """限制标识与角色，避免把任意资料文本塞入可信身份位置。"""
        for value in (self.tenant_id, self.user_id):
            if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value):
                raise ValueError('身份标识无效')
        if self.role not in {'user', 'reviewer'}:
            raise ValueError('角色无效')


@dataclass(frozen=True)
class Action:
    """待执行的具体备注修改：订单、备注内容与动作版本。

    构造时校验格式、长度、假秘密标记和版本，非法动作在进入图前即失败。
    版本改变会改变业务指纹；备注规则只是教学确定性 Guardrail，
    不是通用 Prompt Injection 检测器，也不授权对某个订单执行写入。"""
    order_id: str
    note: str
    version: int = 1

    def __post_init__(self):
        """教学 Guardrail：长度、版本与假秘密检查，不宣称通用注入检测。"""
        if not isinstance(self.order_id, str) or not re.fullmatch(r'[A-Z][0-9]{3}', self.order_id):
            raise ValueError('订单号格式无效')
        if not isinstance(self.note, str) or not self.note.strip() or len(self.note) > 120:
            raise ValueError('备注长度无效')
        if any(word in self.note for word in ('FAKE_SECRET_CH11', '导出全部', '忽略此前')):
            raise ValueError('备注被输入规则阻断')
        if type(self.version) is not int or self.version < 1:
            raise ValueError('动作版本无效')


def authorize(actor: Identity, action: Action) -> None:
    """每次执行或读取前检查资源归属；猜中 A100 不等于获得权限。"""
    if (actor.tenant_id, actor.user_id, actor.role, action.order_id) != ('T1', 'U1', 'user', 'A100'):
        raise PermissionError('当前身份无权访问资源')


def action_digest(actor: Identity, action: Action, operation_id: str) -> str:
    """为同一业务意图的身份、参数和 operation_id 生成稳定内容指纹。

    重试、对账与普通故障恢复必须复用 operation_id；不同意图用新标识。
    规范化 JSON 绑定 Action 全字段，便于拒绝同键替换参数。
    这是业务指纹，不能直接用作审批票据；票据还绑定线程、到期及随机值。"""
    if not isinstance(operation_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', operation_id):
        raise ValueError('操作标识无效')
    data = {'actor': asdict(actor), 'action': asdict(action), 'operation_id': operation_id}
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class UnknownEffect(Exception):
    """没有拿到可靠结果；调用方必须核对账本，不能新建意图盲重放。"""


class MemoryLedger:
    """同进程假服务效果与客户端结果账本，用于观察幂等及结果未知。

    _records 按 (租户, 用户, operation_id) 保存 digest/status；
    _effects 用相同键保存假服务已提交的实际效果。两者分开，才能区分
    “服务端已生效”和“客户端知道成功”。整个检查/写入由同一 RLock 串行化。
    这是受控教学模型，不提供真实服务原子性、跨进程持久化或分布式事务。"""

    def __init__(self):
        """效果与客户端记录分开，才能演示“已提交但响应丢失”的故障窗口。"""
        # _records 是调用方对结果的认识；_effects 是假服务中已发生的效果。
        self._records = {}
        self._effects = {}
        self._lock = RLock()

    @property
    def effects(self) -> list[dict]:
        """返回效果快照，外部检查不能通过修改列表篡改账本。"""
        with self._lock:
            return [dict(effect) for effect in self._effects.values()]

    def _key(self, actor, action, operation_id):
        """先鉴权，再计算 (作用域键, 参数指纹)，供执行和对账共用。

        相同 operation_id 查到不同 digest 时抛 ValueError，不能当作新操作执行。
        本方法本身不加锁，调用者 execute/reconcile 已在同一 RLock 内。"""
        authorize(actor, action)
        digest = action_digest(actor, action, operation_id)
        key = (actor.tenant_id, actor.user_id, operation_id)
        existing = self._records.get(key)
        if existing and existing['digest'] != digest:
            raise ValueError('同操作标识的参数发生冲突')
        return key, digest

    def execute(self, actor: Identity, action: Action, operation_id: str, *,
                lose_response: bool = False, fail_before: bool = False) -> dict:
        """对获准动作执行或去重，返回 SUCCESS 与 deduplicated 标记。

        lose_response/fail_before 是假服务的故障注入开关，不执行网络请求。
        已成功：返回已有结果；结果未知：抛 UnknownEffect，要求先对账；
        提交前失败：记录 FAILED、抛 ConnectionError，未产生效果，可按策略重试。
        正常提交先记 UNKNOWN_EFFECT，再保存效果；丢响应时保持未知，
        只有确认响应路径成功后才将客户端记录转为 SUCCESS。

        同进程并发在锁内去重；真实服务须提供自己的幂等接口及结果核对能力。"""
        with self._lock:
            key, digest = self._key(actor, action, operation_id)
            record = self._records.get(key)
            if record and record['status'] == 'SUCCESS':
                return {'status': 'SUCCESS', 'deduplicated': True}
            if record and record['status'] == 'UNKNOWN_EFFECT':
                raise UnknownEffect('先核对外部状态')
            if fail_before:
                self._records[key] = {'digest': digest, 'status': 'FAILED'}
                raise ConnectionError('虚构服务在提交前失败')
            # 只有假服务本身知道效果已发生；调用方记录先保持结果未知。
            self._records[key] = {'digest': digest, 'status': 'UNKNOWN_EFFECT'}
            self._effects[key] = {'digest': digest, 'order_id': action.order_id, 'note': action.note}
            if lose_response:
                raise UnknownEffect('虚构服务已提交但响应丢失')
            self._records[key]['status'] = 'SUCCESS'
            return {'status': 'SUCCESS', 'deduplicated': False}

    def reconcile(self, actor: Identity, action: Action, operation_id: str) -> dict:
        """核对假服务效果并返回 SUCCESS 或 UNKNOWN_EFFECT，不新增写入。

        能找到同键同指纹效果时，仅修正客户端账本为成功；找不到可靠结果时
        继续保留未知，不能凭“没查到”推断此前没有生效并自动创建新操作。"""
        with self._lock:
            key, digest = self._key(actor, action, operation_id)
            effect = self._effects.get(key)
            if effect and effect['digest'] == digest:
                self._records[key] = {'digest': digest, 'status': 'SUCCESS'}
                return {'status': 'SUCCESS'}
            return {'status': 'UNKNOWN_EFFECT'}


def safe_tool_result(raw: dict, actor: Identity) -> dict:
    """把不可信工具字典投影为授权事实，返回三个白名单字段。

    先验证容器与订单归属，再验证 status/source 的允许值；未知指令、调试
    字段和假秘密不随资料进入后续上下文。非法字段值会被拒绝而非悄悄放行。
    本函数不保存 State；调用方仍须只把返回的投影结果写入自己的状态。"""
    if not isinstance(raw, dict):
        raise ValueError('工具结果必须是对象')
    authorize(actor, Action(raw.get('order_id', ''), '只读校验'))
    if raw.get('status') not in {'运输中', '已签收'} or raw.get('source') != 'fixture-logistics':
        raise ValueError('工具结果不符合允许契约')
    return {key: raw[key] for key in ('order_id', 'status', 'source')}


class ApprovalState(TypedDict, total=False):
    """Checkpoint 仅保存已校验的动作与状态，未过滤的用户或工具原文不入图。"""
    actor: dict
    action: dict
    operation_id: str
    # 这里放审批票据摘要；不是 MemoryLedger 使用的稳定业务指纹。
    action_digest: str
    expires_at: float
    status: str
    reason: str


class ApprovalDemo:
    """带独立入口检查的真实 HITL 教学图：review → write 或直接结束。

    start 校验后登记线程并暂停；get_state 只允许线程所有者查询；
    resume 校验审核角色、租户、票据、决定后消费审批，再交给 Command。
    审核者并不是订单执行身份，write 使用最初的业务用户重新鉴权。
    图与票据、归属、账本均在当前实例内；不要直接暴露内部 graph 作为外部接口。"""

    def __init__(self, *, clock: Callable[[], float], ledger: MemoryLedger | None = None):
        """注入时钟和账本；图、线程归属、待审记录均只保存在本实例内。"""
        self.clock = clock
        self.ledger = ledger if ledger is not None else MemoryLedger()
        self.review_entries = 0
        # thread -> {actor: 原请求身份, digest: 本次票据, pending: 是否仍允许投票}。
        self._threads = {}
        self._lock = RLock()
        builder = StateGraph(ApprovalState)
        # 显式分开审核和写入，避免审核节点重入就再次产生副作用。
        builder.add_node('review', self._review)
        builder.add_node('write', self._write)
        builder.add_edge(START, 'review')
        builder.add_conditional_edges('review', self._route, {'write': 'write', 'stop': END})
        builder.add_edge('write', END)
        self.graph = builder.compile(checkpointer=InMemorySaver())

    def _review(self, state: ApprovalState) -> dict:
        """执行 interrupt 并等待决定；恢复时从本节点开头重入。

        review_entries 因此会增加两次，不能在 interrupt 前执行无保护副作用。
        恢复后先检查到期，再处理 reject；approve/edit 的动作重新构造并授权。
        这里只产出状态，真正写入留给后续节点，使拒绝/过期路径无写入效果。"""
        self.review_entries += 1
        decision = interrupt({'action': state['action'], 'action_digest': state['action_digest']})
        # 时间在恢复后再次检查，不能拿首次暂停前的判断为审批续期。
        if self.clock() >= state['expires_at']:
            return {'status': 'REJECTED', 'reason': 'EXPIRED'}
        if decision['type'] == 'reject':
            return {'status': 'REJECTED', 'reason': 'HUMAN_REJECTED'}
        action = Action(**decision.get('action', state['action']))
        authorize(Identity(**state['actor']), action)
        return {'status': 'APPROVED', 'action': asdict(action)}

    @staticmethod
    def _route(state: ApprovalState) -> str:
        """只有明确批准能进入写入分支；拒绝、过期均直接结束。"""
        return 'write' if state['status'] == 'APPROVED' else 'stop'

    def _write(self, state: ApprovalState) -> dict:
        """执行前重新授权；幂等账本保护相同业务操作，审批本身不代替它。"""
        result = self.ledger.execute(Identity(**state['actor']), Action(**state['action']), state['operation_id'])
        return {'status': result['status'], 'reason': 'APPROVED_ACTION_EXECUTED'}

    @staticmethod
    def _config(thread_id: str) -> dict:
        """thread_id 仅定位图状态，权限校验由外层入口独立完成。"""
        return {'configurable': {'thread_id': thread_id}, 'recursion_limit': 6}

    def start(self, thread_id: str, actor: Identity, action: Action, operation_id: str) -> dict:
        """启动一个新线程，返回 WAITING_APPROVAL 与本次动作票据摘要。

        参数为线程标识、已认证业务身份、已校验 Action 和稳定操作标识。
        先授权并计算业务指纹，再在锁内拒绝线程复用、生成 60 秒有效的独立票据。
        票据摘要与业务幂等指纹不同：相同操作在新线程也不能使用旧线程的批准。
        首次 graph.invoke 返回的是预期中断，不是任务成功或技术异常。"""
        authorize(actor, action)
        business_digest = action_digest(actor, action, operation_id)
        if not isinstance(thread_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', thread_id):
            raise ValueError('线程标识无效')
        with self._lock, tracing_context(enabled=False):
            if thread_id in self._threads:
                raise ValueError('线程已经存在')
            expires_at = self.clock() + 60
            # 业务幂等键须稳定；审批票据却必须绑定本次线程、到期和独立随机值。
            # 分开两种指纹，防止拿旧线程的过期批准给相同业务意图重新续期。
            ticket = {'business_digest': business_digest, 'thread_id': thread_id,
                      'expires_at': expires_at, 'nonce': uuid.uuid4().hex}
            digest = hashlib.sha256(json.dumps(ticket, sort_keys=True).encode()).hexdigest()
            self._threads[thread_id] = {'actor': actor, 'digest': digest, 'pending': True}
            state = {'actor': asdict(actor), 'action': asdict(action), 'operation_id': operation_id,
                     'action_digest': digest, 'expires_at': expires_at, 'status': 'WAITING_APPROVAL'}
            self.graph.invoke(state, self._config(thread_id))
            return {'status': 'WAITING_APPROVAL', 'action_digest': digest}

    def get_state(self, thread_id: str, actor: Identity):
        """查询 Checkpoint 前检查线程归属，不因 ID 难猜而省略鉴权。"""
        with self._lock:
            record = self._threads.get(thread_id)
            if not record or record['actor'] != actor:
                raise PermissionError('线程不属于当前身份')
            return self.graph.get_state(self._config(thread_id))

    def resume(self, thread_id: str, reviewer: Identity, decision: dict) -> dict:
        """通过受控入口提交一次审批决定，并返回最终业务 status/reason。

        decision.type 只接受 approve/edit/reject，action_digest 必须匹配当前票据；
        仅 edit 可附完整 action，且修改后必须重新校验参数与资源归属。
        到期在 review 节点恢复后检查，不能因首次暂停时有效而无限续期。

        全部入口检查通过后先消费 pending，再调用 Command(resume=...)。
        即使后续写入失败也不能再次投票；图此时停在普通故障节点，重新发送
        “reject”不会退回审核节点。本例选择停止，统一故障恢复是另行扩展。
        入口抛出的异常不包含原始工具/用户内容；真实接口还须独立做身份认证。"""
        with self._lock, tracing_context(enabled=False):
            record = self._threads.get(thread_id)
            if not record or reviewer.role != 'reviewer' or reviewer.tenant_id != record['actor'].tenant_id:
                raise PermissionError('无权审核此线程')
            # 先校验审核资格，再解释票据/参数，避免未授权调用获取线程细节。
            if not record['pending']:
                raise ValueError('审批已经处理')
            if not isinstance(decision, dict) or decision.get('type') not in {'approve', 'edit', 'reject'}:
                raise ValueError('审批决定无效')
            if decision.get('action_digest') != record['digest']:
                raise ValueError('审批未绑定待审动作')
            if set(decision) - {'type', 'action_digest', 'action'}:
                raise ValueError('审批存在未知字段')
            if decision['type'] == 'edit':
                if not isinstance(decision.get('action'), dict):
                    raise ValueError('编辑后动作缺失')
                try:
                    action = Action(**decision['action'])
                except TypeError as exc:
                    raise ValueError('动作字段无效') from exc
                authorize(record['actor'], action)
            elif 'action' in decision:
                raise ValueError('仅 edit 可以修改动作')
            # 锁将检查与消费同一审批串行化，避免同进程并发 resume 通过两次。
            # 一旦消费审批就不能再次投票；write 失败后图停在普通故障节点，
            # 再送 Command(resume=reject) 并不会退回 review，反而可能重放写入。
            # 本例选择失败后停止，审批后故障的恢复服务留作单独扩展。
            record['pending'] = False
            result = self.graph.invoke(Command(resume=dict(decision)), self._config(thread_id))
            return {'status': result['status'], 'reason': result['reason']}
