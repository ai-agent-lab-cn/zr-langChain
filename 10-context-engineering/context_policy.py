"""第10章上下文策略：完整消息组裁剪、可信作用域、进程内记忆治理。"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from threading import RLock
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.store.memory import InMemoryStore


# 【身份上下文】把“当前是谁”与聊天消息分开。
# frozen=True 禁止常规字段赋值，减少执行中意外替换身份；它本身不验证登录凭据。
@dataclass(frozen=True)
class AppContext:
    """教学后端注入的身份；真实应用必须由认证层创建，不能相信模型提供的 ID。"""
    tenant_id: str
    user_id: str

    # dataclass 创建实例后执行此检查，只保证标识非空，不承担真实身份认证。
    def __post_init__(self):
        if not self.tenant_id.strip() or not self.user_id.strip():
            raise ValueError("租户与用户不能为空")


# 独立异常类型便于调用方区分“预算不够”和“消息协议错误”。
class BudgetExceeded(ValueError):
    """最新完整交互无法装入预算；不静默删掉当前问题。"""


# 【消息协议】pending 跟踪尚未返回的调用，seen 跟踪已经用过的调用 ID。
# 本函数要求传入的是完整交互；不用于允许工具仍未返回的中间状态。
def validate_protocol(messages):
    """拒绝孤立、重复或未完成的工具消息；返回已配对的工具数量。"""
    # 等待结果的调用 ID；同一 AI 响应可同时登记多个工具调用。
    pending = set()
    # 累计历史中已出现的 ID，防止跨轮次复用同一个工具调用标识。
    seen = set()
    pairs = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            if message.tool_call_id not in pending:
                raise ValueError("工具结果没有唯一对应的调用")
            # 配对完成后移出待处理集合；同一个结果第二次出现会被拒绝。
            pending.remove(message.tool_call_id)
            pairs += 1
        else:
            # 普通消息意味着要开始下一段交互；上一组工具结果必须已经收齐。
            if pending:
                raise ValueError("工具结果尚未完整返回")
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    if not call["id"] or call["id"] in seen:
                        raise ValueError("工具调用 ID 为空或重复")
                    pending.add(call["id"])
                    seen.add(call["id"])
    if pending:
        raise ValueError("工具调用未配对")
    return pairs


# 【临时裁剪】输入为消息序列，输出为选中的新列表；消息对象本身不复制或修改。
# counter 可注入：运行时估算 Token，测试时用 len 明确验证消息组边界。
# 这里只计算传入消息；动态系统提示、工具 schema 等额外开销需另行核算。
def select_history(messages, budget, *, counter=count_tokens_approximately):
    """以 HumanMessage 开始的完整轮次为单位保留后缀，不拆并行工具组。

    budget 只用于消息估算；系统提示与工具 schema 由外层单独预留预算。
    系统消息可保留；当前轮装不下时明确拒绝。不会更改输入 State。
    """
    if budget <= 0:
        raise BudgetExceeded("预算必须大于零")
    validate_protocol(messages)
    prefix = []
    # 浅拷贝列表，下面 pop 只操作局部列表，不改原 State 的消息序列。
    body = list(messages)
    # 已有的前导系统消息随候选历史保留，并计入 counter；其他额外提示不在此计算。
    # 当前锁定版本的摘要 Middleware 用 lc_source 标记生成的摘要消息。
    # 摘要作为续接依据与前导系统消息一起保留；放不下则报错，不悄悄丢失更正事实。
    while body and (isinstance(body[0], SystemMessage) or (
        isinstance(body[0], HumanMessage)
        and body[0].additional_kwargs.get("lc_source") == "summarization"
    )):
        prefix.append(body.pop(0))
    # 每个 HumanMessage 作为候选起点，跳过的只能是此前完整轮次。
    starts = [i for i, message in enumerate(body) if isinstance(message, HumanMessage)]
    if not starts:
        raise ValueError("没有当前用户请求")
    # 从最长可用后缀开始尝试；只在完整用户轮次边界裁剪。
    for start in starts:
        selected = prefix + body[start:]
        if counter(selected) <= budget:
            return selected
    raise BudgetExceeded("当前完整轮次超过消息预算")


# 【长期记忆入口】把 namespace、允许字段、有效期与版本规则收拢到一处。
# Store 负责存数据；这里的应用策略负责决定什么数据可以返回给工具和 Prompt。
class MemoryRepository:
    """复用真实 Store 的教学仓库；锁、版本号与 tombstone 仅限当前进程。

    不宣称数据库事务、真实认证或清除旧 Checkpoint / Trace。
    """
    # store 可注入以共享记忆；clock 可注入以在测试中直接推进时间，无需等待。
    def __init__(self, store=None, *, clock=time.time):
        self.store = store if store is not None else InMemoryStore()
        self.clock = clock
        # 记录仓库实例内的最新版本，包含已删条目的版本墓碑。
        self._versions = {}
        # 串行化同一实例中的写入与删除；不保护其他实例、进程或绕过仓库的直接写入。
        self._lock = RLock()

    # 元组按“租户 / 用户 / 用途”分区；调用方不能通过工具参数任意指定他人空间。
    # namespace 是组织方式，授权仍依赖 ctx 来自可信认证层这一前提。
    @staticmethod
    def namespace(ctx):
        return (ctx.tenant_id, ctx.user_id, "preferences")

    # 统一读取入口：不存在、已过期、偏好不在白名单中，都返回 None。
    # 因此调用方必须提供默认行为，不能假定每次查询一定命中。
    def get(self, ctx):
        item = self.store.get(self.namespace(ctx), "reply-style")
        if item is None:
            return None
        value = item.value
        expires_at = value.get("expires_at")
        # 到期时刻本身也视为失效；仅隐藏条目，不在读取中自动执行物理清理。
        if expires_at is not None and expires_at <= self.clock():
            return None
        # 即使 Store 被写入非法文本，也不能把它提升为系统指令。
        if value.get("style") not in {"简洁", "详细"}:
            return None
        # 返回浅拷贝，避免调用方改动顶层字段时直接修改本次取到的映射。
        return dict(value)

    # Store.search 支持 namespace 前缀；这里传完整用户空间以避免范围过宽。
    # 搜索结果还要经过 get 的有效期与枚举检查，不能绕过读取策略。
    def search(self, ctx):
        # 使用完整用户 namespace，而不是宽泛租户前缀。
        items = self.store.search(self.namespace(ctx), limit=10)
        current = self.get(ctx)
        # namespace 参数是前缀，不是精确相等条件；必须排除子空间中的同名条目。
        return [item for item in items if item.namespace == self.namespace(ctx)
                and item.key == "reply-style" and current is not None]

    # expected_version 表示调用方认为的旧版本；只有与当前版本一致才允许写入。
    # ttl 的单位为秒；本例保存 expires_at 并在读取时检查，没有启动自动清理任务。
    def put(self, ctx, style, *, expected_version, ttl=None):
        if style not in {"简洁", "详细"}:
            raise ValueError("偏好只能是允许的枚举")
        if ttl is not None and (not isfinite(ttl) or ttl <= 0):
            raise ValueError("有效期必须为正数")
        ns = self.namespace(ctx)
        with self._lock:
            existing = self.store.get(ns, "reply-style")
            # 优先使用实例内版本；首次接触已有条目时才回读其 version，空空间从 0 开始。
            current = self._versions.get(ns, existing.value.get("version", 0) if existing else 0)
            if expected_version != current:
                raise ValueError("记忆版本冲突")
            version = current + 1
            # Store.put 替换同 namespace / key 的完整文档；这里明确构造每一个需要保存的字段。
            self.store.put(ns, "reply-style", {
                "style": style, "language": "zh-CN", "version": version,
                "source": "user_explicit", "updated_at": self.clock(),
                "expires_at": None if ttl is None else self.clock() + ttl,
            })
            self._versions[ns] = version
            return version

    # 删除条目后仍推进进程内版本，形成 tombstone（版本墓碑）。
    # 旧调用方不能再以“初次创建”的旧版本写回；不代表清除所有历史副本。
    def delete(self, ctx):
        ns = self.namespace(ctx)
        with self._lock:
            item = self.store.get(ns, "reply-style")
            current = self._versions.get(ns, item.value.get("version", 0) if item else 0)
            self.store.delete(ns, "reply-style")
            self._versions[ns] = current + 1  # 保留版本墓碑，拒绝旧版本回写。
