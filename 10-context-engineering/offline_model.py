"""离线教学模型：只替换模型决策，不替换 Agent、工具和持久化。

固定响应只用于验证结构和可重复的输入边界，不代表真实模型质量。
"""
from __future__ import annotations

import json
import re
from uuid import uuid4
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


# 【模型替身】继承 BaseChatModel，使 create_agent 仍走标准模型调用与回调流程。
# role 决定教学角色，tool_names 记录当前允许调用的工具；不会连接模型服务。
# 语义响应可预测，但消息调用 ID 随机生成，所以它们不应被写成固定测试期望。
class OfflineChatModel(BaseChatModel):
    # BaseChatModel 使用 Pydantic 字段；这些声明是实例配置，不是联网模型参数。
    role: str = "assistant"
    tool_names: list[str] = []

    # 供框架标识模型实现的类型名称，不是可在 DeepSeek 平台选择的模型 ID。
    @property
    def _llm_type(self):
        return "chapter10-deterministic-offline"

    # create_agent 会绑定可用工具；返回带工具名称的新模型副本，保留原实例配置。
    # 该替身只实现本章用到的工具绑定能力，不模拟完整提供商协议。
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        names = [tool.name if hasattr(tool, "name") else tool["name"] for tool in tools]
        return self.model_copy(update={"tool_names": names})

    # 同步模型生成入口：输入已经过 Middleware 处理的消息，返回标准 ChatResult。
    # 分支顺序很关键：先处理专家角色与工具结果，避免工具返回后再次请求同一工具。
    # stop、run_manager 等参数保留框架调用接口；此替身不实现流式或真实推理。
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # 倒序找到最新用户消息；其余历史仍保留给订单回忆分支读取。
        user = next((str(x.content) for x in reversed(messages) if isinstance(x, HumanMessage)), "")
        # 专家只返回对应领域的固定建议，不再进入主 Agent 的委派分支。
        if self.role in {"account", "logistics"}:
            answer = ("账号建议：核实登录提示；不索取密码。" if self.role == "account"
                      else "物流建议：核实运单与最新节点。")
        elif isinstance(messages[-1], ToolMessage):
            # 同一轮可能有多个专家调用；汇总最近完整组中的结果。
            results = []
            for msg in reversed(messages):
                if not isinstance(msg, ToolMessage):
                    break
                results.append(str(msg.content))
            # 上面逆序收集结果，这里恢复原先的返回顺序后汇总。
            answer = "；".join(reversed(results))
        # 生成标准 tool_calls 交给 Agent 实际执行；此处不直接调用任何专家函数。
        elif "专家协作" in user and self.tool_names:
            calls = [{"name": name, "args": ({} if name == "account_expert" else {"order_id": "A100"}),
                      "id": f"offline-{uuid4().hex}"} for name in self.tool_names
                     if name in {"account_expert", "logistics_expert"}]
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=calls))])
        # 工具已绑定且当前请求涉及物流时，输出工具名称、JSON 参数与唯一 ID。
        elif "物流" in user and "lookup_order" in self.tool_names:
            order = re.search(r"[ABX]\d{3}", user)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[{
                "name": "lookup_order", "args": {"order_id": order.group(0) if order else "A100"},
                "id": "offline-" + uuid4().hex}]))])
        elif "回复风格" in user and "read_preference" in self.tool_names:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[{
                "name": "read_preference", "args": {}, "id": "offline-" + uuid4().hex}]))])
        # 无工具请求时从本次实际可见输入中回忆订单；读不到就明确报告没有订单号。
        else:
            history = "\n".join(str(x.content) for x in messages)
            order_ids = re.findall(r"[ABX]\d{3}", history)
            answer = f"当前订单：{order_ids[-1]}" if order_ids else "当前线程没有订单号"
        # 最终回复统一包装为框架标准返回对象，触发后续回调和消息状态更新。
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])


# 【摘要服务替身】读取收到的历史，提取最后出现的订单号作为受控夹具的更正结果。
# 这是针对本章数据约定的规则，不具备通用自然语言摘要或事实核验能力。
def offline_summary(messages):
    """可控摘要替身；从夹具实际内容抽取最后一次订单更正，不硬编码摘要答案。"""
    text = "\n".join(str(x.content) for x in messages)
    ids = re.findall(r"[ABX]\d{3}", text)
    if not ids:
        raise ValueError("摘要夹具缺少订单信息")
    # 显式区分已作废与当前有效的订单，便于验收事实关系，而不是只搜最终答案关键词。
    # 该规则只适用于本章“最后一次订单更正生效”的受控夹具。
    active = ids[-1]
    superseded = list(dict.fromkeys(order for order in ids if order != active))
    return json.dumps({"superseded_order_ids": superseded, "active_order_id": active,
                       "constraints": []}, ensure_ascii=False)
