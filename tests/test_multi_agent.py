"""真实 StateGraph 的离线协议、路由与控制权回归测试。"""
import importlib.util
import unittest
import sys
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

spec = importlib.util.spec_from_file_location("multi_agent", Path(__file__).resolve().parents[1] / "09-langgraph-multi-agent/main.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

class MultiAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_supervisor_returns_after_subagent(self):
        result = await m.build_supervisor().ainvoke({"messages": [HumanMessage(content="账号无法登录")], "steps": []})
        m.validate(result, "supervisor")
        self.assertEqual(result["steps"], ["supervisor", "account_agent", "supervisor"])

    async def test_handoff_changes_owner_and_checkpoint(self):
        graph = m.build_handoff()
        config = {"configurable": {"thread_id": "test-handoff"}}
        result = await graph.ainvoke({"messages": [HumanMessage(content="包裹物流异常")], "steps": []}, config)
        m.validate(result, "handoff")
        self.assertEqual(result["active_agent"], "logistics_agent")
        self.assertEqual((await graph.aget_state(config)).next, ())
        self.assertEqual(result["steps"], ["triage", "logistics_agent"])

    async def test_unknown_request_does_not_transfer(self):
        for builder in (m.build_supervisor, m.build_handoff):
            result = await builder().ainvoke({"messages": [HumanMessage(content="你好")], "steps": []}, {"configurable": {"thread_id": "unknown"}})
            self.assertFalse(any(isinstance(x, ToolMessage) for x in result["messages"]))

    async def test_empty_request_rejected(self):
        with self.assertRaises(ValueError):
            await m.build_supervisor().ainvoke({"messages": [HumanMessage(content=" ")], "steps": []})

    def test_unpaired_tool_call_rejected(self):
        with self.assertRaises(AssertionError):
            m.validate({"messages": [AIMessage(content="", tool_calls=[{"name": "account_agent", "args": {"task": "x"}, "id": "missing"}])], "steps": []}, "supervisor")
