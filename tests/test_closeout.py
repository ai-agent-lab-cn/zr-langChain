"""整库收尾回归：只验证已复现的错误，不请求模型或远程服务。"""
import asyncio
from contextlib import redirect_stdout
from dataclasses import replace
import importlib.util
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langsmith import tracing_context

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    """按路径载入数字章节，避免不同 main.py 共享模块名。"""
    path = ROOT / relative
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class BasicsCloseoutTests(unittest.TestCase):
    def test_ch01_explicit_gemini_key_wins_over_legacy_google_key(self):
        from langchain.chat_models import init_chat_model
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'dummy-gemini', 'GOOGLE_API_KEY': 'dummy-google'}):
            with patch('langchain.agents.create_agent') as create:
                load('01-basic-agent/main.py', 'closeout_ch01')
            model = create.call_args.kwargs['model']
            if isinstance(model, str):
                model = init_chat_model(model)
            self.assertEqual(model.google_api_key.get_secret_value(), 'dummy-gemini')

    def test_ch03_unrelated_tasks_and_swapped_owners_are_rejected(self):
        m = load('03-structured-output/main.py', 'closeout_ch03')
        def meeting(first, second):
            return m.MeetingActions(topic='客服工单智能分类', tasks=[
                m.ActionItem(task=first, owner='张三', priority='high', due_date='2026-10-25'),
                m.ActionItem(task=second, owner='李四', priority='medium', due_date='2026-10-28')])
        self.assertEqual(m.validate_result('meeting', meeting('提交需求清单', '完成演示文稿')), [])
        for texts in [('删除全部数据库', '伪造报告'), ('完成演示文稿', '提交需求清单')]:
            with self.subTest(texts=texts):
                self.assertTrue(m.validate_result('meeting', meeting(*texts)))

    def test_ch05_real_tool_validation_error_cannot_satisfy_result(self):
        m = load('05-mcp-agent/main.py', 'closeout_ch05')
        def calculate(operation: str, a: float, b: float) -> float:
            """Calculate a product for the local tool validation test."""
            return a * b
        builder = StateGraph(MessagesState)
        builder.add_node('tools', ToolNode([calculate]))
        builder.add_edge(START, 'tools')
        builder.add_edge('tools', END)
        request = AIMessage(content='', tool_calls=[{
            'name': 'calculate', 'args': {'operation': 'multiply', 'a': 1000}, 'id': 'bad-call'}])
        messages = builder.compile().invoke({'messages': [request]})['messages']
        self.assertEqual(messages[-1].status, 'error')
        messages.append(AIMessage(content='125 × 8 = 1000'))
        self.assertTrue(m.check_online('math', messages))


class WorkflowCloseoutTests(unittest.TestCase):
    def test_other_offline_helpers_override_inherited_tracing(self):
        from langchain_core.tools import tool

        @tool
        def get_mock_weather(city: str) -> str:
            """Return the same fixed teaching weather facts without MCP transport."""
            return '南京 22 模拟' if city == '南京' else '仅支持南京'

        @tool
        def calculate(operation: str, a: float, b: float) -> float:
            """Calculate the local self-test values without a provider."""
            return a * b if operation == 'multiply' else a + b

        for chapter in ('04-agent-memory', '05-mcp-agent', '06-rag-agent', '09-langgraph-multi-agent'):
            with self.subTest(chapter=chapter):
                m = load(f'{chapter}/main.py', f'closeout_extra_{chapter[:2]}')
                with patch('langchain_core.tracers.langchain.LangChainTracer._persist_run_single') as persist, \
                     patch('langsmith.run_trees.RunTree.post') as post, \
                     patch('langsmith.run_trees.RunTree.patch') as update, \
                     patch.object(m, 'load_dotenv') as dotenv:
                    with tracing_context(enabled=True), redirect_stdout(io.StringIO()):
                        if chapter.startswith('05'):
                            self.assertTrue(asyncio.run(m.self_test([get_mock_weather, calculate])))
                        elif chapter.startswith('09'):
                            args = NS(self_test=True, offline=True, case='all', evidence=None)
                            self.assertEqual(asyncio.run(m.run(args)), 0)
                        else:
                            self.assertTrue(m.self_test())
                    self.assertEqual((persist.call_count, post.call_count, update.call_count), (0, 0, 0))
                    dotenv.assert_not_called()

    def test_ch07_ch08_offline_ignore_inherited_tracing_and_dotenv(self):
        for chapter in ('07-langgraph-workflow', '08-langgraph-hitl'):
            for self_test in (True, False):
                with self.subTest(chapter=chapter, self_test=self_test):
                    with patch('dotenv.load_dotenv') as dotenv:
                        m = load(f'{chapter}/main.py', f'closeout_{chapter[:2]}_{self_test}')
                        args = NS(self_test=self_test, offline=True, case='all', stream=False,
                                  interactive=False, resume_batch=False)
                        with patch('langchain_core.tracers.langchain.LangChainTracer._persist_run_single') as persist, \
                             patch('langsmith.run_trees.RunTree.post') as post, \
                             patch('langsmith.run_trees.RunTree.patch') as update:
                            with tracing_context(enabled=True), redirect_stdout(io.StringIO()):
                                self.assertEqual(asyncio.run(m.main_async(args)), 0)
                        self.assertEqual(persist.call_count, 0)
                        self.assertEqual(post.call_count, 0)
                        self.assertEqual(update.call_count, 0)
                        dotenv.assert_not_called()

    def test_ch10_exact_memory_survives_many_child_namespaces(self):
        m = load('10-context-engineering/context_policy.py', 'closeout_policy')
        repo, ctx = m.MemoryRepository(), m.AppContext('tenant', 'user')
        for i in range(11):
            repo.store.put((*repo.namespace(ctx), str(i)), 'reply-style', {'style': '详细'})
        repo.put(ctx, '简洁', expected_version=0)
        rows = repo.search(ctx)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].namespace, ('tenant', 'user', 'preferences'))
        self.assertEqual(rows[0].value['style'], '简洁')


class EvaluationCloseoutTests(unittest.TestCase):
    def test_ch09_audit_rejects_duplicate_and_empty_call_ids(self):
        m = load('scripts/verify_chapter09_trace.py', 'closeout_audit09')
        for ids in [('same', 'same'), ('',)]:
            calls = [{'id': call_id, 'name': 'logistics_agent'} for call_id in ids]
            data = {'steps': ['triage', 'logistics_agent'], 'active_agent': 'logistics_agent', 'messages': [
                {'type': 'ai', 'tool_calls': calls},
                {'type': 'tool', 'name': 'logistics_agent', 'tool_call_id': ids[0]},
                {'type': 'ai', 'content': 'done'}]}
            root = NS(id='root', outputs=data, error=None, end_time=1, trace_id='root',
                      parent_run_id=None, run_type='chain', name='root', url='LOCAL_ONLY')
            llms = [NS(id=f'llm-{i}', error=None, end_time=1, trace_id='root', parent_run_id='root',
                       run_type='llm', name='model', prompt_tokens=10, completion_tokens=2, total_tokens=12)
                    for i in range(2)]
            with self.subTest(ids=ids), self.assertRaises((AssertionError, ValueError)):
                m.audit(root, [root, *llms], 'handoff')

    def test_ch12_candidate_fail_cannot_receive_regression_pass(self):
        m = load('12-agent-evaluation/evaluation_core.py', 'closeout_eval')
        for previous in ('REVIEW', 'FAIL'):
            with self.subTest(previous=previous):
                comparison = m.compare_versions([m.Score('a', previous, {}, ())],
                    [m.Score('a', 'FAIL', {'no_side_effect': False}, ())])
                self.assertEqual(comparison['status'], 'FAIL')

    def test_ch12_raw_message_order_is_part_of_protocol_evidence(self):
        m = load('12-agent-evaluation/graph_cases.py', 'closeout_graph12')
        from evaluation_core import GOLDEN_DATASET, evaluate_case
        combined = next(c for c in GOLDEN_DATASET if c.case_id == 'combined')
        ai1 = AIMessage(content='', tool_calls=[{'id': 'c1', 'name': 'lookup_account', 'args': {}}])
        ai2 = AIMessage(content='', tool_calls=[{'id': 'c2', 'name': 'lookup_business', 'args': {}}])
        t1 = ToolMessage(content='ok', tool_call_id='c1', name='lookup_account')
        t2 = ToolMessage(content='ok', tool_call_id='c2', name='lookup_business')
        for messages, expected in [([ai1, t1, ai2, t2], 'PASS'), ([ai1, ai2, t1, t2], 'FAIL')]:
            with self.subTest(expected=expected):
                def output(state):
                    return {'messages': messages, 'status': 'advice_ready', 'steps': ['finalize']}
                builder = StateGraph(dict)
                builder.add_node('observed', output)
                builder.add_edge(START, 'observed')
                builder.add_edge('observed', END)
                observed = m.run_ticket(builder.compile(), combined.case_id, combined.ticket, 'candidate')
                self.assertEqual(evaluate_case(combined, observed).outcome, expected)


if __name__ == '__main__':
    unittest.main()
