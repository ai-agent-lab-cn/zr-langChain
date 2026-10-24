"""第10章 Trace 验收边界测试；只使用固定消息与计量，不读取密钥或请求网络。"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('chapter10_trace', SCRIPTS / 'verify_chapter10_trace.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def evidence():
    """手工期望：A=2、B=2、C=6、D=4 次模型调用；不是由被测函数反算。"""
    return [{'case': case, 'mode': 'online', 'status': 'PASS', 'trace_requested': True,
             'telemetry': {'model_calls': count, 'usage_observed_calls': count,
                           'observed_model_run_ids': [f'{case}-{i}' for i in range(count)],
                           'usage_metadata': [{'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12}
                                              for _ in range(count)]}}
            for case, count in [('A', 2), ('B', 2), ('C', 6), ('D', 4)]]


class TraceTests(unittest.TestCase):
    def test_online_evidence_requires_complete_unique_abcd_calls(self):
        rows = evidence()
        self.assertEqual(len(m.validate_evidence(rows)), 14)
        rows[1]['telemetry']['observed_model_run_ids'][0] = 'A-0'
        with self.assertRaises(m.AuditFailure):
            m.validate_evidence(rows)

    def test_offline_untraced_failed_or_partial_batches_are_rejected(self):
        for field, value in [('mode', 'offline'), ('trace_requested', False), ('status', 'FAILED')]:
            rows = evidence()
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(m.AuditFailure):
                m.validate_evidence(rows)
        with self.assertRaises(m.AuditFailure):
            m.validate_evidence(evidence()[:-1])

    def test_usage_count_must_match_model_calls(self):
        rows = evidence()
        rows[0]['telemetry']['usage_metadata'].pop()
        with self.assertRaises(m.AuditFailure):
            m.validate_evidence(rows)

    def test_constructor_messages_keep_tool_identity(self):
        ai = m.message({'type': 'constructor', 'id': ['langchain', 'AIMessage'], 'kwargs': {
            'content': '', 'tool_calls': [{'id': 'call-1', 'name': 'read_preference', 'args': {}}]}})
        tool = m.message({'type': 'constructor', 'id': ['langchain', 'ToolMessage'], 'kwargs': {
            'content': '简洁', 'tool_call_id': 'call-1', 'name': 'read_preference'}})
        self.assertEqual(m.protocol([ai, tool]), [
            {'tool': 'read_preference', 'tool_call_id': 'call-1', 'matched': True}])
        tool['name'] = 'wrong_tool'
        with self.assertRaises(m.AuditFailure):
            m.protocol([ai, tool])

    def test_orphan_or_unfinished_tool_exchange_is_rejected(self):
        ai = {'type': 'ai', 'tool_calls': [{'id': 'c1', 'name': 'x', 'args': {}}]}
        tool = {'type': 'tool', 'tool_call_id': 'c1', 'name': 'x'}
        for rows in [[tool], [ai], [ai, tool, tool], [ai, {'type': 'human', 'content': 'next'}]]:
            with self.subTest(rows=rows), self.assertRaises(m.AuditFailure):
                m.protocol(rows)

    def test_tokens_only_count_unique_leaf_models(self):
        parent = NS(id='root', run_type='chain', parent_run_ids=[], prompt_tokens=999,
                    completion_tokens=999, total_tokens=1998)
        llm = NS(id='llm', run_type='llm', parent_run_ids=['root'], prompt_tokens=10,
                 completion_tokens=2, total_tokens=12)
        self.assertEqual(m.tokens([parent, llm]), {'input': 10, 'output': 2, 'total': 12})
        with self.assertRaises(m.AuditFailure):
            m.tokens([llm, llm])
        child = NS(id='child', run_type='llm', parent_run_ids=['root', 'llm'], prompt_tokens=3,
                   completion_tokens=1, total_tokens=4)
        with self.assertRaises(m.AuditFailure):
            m.tokens([llm, child])

    def test_missing_or_inconsistent_token_fields_are_rejected(self):
        for completion, total in [(None, 10), (2, 99)]:
            row = NS(id='llm', run_type='llm', parent_run_ids=[], prompt_tokens=10,
                     completion_tokens=completion, total_tokens=total)
            with self.subTest(total=total), self.assertRaises(m.AuditFailure):
                m.tokens([row])

    def test_summary_facts_require_structured_superseded_array(self):
        def msg(facts):
            return {'type': 'human', 'content': 'Here is a summary:\n\n' + json.dumps(facts),
                    'additional_kwargs': {'lc_source': 'summarization'}}
        good = {'superseded_order_ids': ['A100'], 'active_order_id': 'B200', 'constraints': []}
        self.assertEqual(m.order_summary([msg(good)]), good)
        for changed in [{'superseded_order_ids': 'A100'}, {'superseded_order_ids': []},
                        {'superseded_order_ids': ['A100', 'B200']}, {'constraints': '中文'}]:
            with self.subTest(changed=changed), self.assertRaises(m.AuditFailure):
                m.order_summary([msg({**good, **changed})])

    def test_main_model_must_receive_same_tagged_summary(self):
        facts = {'superseded_order_ids': ['A100'], 'active_order_id': 'B200', 'constraints': []}
        state = [{'type': 'human', 'content': 'Summary:\n\n' + json.dumps(facts),
                  'additional_kwargs': {'lc_source': 'summarization'}}]
        self.assertTrue(m.verify_summary_delivery(state, [state]))
        # 仅出现字段名，或者收到另一份合法但不同的摘要，都不足以证明原摘要被送达。
        fake = [[{'type': 'human', 'content': 'superseded_order_ids'}]]
        changed = [[{'type': 'human', 'content': 'Summary:\n\n' + json.dumps({**facts, 'constraints': ['简洁']}),
                     'additional_kwargs': {'lc_source': 'summarization'}}]]
        for batches in [fake, changed]:
            with self.subTest(batches=batches), self.assertRaises(m.AuditFailure):
                m.verify_summary_delivery(state, batches)

    def test_existing_output_is_rejected_before_credentials_are_read(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / 'input.json', Path(tmp) / 'audit.json'
            source.write_text(json.dumps(evidence()))
            target.write_text('keep-me')
            with patch('dotenv.load_dotenv', side_effect=AssertionError('不允许读密钥')):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as result:
                        m.main([str(source), str(target)])
            self.assertEqual(result.exception.code, 2)
            self.assertEqual(target.read_text(), 'keep-me')


class QueryScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_id_lookup_avoids_default_time_window(self):
        from langsmith_reader import FIELDS

        class Client:
            def __init__(self):
                self.runs = self
                self.traces = self
                self.retrieved = []

            async def aread_project(self, *, project_name):
                return NS(id='project')

            def query(self, **kwargs):
                raise AssertionError('精确 ID 验收不应使用带默认时间窗口的查询')

            async def retrieve(self, run_id, **kwargs):
                self.retrieved.append((run_id, kwargs))
                return NS(id=run_id, run_type='LLM', trace_id='root', parent_run_ids=['root'])

            async def list_runs(self, *args, **kwargs):
                # 边界替身只负责精确ID读取，完整调用树由现有适配层测试与在线只读复验覆盖。
                raise m.AuditFailure('stop_after_exact_id_lookup')

        client = Client()
        with self.assertRaises(m.AuditFailure):
            await m.verify(client, evidence(), project='demo')
        self.assertEqual([rid for rid, _ in client.retrieved],
                         [f'{case}-{i}' for case, count in [('A', 2), ('B', 2), ('C', 6), ('D', 4)] for i in range(count)])
        self.assertTrue(all(kwargs['project_id'] == 'project' for _, kwargs in client.retrieved))
        self.assertTrue(all('INPUTS' in kwargs['selects'] for _, kwargs in client.retrieved))
        self.assertNotIn('INPUTS', FIELDS)


if __name__ == '__main__':
    unittest.main()
