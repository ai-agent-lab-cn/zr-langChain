"""新 SDK 响应、分页边界与完整性测试；不请求网络或模型。"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import langsmith_reader as reader


def run(rid, typ='CHAIN', parents=None, **kw):
    return NS(id=rid, trace_id='root', parent_run_ids=parents or [], name=rid,
              run_type=typ, **kw)


class FakeClient:
    def __init__(self, items):
        self.items = items
        self.traces = self
        self.runs = self
        self.query_args = None
        self.trace_args = None

    async def aread_project(self, *, project_name):
        return NS(id='project')

    async def list_runs(self, trace_id, **kwargs):
        self.trace_args = kwargs
        return NS(items=self.items)

    def query(self, **kwargs):
        self.query_args = kwargs
        async def rows():
            for item in self.items:
                yield item
        return rows()


class ReaderTests(unittest.IsolatedAsyncioTestCase):
    def test_normalizes_new_sdk_fields_without_losing_missing_tokens(self):
        raw = run('llm', 'LLM', ['root', 'tool'], prompt_tokens=10,
                  completion_tokens=2, total_tokens=12,
                  completion_token_details=NS(raw={'reasoning': 0}), metadata={'chapter': '09'})
        row = reader.normalize_run(raw)
        self.assertEqual(row.run_type, 'llm')
        self.assertEqual(row.parent_run_id, 'tool')
        self.assertEqual(row.completion_token_details, {'reasoning': 0})
        self.assertEqual(row.total_tokens, 12)
        self.assertIsNone(reader.normalize_run(run('unknown')).total_tokens)

    async def test_exact_trace_cap_is_complete_and_root_is_found(self):
        client = FakeClient([run('root'), run('llm', 'LLM', ['root'])])
        root, rows = await reader.fetch_trace(client, project_id='project', trace_id='root', max_spans=2)
        self.assertEqual(root.id, 'root')
        self.assertEqual(len(rows), 2)
        self.assertIn('OUTPUTS', client.trace_args['selects'])
        self.assertIn('PARENT_RUN_IDS', client.trace_args['selects'])
        with self.assertRaises(reader.TraceReadLimitExceeded):
            await reader.fetch_trace(client, project_id='project', trace_id='root', max_spans=1)

    async def test_missing_root_or_ancestor_cannot_pass(self):
        for items in ([run('llm', 'LLM', ['root'])], [run('root'), run('llm', 'LLM', ['root', 'missing'])]):
            with self.subTest(items=items), self.assertRaises(ValueError):
                await reader.fetch_trace(FakeClient(items), project_id='project', trace_id='root')

    async def test_root_query_consumes_async_pages_and_enforces_cap(self):
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)
        client = FakeClient([run('root'), run('other'), run('third')])
        rows, limited = await reader.fetch_roots(client, project_id='project', since=since, max_roots=2)
        self.assertTrue(limited)
        self.assertEqual([r.id for r in rows], ['root', 'other'])
        self.assertEqual(client.query_args['min_start_time'], since)
        self.assertEqual(client.query_args['project_ids'], ['project'])
        self.assertNotIn('limit', client.query_args)
