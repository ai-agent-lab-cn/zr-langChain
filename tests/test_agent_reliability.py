"""第11章集成测试：真实图、离线入口、错误退出及证据防覆盖。"""
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

CHAPTER = Path(__file__).resolve().parents[1] / '11-agent-reliability'
sys.path.insert(0, str(CHAPTER))
import scenarios
from langchain_core.messages import AIMessage, ToolMessage


class ScenarioTests(unittest.TestCase):
    """断言计数和最终状态，错误包装不能把业务失败变为成功。"""

    def test_a_actual_retries_timeout_and_virtual_budget(self):
        """可控退避与真实节点超时分开验收，避免拿虚拟时钟代替取消。"""
        result = scenarios.run_case('A')
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['observed']['retry']['requests'], 3)
        self.assertEqual(result['observed']['retry']['waits'], [1, 2])
        self.assertEqual(result['observed']['deadline']['requests'], 2)
        self.assertEqual(result['observed']['graph_attempts'], [1, 2, 3])
        self.assertEqual(result['observed']['timeout_attempts'], 1)

    def test_b_checkpoint_recovery_does_not_repeat_successful_sibling_or_effect(self):
        """恢复必须沿真实 Checkpoint 续接，不重跑已经完成的并行分支。"""
        result = scenarios.run_case('B')
        self.assertEqual(result['observed']['parallel_calls'], {'good': 1, 'flaky': 2})
        self.assertEqual(result['observed']['write_attempts'], 2)
        self.assertEqual(result['observed']['effects'], 1)
        self.assertTrue(all(result['checks'].values()))

    def test_c_approval_and_injection_keep_effects_bounded(self):
        """恶意资料不扩权；拒绝和过期不能造成真实假服务写入。"""
        result = scenarios.run_case('C')
        self.assertEqual(result['observed']['approved_effects'], 1)
        self.assertEqual(result['observed']['denied_effects'], 0)
        self.assertTrue(result['checks']['untrusted_fields_removed'])
        self.assertTrue(result['checks']['checkpoint_has_no_secret'])

    def test_d_required_optional_failures_and_bounded_handoff(self):
        """必要专家失败须失败，可选专家失败只能得到明确降级。"""
        result = scenarios.run_case('D')
        self.assertEqual(result['observed']['required_status'], 'FAILED')
        self.assertEqual(result['observed']['optional_status'], 'DEGRADED')
        self.assertEqual(result['observed']['handoffs'], 3)
        self.assertEqual(result['observed']['model_calls'], 4)
        self.assertEqual(result['observed']['fixture_tokens'], 12)
        self.assertEqual(result['observed']['incomplete_fixture'], 'REVIEW')

    def test_handoff_model_budget_and_invalid_target_stop_before_transition(self):
        """局部转交限制不能盖过全局模型预算，未知目标不能被调度。"""
        result = scenarios.handoff_demo(max_model_calls=2)
        self.assertEqual(result['model_calls'], 2)
        self.assertEqual(result['handoffs'], 2)
        self.assertEqual(result['reason'], 'MODEL_BUDGET')
        invalid = scenarios.handoff_demo(routes={'A': 'admin', 'B': 'A'})
        self.assertEqual(invalid['reason'], 'INVALID_TARGET')
        self.assertEqual(invalid['handoffs'], 0)

    def test_direct_graph_helpers_disable_inherited_tracing(self):
        """公开图示例独立保证离线，不能只依赖 CLI 或 run_case 外层上下文。"""
        import asyncio
        from langsmith import tracing_context
        from langsmith.run_trees import RunTree
        with patch.object(RunTree, 'post') as post, patch.object(RunTree, 'patch') as update, \
             patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden')), \
             patch.object(socket, 'getaddrinfo', side_effect=AssertionError('dns forbidden')), \
             tracing_context(enabled=True):
            scenarios.graph_retry_demo()
            asyncio.run(scenarios.node_timeout_demo())
            scenarios.parallel_recovery_demo()
            scenarios.write_recovery_demo()
            scenarios.handoff_demo(max_model_calls=1)
            scenarios.expert_failure_demo(required=True)
        self.assertEqual(post.call_count, 0)
        self.assertEqual(update.call_count, 0)

    def test_protocol_rejects_orphan_duplicate_and_missing_reply(self):
        """ToolMessage 必须与一次真实 tool_call_id 配对。"""
        call = AIMessage(content='', tool_calls=[{'name': 'x', 'args': {}, 'id': 'c1'}])
        reply = ToolMessage(content='ok', tool_call_id='c1')
        self.assertEqual(scenarios.paired_tools([call, reply]), 1)
        for messages in [[reply], [call], [call, reply, reply]]:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                scenarios.paired_tools(messages)


class CLITests(unittest.TestCase):
    """CLI 测试只运行本章入口，不生成任何提供商请求。"""

    @classmethod
    def setUpClass(cls):
        """使用唯一模块名加载相邻入口，避免与其他章 main.py 混淆。"""
        spec = importlib.util.spec_from_file_location('chapter11_cli', Path(scenarios.__file__).with_name('main.py'))
        cls.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cli)

    def test_self_test_forbids_network_and_env_file_reads_even_with_trace_env(self):
        """继承开启追踪的环境变量，也必须保持离线且不读 .env。"""
        original_open = io.open
        def guarded_open(file, *args, **kwargs):
            """只拒绝凭据文件，保留真实证据文件和 Python 读取行为。"""
            if isinstance(file, (str, Path)) and Path(file).name == '.env':
                raise AssertionError('offline read .env')
            return original_open(file, *args, **kwargs)
        with patch.dict(os.environ, {'LANGSMITH_TRACING': 'true', 'LANGSMITH_API_KEY': 'FAKE_SECRET_CH11'}), \
             patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden')), \
             patch('io.open', side_effect=guarded_open), redirect_stdout(io.StringIO()) as out:
            code = self.cli.main(['--self-test'])
        self.assertEqual(code, 0)
        records = json.loads(out.getvalue())['records']
        self.assertEqual([r['status'] for r in records], ['PASS'] * 4)
        self.assertTrue(all(r['provider_calls'] == 0 and not r['trace_requested'] for r in records))
        self.assertNotIn('FAKE_SECRET_CH11', out.getvalue())

    def test_evidence_existing_path_prevents_execution_and_preserves_contents(self):
        """在模型或工具运行前独占创建证据，不能覆盖历史或先执行再发现重名。"""
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp) / 'evidence.json'
            evidence.write_text('keep')
            with patch.object(self.cli, 'run_case', side_effect=AssertionError('must not execute')), redirect_stdout(io.StringIO()):
                code = self.cli.main(['--case', 'A', '--evidence', str(evidence)])
            self.assertEqual(code, 1)
            self.assertEqual(evidence.read_text(), 'keep')

    def test_failed_case_is_saved_remaining_not_run_and_exception_redacted(self):
        """运行异常必须对应非零退出与 FAILED，原始错误不能进入日志或证据。"""
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp) / 'evidence.json'
            with patch.object(self.cli, 'run_case', side_effect=ValueError('FAKE_SECRET_CH11')), redirect_stdout(io.StringIO()) as out:
                code = self.cli.main(['--self-test', '--evidence', str(evidence)])
            data = json.loads(evidence.read_text())
        self.assertEqual(code, 1)
        self.assertEqual([r['status'] for r in data['records']], ['FAILED', 'NOT_RUN', 'NOT_RUN', 'NOT_RUN'])
        self.assertNotIn('FAKE_SECRET_CH11', out.getvalue())
        self.assertNotIn('FAKE_SECRET_CH11', json.dumps(data))

    def test_unknown_case_is_rejected_without_silent_pass(self):
        """场景拼写错误不能产生空 checks 的假 PASS。"""
        with self.assertRaises(ValueError):
            scenarios.run_case('Z')


if __name__ == '__main__':
    unittest.main()
