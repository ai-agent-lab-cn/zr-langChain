"""第10章只读在线验收：精确模型ID定位完整Trace，原始输入输出仅留内存。"""
import argparse
import asyncio
import json
import logging
import os
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langsmith_reader import FIELDS, fetch_trace, field, normalize_run
ROOT = Path(__file__).resolve().parents[1]

class AuditFailure(RuntimeError):
    pass

def check(condition, label):
    """使用固定错误代码报告验收失败，不拼接服务端原文或凭据。"""
    if not condition:
        raise AuditFailure(label)
    return True

def message(value):
    """兼容普通消息与 LangChain constructor 序列化，保留工具名称及调用 ID。"""
    check(isinstance(value, dict), 'message_not_dict')
    original = value
    value = value.get('kwargs', value)
    kind = value.get('type') or value.get('role')
    if kind == 'constructor' or kind is None:
        kind = {'AIMessage': 'ai', 'HumanMessage': 'human', 'ToolMessage': 'tool', 'SystemMessage': 'system'}.get((original.get('id') or [''])[-1])
    return {**value, 'type': {'assistant': 'ai', 'user': 'human'}.get(kind, kind)}

def model_messages(run):
    """模型输入可能是单组或批次列表；统一为多组消息供协议检查。"""
    batches = run.inputs['messages']
    if batches and isinstance(batches[0], dict):
        batches = [batches]
    return [[message(value) for value in batch] for batch in batches]

def root_messages(run):
    """从根 Graph 的输出读取实际消息状态，不根据工具 run 名字推测结果。"""
    return [message(value) for value in run.outputs['messages']]

def protocol(messages):
    """检查完整消息交互，拒绝孤立、重复、悬空或名称不匹配的工具结果。"""
    pending = {}
    seen = set()
    pairs = []
    for item in messages:
        if item['type'] == 'tool':
            call_id = item.get('tool_call_id')
            check(call_id in pending, 'orphan_tool_result')
            name = pending.pop(call_id)
            check(item.get('name') == name, 'tool_name_mismatch')
            pairs.append({'tool': name, 'tool_call_id': call_id, 'matched': True})
        else:
            check(not pending, 'tool_results_incomplete_before_next_message')
            if item['type'] == 'ai':
                for call in item.get('tool_calls', []):
                    check(call.get('id') and call['id'] not in seen, 'duplicate_tool_call_id')
                    pending[call['id']] = call['name']
                    seen.add(call['id'])
    check(not pending, 'tool_results_incomplete_at_end')
    return pairs

def serialized(value):
    return json.dumps(value, ensure_ascii=False, default=str)

def order_summary(messages):
    """校验真实标记摘要中的事实类型与更正关系；字段关键词不能替代结构化内容。"""
    summaries = [item for item in messages
                 if item.get('additional_kwargs', {}).get('lc_source') == 'summarization']
    check(len(summaries) == 1, 'B_summary_state_missing')
    try:
        facts = json.loads(str(summaries[0]['content']).split('\n\n', 1)[-1])
    except (ValueError, TypeError, KeyError):
        raise AuditFailure('B_summary_json_invalid') from None
    check(isinstance(facts, dict), 'B_summary_not_object')
    superseded = facts.get('superseded_order_ids')
    constraints = facts.get('constraints')
    check(isinstance(superseded, list) and all(isinstance(x, str) for x in superseded),
          'B_superseded_must_be_array')
    check(isinstance(constraints, list) and all(isinstance(x, str) for x in constraints),
          'B_constraints_must_be_array')
    check(facts.get('active_order_id') == 'B200' and 'A100' in superseded and 'B200' not in superseded,
          'B_summary_fact_conflict')
    return facts


def verify_summary_delivery(state_messages, model_batches):
    """确认主模型实际收到与 State 中一致的标记摘要，含更正事实和约束。"""
    expected = order_summary(state_messages)
    check(len(model_batches) == 1, 'B_unexpected_model_input_batches')
    observed = order_summary(model_batches[0])
    return check(expected == observed, 'B_summary_changed_before_model')


def tokens(runs):
    """仅统计唯一叶子 LLM；不叠加 Graph / Tool 的用量或 reasoning/cache 明细。"""
    llms = [r for r in runs if r.run_type == 'llm']
    model_ids = {str(r.id) for r in llms}
    check(len(model_ids) == len(llms), 'duplicate_model_run_id')
    check(not any(model_ids.intersection(r.parent_run_ids) for r in llms), 'nested_models_would_double_count')
    for r in llms:
        check(all((type(x) is int and x >= 0 for x in [r.prompt_tokens, r.completion_tokens, r.total_tokens])), 'missing_server_tokens')
        check(r.total_tokens == r.prompt_tokens + r.completion_tokens, 'server_token_total_mismatch')
    return {'input': sum((r.prompt_tokens for r in llms)), 'output': sum((r.completion_tokens for r in llms)), 'total': sum((r.total_tokens for r in llms))}

async def verify(client, evidence, *, project, max_spans=250):
    """按精确模型 ID 找回完整 Trace；原始输入输出只在内存中分析。"""
    ids = validate_evidence(evidence)
    project = await client.aread_project(project_name=project)
    # 仅对本次精确 ID 查询额外 INPUTS，不修改共享适配层的默认隐私字段。
    found = {}
    # retrieve 直接按精确 ID 读取，避免 query 默认时间窗口或过宽时间跨度限制。
    for run_id in ids:
        raw = await client.runs.retrieve(run_id, project_id=str(project.id), selects=[*FIELDS, 'INPUTS'])
        normalized = normalize_run(raw)
        normalized.inputs = field(raw, 'inputs')
        found[str(raw.id)] = normalized
    check(set(ids) == set(found), 'server_missing_observed_model_ids')
    # 复用现有适配层检查缺根、缺父、重复 ID、自引用与跨 Trace 污染。
    traces = {}
    for trace_id in dict.fromkeys((str(run.trace_id) for run in found.values())):
        traces[trace_id] = await fetch_trace(client, project_id=str(project.id), trace_id=trace_id, max_spans=max_spans)
    # 替换成精确查询的模型对象，以取得输入；其他 span 仍使用完整调用树。
    records = []
    for item in evidence:
        case = item['case']
        model_ids = item['telemetry']['observed_model_run_ids']
        roots = list(dict.fromkeys((str(found[i].trace_id) for i in model_ids)))
        spans = [found.get(str(r.id), r) for tid in roots for r in traces[tid][1]]
        llms = [r for r in spans if r.run_type == 'llm']
        check({str(r.id) for r in llms} == set(model_ids), case + '_model_call_set_mismatch')
        check(all((not r.error and r.end_time and (r.status == 'success') for r in spans)), case + '_unfinished_or_failed_span')
        check(len(roots) == {'A': 2, 'B': 1, 'C': 3, 'D': 1}[case], case + '_unexpected_root_count')
        check(len(llms) == {'A': 2, 'B': 2, 'C': 6, 'D': 4}[case], case + '_unexpected_model_count')
        remote_usage = Counter(((r.prompt_tokens, r.completion_tokens, r.total_tokens) for r in llms))
        local_usage = Counter(((r['input_tokens'], r['output_tokens'], r['total_tokens']) for r in item['telemetry']['usage_metadata']))
        check(remote_usage == local_usage, case + '_local_server_usage_mismatch')
        pairs = []
        for tid in roots:
            pairs.extend(protocol(root_messages(traces[tid][0])))
        for r in llms:
            for batch in model_messages(r):
                protocol(batch)
        details = {'trace_complete': True, 'all_spans_success': True, 'usage_matches_local_provider_values': True}
        if case == 'A':
            first = found[model_ids[0]]
            other = found[model_ids[1]]
            details['trimmed_model_context'] = check('旧套餐' not in serialized(first.inputs) and 'A100' in serialized(first.inputs), 'A_trim_not_observed')
            details['checkpoint_kept_history'] = check('旧套餐' in serialized(traces[str(first.trace_id)][0].outputs), 'A_state_history_missing')
            details['other_thread_isolated'] = check('A100' not in serialized(other.inputs), 'A_other_thread_leak')
        elif case == 'B':
            summary_runs = [r for r in spans if r.name == 'summary_once']
            check(len(summary_runs) == 1, 'B_summary_service_missing')
            summary_llms = [r for r in llms if str(summary_runs[0].id) in r.parent_run_ids]
            check(len(summary_llms) == 1, 'B_summary_model_not_descendant')
            messages = root_messages(traces[roots[0]][0])
            order_summary(messages)
            details['summary_corrects_old_fact'] = True
            main_llms = [r for r in llms if str(r.id) != str(summary_llms[0].id)]
            check(len(main_llms) == 1, 'B_main_model_missing')
            details['main_model_receives_summary'] = verify_summary_delivery(messages, model_messages(main_llms[0]))
            details['raw_old_message_removed'] = check(not any((r.get('content') == '旧订单 A100' for r in messages)), 'B_raw_history_not_removed')
        elif case == 'C':
            tool_runs = [r for r in spans if r.run_type == 'tool' and r.name == 'read_preference']
            details['three_real_preference_tools'] = check(len(tool_runs) == 3 and len(pairs) == 3, 'C_missing_tool_runs')
            for (tid, expected) in zip(roots, ['简洁', '简洁', '默认']):
                tool_messages = [r for r in root_messages(traces[tid][0]) if r['type'] == 'tool']
                check(len(tool_messages) == 1 and tool_messages[0]['content'] == expected, 'C_preference_scope_wrong')
            details['cross_thread_and_tenant_boundary'] = True
        elif case == 'D':
            tools = {r.name: r for r in spans if r.run_type == 'tool'}
            check(set(tools) == {'account_expert', 'logistics_expert'} and len(pairs) == 2, 'D_missing_expert_tools')
            expert_inputs = {}
            for name in ['account_expert', 'logistics_expert']:
                descendants = [r for r in llms if str(tools[name].id) in r.parent_run_ids]
                check(len(descendants) == 1, 'D_expert_model_not_tool_descendant')
                expert_inputs[name] = serialized(descendants[0].inputs)
            details['expert_context_isolated'] = check('A100' not in expert_inputs['account_expert'] and '登录失败' not in expert_inputs['logistics_expert'] and all(('PAYMENT_PRIVATE_MARKER' not in value for value in expert_inputs.values())), 'D_expert_context_leak')
            details['tool_fields_filtered'] = check(all(('PRIVATE_PHONE_MARKER' not in serialized(r.inputs) and 'INTERNAL_DEBUG_MARKER' not in serialized(r.inputs) for r in llms)), 'D_tool_field_leak')
            result_tools = [r for r in root_messages(traces[roots[0]][0]) if r['type'] == 'tool']
            for r in result_tools:
                check(set(json.loads(r['content'])) == {'domain', 'advice', 'source', 'truncated'}, 'D_unexpected_expert_output_fields')
            details['both_expert_models_below_tools'] = True
        records.append({'case': case, 'status': 'PASS', 'trace_ids': roots, 'span_count': len(spans), 'llm_calls': len(llms), 'tool_pairs': pairs, 'checks': details, 'tokens': tokens(spans), 'model_runs': [{'id': str(r.id), 'trace_id': str(r.trace_id), 'parent_id': r.parent_run_id, 'input_tokens': r.prompt_tokens, 'output_tokens': r.completion_tokens, 'total_tokens': r.total_tokens, 'completion_token_details': r.completion_token_details} for r in llms], 'call_tree': [{'id': str(r.id), 'parent_id': r.parent_run_id, 'name': r.name, 'type': r.run_type} for r in spans]})
    # 各场景已验证叶子计量，最后按场景求和；避免把父级聚合再次累加。
    totals = {key: sum((r['tokens'][key] for r in records)) for key in ['input', 'output', 'total']}
    result = {'status': 'PASS', 'mode': 'online', 'root_traces': len(traces), 'llm_calls': len(ids), 'span_count': sum((r['span_count'] for r in records)), 'token_counting_rule': 'unique leaf LLM spans only; reasoning/cache detail not added again', 'tokens': totals, 'cases': records, 'raw_inputs_outputs_exported': False, 'all_counted_model_runs_are_leaf_llms': True}
    return result

def validate_evidence(evidence):
    """只接受完整 A—D 的已追踪在线批次，先拒绝部分、离线或计量不全的输入。"""
    check(isinstance(evidence, list) and len(evidence) == 4, 'evidence_requires_abcd')
    check(all(isinstance(row, dict) for row in evidence), 'evidence_row_not_object')
    check([row.get('case') for row in evidence] == list('ABCD'), 'evidence_cases_must_be_unique_abcd')
    expected_calls = {'A': 2, 'B': 2, 'C': 6, 'D': 4}
    ids = []
    for row in evidence:
        check(row.get('mode') == 'online' and row.get('status') == 'PASS'
              and row.get('trace_requested') is True, 'evidence_must_be_successful_traced_online')
        telemetry = row.get('telemetry')
        check(isinstance(telemetry, dict), 'missing_telemetry')
        model_ids = telemetry.get('observed_model_run_ids')
        usage = telemetry.get('usage_metadata')
        count = expected_calls[row['case']]
        check(isinstance(model_ids, list) and len(model_ids) == count
              and all(isinstance(rid, str) and rid for rid in model_ids), 'model_ids_missing_or_wrong_count')
        check(telemetry.get('model_calls') == count and telemetry.get('usage_observed_calls') == count
              and isinstance(usage, list) and len(usage) == count, 'provider_usage_count_mismatch')
        for item in usage:
            check(isinstance(item, dict) and all(type(item.get(key)) is int and item[key] >= 0
                  for key in ['input_tokens', 'output_tokens', 'total_tokens']), 'provider_usage_missing')
            check(item['total_tokens'] == item['input_tokens'] + item['output_tokens'], 'provider_usage_inconsistent')
        ids.extend(model_ids)
    check(len(ids) == len(set(ids)), 'duplicate_observed_model_ids')
    return ids

def main(argv=None):
    """校验路径与证据后才读取本地配置；只请求 LangSmith，不创建模型客户端。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path, help='A—D 全部通过、已请求 Trace 的在线实验 JSON')
    parser.add_argument('output', type=Path, help='新建脱敏审计 JSON，拒绝覆盖')
    parser.add_argument('--project', help='默认读取本地 LANGSMITH_PROJECT')
    parser.add_argument('--max-spans', type=int, default=250, help='每条完整 Trace 的本地分析上限')
    args = parser.parse_args(argv)
    if not args.evidence.is_file():
        parser.error('输入证据文件不存在')
    if args.output.exists():
        parser.error('输出文件已存在，请使用新路径')
    if args.max_spans <= 0:
        parser.error('--max-spans 必须大于零')
    try:
        evidence = json.loads(args.evidence.read_text(encoding='utf-8'))
        validate_evidence(evidence)
        load_dotenv(ROOT / '.env')
        if not os.getenv('LANGSMITH_API_KEY'):
            raise AuditFailure('missing_langsmith_key')
        project = args.project or os.getenv('LANGSMITH_PROJECT') or 'default'
        result = asyncio.run(verify(Client(), evidence, project=project, max_spans=args.max_spans))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, default=str)
            stream.write('\n')
        print(json.dumps({key: result[key] for key in ['status', 'root_traces', 'llm_calls', 'span_count', 'tokens']}, ensure_ascii=False))
        return 0
    except AuditFailure as exc:
        print('AUDIT_FAILED:', str(exc))
        return 1
    except Exception as exc:
        print('AUDIT_ERROR:', type(exc).__name__)
        return 1
if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    raise SystemExit(main())
