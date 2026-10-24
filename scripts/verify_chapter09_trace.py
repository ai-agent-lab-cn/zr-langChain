"""第09章：只读验收已知 run_id，不发送任何模型请求，不导出原始对话。"""
import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client


def audit(root, runs, case):
    data = root.outputs
    expected = ["supervisor", "account_agent", "supervisor"] if case == "supervisor" else ["triage", "logistics_agent"]
    assert data['steps'] == expected
    assert data['active_agent'] == ("supervisor" if case == "supervisor" else "logistics_agent")
    pending = {}
    pairs = []
    for message in data['messages']:
        if message['type'] == 'ai':
            assert not pending, '模型收到未配对工具调用'
            for call in message.get('tool_calls', []):
                pending[call['id']] = call['name']
        elif message['type'] == 'tool':
            call_id = message['tool_call_id']
            assert pending.pop(call_id) == message['name']
            pairs.append({'tool': message['name'], 'tool_call_id': call_id, 'matched': True})
    assert not pending and len(pairs) == 1
    assert data['messages'][-1]['type'] == 'ai' and data['messages'][-1]['content']
    assert runs and all(not r.error and r.end_time for r in runs)
    by_id = {str(r.id): r for r in runs}
    assert str(root.id) in by_id
    assert all(str(r.trace_id) == str(root.id) for r in runs)
    assert all(str(r.parent_run_id) in by_id for r in runs if r.parent_run_id)
    llms = [r for r in runs if r.run_type == 'llm']
    assert len(llms) == (3 if case == 'supervisor' else 2)
    if case == 'supervisor':
        tool = next(r for r in runs if r.run_type == 'tool' and r.name == 'account_agent')
        # 专员 LLM 必须是工具 run 的后代，而不是仅根据名字推断委派成功。
        def below(run):
            while run.parent_run_id:
                if str(run.parent_run_id) == str(tool.id):
                    return True
                run = by_id[str(run.parent_run_id)]
            return False
        assert any(below(r) for r in llms)
    assert all(r.total_tokens is not None and r.prompt_tokens is not None and r.completion_tokens is not None for r in llms)
    return {'case': case, 'run_id': str(root.id), 'trace_url': root.url,
            'status': 'PASS', 'steps': data['steps'], 'active_agent': data['active_agent'],
            'tool_pairs': pairs, 'run_count': len(runs), 'llm_calls': len(llms),
            'tokens': {'input': sum(r.prompt_tokens for r in llms),
                       'output': sum(r.completion_tokens for r in llms),
                       'total': sum(r.total_tokens for r in llms)},
            'runs': [{'id': str(r.id), 'parent_id': str(r.parent_run_id) if r.parent_run_id else None,
                      'name': r.name, 'type': r.run_type} for r in runs]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence'); p.add_argument('output')
    args = p.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    client = Client()
    result = []
    for item in json.loads(Path(args.evidence).read_text()):
        assert item['offline'] is False, '不接受离线证据'
        root = client.read_run(item['run_id'])
        runs = list(client.list_runs(trace_id=item['run_id'], limit=100))
        result.append(audit(root, runs, item['case']))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for item in result:
        print(item['case'], item['status'], 'LLM calls:', item['llm_calls'], 'tokens:', item['tokens'])


if __name__ == '__main__':
    main()
