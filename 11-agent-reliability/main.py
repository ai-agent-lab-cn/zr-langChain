"""第11章离线运行入口：python 11-agent-reliability/main.py --self-test。

仅支持本地替身，无法通过参数开启 DeepSeek 或 LangSmith 服务调用。
成功、故障和未运行项均保存真实状态；输出不含用户/工具原文。


配套流程图：同目录 basic-knowledge.md，入口与场景分派（entry）。
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import uuid

from langsmith import tracing_context
from scenarios import FIXTURE_VERSION, run_case


def source_fingerprint() -> str:
    """哈希实际运行的五个模块，给未提交的源码生成可核对的版本标识。

    返回 SHA-256 十六进制字符串。文件名后加入空字节分隔，避免拼接歧义。
    此处哈希原始文件字节，注释变化也会改变指纹；历史证据不能回写新指纹。
    这不是 Git commit，也不包含依赖版本，依赖在 report.versions 中另记。"""
    digest = hashlib.sha256()
    for name in ['main.py', 'retry_policy.py', 'safety.py', 'scenarios.py', 'trace_audit.py']:
        digest.update(name.encode() + b'\0')
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def _write_progress(stream, report: dict) -> None:
    """将本次报告写入已独占创建的文件；stream=None 表示仅在终端输出。

    seek(0) 覆盖本次快照，truncate() 清除较长旧快照的尾部，flush() 刷新缓冲。
    这里只重写本批文件，从不重新打开其他批次；它不等同于原子落盘或 fsync。
    强制中断仍可能留下 RUNNING 或不完整 JSON，读取者必须检查最终状态。"""
    if stream is not None:
        stream.seek(0)
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.truncate()
        stream.flush()


def main(argv=None) -> int:
    """顺序运行离线 A—D，并用终态和退出码向调用方报告结果。

    参数 argv 为命令行字符串列表；None 时由 argparse 读取实际命令行。
    --self-test 固定运行四组；--case 只选组；--offline 仅显式声明已有默认。
    返回 0 表示所选场景断言全部通过，1 表示运行/证据失败；参数错误由
    argparse 以 2 退出。业务中预期的 FAILED 不等于场景验收 FAILED。

    顺序：解析参数 → 独占证据路径 → 初始化 RUNNING → run_case → 写结果。
    某组抛异常后不再运行后续组，而是标 NOT_RUN；finally 始终关闭文件。
    本入口不读 .env、不创建在线客户端，追踪由本次上下文显式关闭。"""
    parser = argparse.ArgumentParser(description='第11章 Agent Reliability 离线教学实验')
    parser.add_argument('--case', choices=['A', 'B', 'C', 'D', 'all'], default='all', help='选择一组实验，默认全部')
    parser.add_argument('--offline', action='store_true', help='显式声明离线；本章始终离线')
    parser.add_argument('--self-test', action='store_true', help='运行并验收 A—D 全部场景')
    parser.add_argument('--evidence', type=Path, help='保存脱敏证据；已存在时在执行前拒绝')
    args = parser.parse_args(argv)
    # stream 只属于本次批次；未指定 --evidence 时所有快照写入均为空操作。
    stream = None
    try:
        if args.evidence:
            args.evidence.parent.mkdir(parents=True, exist_ok=True)
            # x 模式原子占用路径；不能先跑效果后才检查文件是否重名。
            stream = args.evidence.open('x', encoding='utf-8')
        # 报告分两层：批次状态控制退出码，records 保留每组的独立验收结果。
        report = {'schema_version': 1, 'batch_id': str(uuid.uuid4()), 'mode': 'offline',
                  'status': 'RUNNING', 'records': [], 'trace_source': 'local_fixture_only',
                  'fixture_version': FIXTURE_VERSION, 'source_sha256': source_fingerprint(),
                  'versions': {name: version(name) for name in ['langchain', 'langgraph', 'langsmith']}}
        _write_progress(stream, report)
        cases = 'ABCD' if args.self_test or args.case == 'all' else args.case
        failed = False
        with tracing_context(enabled=False):
            for case in cases:
                # failed 是批次停止标记；预期的业务失败由场景内部断言判断，不设置它。
                if failed:
                    report['records'].append({'case': case, 'status': 'NOT_RUN'})
                    continue
                try:
                    record = run_case(case)
                except Exception as exc:
                    # 原始异常可能携带秘密；只保存安全类型名，不把正文送入日志。
                    record = {'case': case, 'mode': 'offline', 'status': 'FAILED',
                              'error_type': type(exc).__name__, 'provider_calls': 0, 'trace_requested': False}
                    failed = True
                report['records'].append(record)
                _write_progress(stream, report)
        # 收尾会把 NOT_RUN 也写入本次证据，不能把缺失记录误算为通过。
        report['status'] = 'FAILED' if failed else 'PASS'
        _write_progress(stream, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if failed else 0
    except Exception as exc:
        print(json.dumps({'status': 'FAILED', 'error_type': type(exc).__name__}, ensure_ascii=False))
        return 1
    finally:
        if stream is not None:
            stream.close()


if __name__ == '__main__':
    raise SystemExit(main())
