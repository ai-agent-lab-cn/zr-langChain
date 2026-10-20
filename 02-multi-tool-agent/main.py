"""Chapter 02: Multi-Tool Agent — Gemini + LangChain + LangSmith.

Run from the zrLangChain repository root:
    uv run python 02-multi-tool-agent/main.py
    uv run python 02-multi-tool-agent/main.py --case all
"""

import argparse
import math
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_google_genai import ChatGoogleGenerativeAI


# 读取仓库根目录配置；无论在哪个工作目录执行，都能找到 .env。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


@tool
def get_weather(city: str) -> str:
    """Get simulated weather for a city; this is mock data, not a live forecast."""
    return f"Simulated weather for {city}: sunny (mock data; not real-time)."


@tool
def calculate(
    a: float,
    b: float,
    operation: Literal["add", "subtract", "multiply", "divide"],
) -> str:
    """Calculate one arithmetic step with two numbers: add, subtract, multiply, or divide."""
    if not all(math.isfinite(x) and abs(x) <= 1e12 for x in (a, b)):
        return "Error: inputs must be finite numbers with absolute values <= 1e12."

    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return "Error: division by zero is not allowed."
        result = a / b
    else:
        return f"Error: unsupported operation {operation}."

    if not math.isfinite(result):
        return "Error: result is not finite."
    return f"{result:.15g}"


@tool
def get_current_time(timezone_name: str) -> str:
    """Get the current local time in an IANA timezone, e.g. Asia/Tokyo."""
    try:
        timezone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        return f"Error: unknown IANA timezone '{timezone_name}'."

    return datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S %Z (UTC%z)")


# 每个场景预期的工具执行次数（严格匹配）；模型实际行为可能偏离预期。
SCENARIOS = {
    "weather": {
        "question": "请使用工具查询旧金山的天气，明确告诉我这是否是真实天气数据。",
        "expected": {"get_weather": 1},
    },
    "math": {
        "question": "请使用 calculate 工具分两步计算：(125 × 8) + 360。每步都调用工具，不要心算。",
        "expected": {"calculate": 2},
    },
    "time": {
        "question": "请通过工具查询东京现在的当地时间，使用 Asia/Tokyo 时区。",
        "expected": {"get_current_time": 1},
    },
    "combined": {
        "question": (
            "完成三个任务：① 查询旧金山的模拟天气；② 查询东京当前时间（Asia/Tokyo）；"
            "③ 分两步用 calculate 计算 (125 × 8) + 360。"
            "请调用对应工具完成每项任务，不要心算，并明确指出天气是模拟数据。"
        ),
        "expected": {"get_weather": 1, "get_current_time": 1, "calculate": 2},
    },
}


def build_agent():
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY 未配置；请检查仓库根目录 .env。")

    # 显式传入 GEMINI_API_KEY，避免其他环境变量影响密钥选择。
    model = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        api_key=gemini_api_key,
    )

    return create_agent(
        model=model,
        tools=[get_weather, calculate, get_current_time],
        system_prompt=(
            "You are a careful assistant that uses tools for verifiable answers. "
            "Use get_weather for weather questions and always label the result as simulated. "
            "Use calculate for EACH arithmetic operation; never do arithmetic mentally. "
            "Use get_current_time for questions about current time, with a valid IANA timezone. "
            "For multi-part questions, use all relevant tools and combine the results. "
            "Do not invent tool outputs. If a tool reports an error, state the error. "
            "Answer the user in Chinese."
        ),
    )


def _final_text(message) -> str:
    """Extract readable text from the final AIMessage."""
    if not isinstance(message, AIMessage):
        return ""
    return "\n".join(
        block.get("text", "")
        for block in message.content_blocks
        if isinstance(block, dict) and block.get("type") == "text"
    )


def verify_trace(messages, expected: dict[str, int], case_name: str) -> list[str]:
    """Return verification errors; an empty list means this demo case passed."""
    errors: list[str] = []
    requested = Counter()
    executed = Counter()
    requests_by_id = {}
    results_by_id = {}

    for index, message in enumerate(messages):
        if isinstance(message, AIMessage):
            if message.invalid_tool_calls:
                errors.append(f"第 {index + 1} 步包含无法解析的工具调用")
            for call in message.tool_calls:
                name, call_id = call.get("name"), call.get("id")
                requested[name] += 1
                if not isinstance(call.get("args"), dict):
                    errors.append(f"工具 {name} 的参数不是字典")
                if not call_id:
                    errors.append(f"工具 {name} 缺少 Tool Call ID")
                elif call_id in requests_by_id:
                    errors.append(f"Tool Call ID 重复请求: {call_id}")
                else:
                    requests_by_id[call_id] = (call, index)

        elif isinstance(message, ToolMessage):
            name = message.name or "unknown"
            call_id = message.tool_call_id
            executed[name] += 1
            if not call_id:
                errors.append(f"工具 {name} 的返回缺少 Tool Call ID")
            elif call_id in results_by_id:
                errors.append(f"Tool Call ID 重复返回: {call_id}")
            else:
                results_by_id[call_id] = (message, index)

            if message.status != "success":
                errors.append(f"工具 {name} 执行失败: status={message.status}")
            if not isinstance(message.content, str) or not message.content.strip():
                errors.append(f"工具 {name} 未返回非空文本结果")
            elif message.content.lstrip().lower().startswith("error:"):
                errors.append(f"工具 {name} 返回错误: {message.content}")

    if requested != Counter(expected):
        errors.append(f"模型请求次数不匹配: 实际 {dict(requested)}，预期 {expected}")
    if executed != Counter(expected):
        errors.append(f"工具执行次数不匹配: 实际 {dict(executed)}，预期 {expected}")

    for call_id, (call, request_index) in requests_by_id.items():
        match = results_by_id.get(call_id)
        if match is None:
            errors.append(f"工具请求没有对应返回: {call_id}")
        else:
            result, result_index = match
            if result.name != call["name"]:
                errors.append(f"工具名称不匹配: {call_id}，请求 {call['name']}，返回 {result.name}")
            if result_index <= request_index:
                errors.append(f"工具返回早于对应请求: {call_id}")
    for call_id in results_by_id.keys() - requests_by_id.keys():
        errors.append(f"存在没有对应请求的工具返回: {call_id}")

    # 本章四个实验的确定性参数/结果校验；不调用真实天气 API。
    for call_id, (call, _) in requests_by_id.items():
        name = call["name"]
        args = call.get("args")
        result = results_by_id.get(call_id)
        if not isinstance(args, dict) or result is None:
            continue
        output = result[0].content
        if name == "get_weather":
            if not isinstance(args.get("city"), str) or args["city"].strip().lower() != "san francisco":
                errors.append(f"天气工具参数错误: {args}")
            if output != "Simulated weather for San Francisco: sunny (mock data; not real-time).":
                errors.append(f"天气工具结果不符预期: {output}")
        elif name == "get_current_time":
            if args.get("timezone_name") != "Asia/Tokyo":
                errors.append(f"时区工具参数错误: {args}")
            if not isinstance(output, str) or not re.fullmatch(
                r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} JST \(UTC\+0900\)", output
            ):
                errors.append(f"时区工具结果格式异常: {output}")
            else:
                try:
                    datetime.strptime(output, "%Y-%m-%d %H:%M:%S JST (UTC+0900)")
                except ValueError:
                    errors.append(f"时区工具日期时间无效: {output}")

    math_calls = [
        (call, request_index, results_by_id.get(call_id))
        for call_id, (call, request_index) in requests_by_id.items()
        if call["name"] == "calculate"
    ]
    math_calls.sort(key=lambda item: item[1])
    if case_name in ("math", "combined"):
        steps = [(125, 8, "multiply", "1000"), (1000, 360, "add", "1360")]
        for i, (a, b, operation, value) in enumerate(steps):
            if i >= len(math_calls):
                break  # 工具次数错误已由 Counter 校验。
            call, _, response = math_calls[i]
            args = call.get("args")
            if not isinstance(args, dict) or (
                args.get("a") != a or args.get("b") != b or args.get("operation") != operation
            ):
                errors.append(f"第 {i + 1} 步计算参数错误: {args}")
            if response is not None and response[0].content != value:
                errors.append(f"第 {i + 1} 步计算结果错误: {response[0].content}（预期 {value}）")
        if len(math_calls) >= 2 and math_calls[0][2] is not None:
            if math_calls[1][1] <= math_calls[0][2][1]:
                errors.append("第二次计算请求没有等待第一次工具结果")

    final = _final_text(messages[-1]) if messages else ""
    if not final.strip():
        errors.append("缺少最终 AI 文本回答")
    if case_name in ("weather", "combined") and not (
        ("模拟" in final or "mock" in final.lower())
        and ("非实时" in final or "不是真实" in final or "不是实时" in final or "not real-time" in final.lower())
    ):
        errors.append("最终回答未明确标注天气为模拟、非实时数据")
    if case_name in ("math", "combined") and "1360" not in final:
        errors.append("最终回答缺少计算结果 1360")
    if case_name in ("time", "combined") and "东京" not in final and "Tokyo" not in final:
        errors.append("最终回答没有说明东京时间")
    return errors


def show_trace(messages, expected: dict[str, int], case_name: str) -> bool:
    """Print the execution chain and checks; return True only if all checks pass."""
    executed = Counter()
    requested = Counter()

    print("\n========== Agent 执行链路 ==========")
    for index, message in enumerate(messages, start=1):
        print(f"\nStep {index}: {message.type}")
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                requested[call["name"]] += 1
                print(f"  请求工具: {call['name']}")
                print(f"  调用参数: {call['args']}")
                print(f"  Tool Call ID: {call.get('id', '无')}")
        elif isinstance(message, ToolMessage):
            tool_name = message.name or "unknown"
            executed[tool_name] += 1
            print(f"  已执行工具: {tool_name}")
            print(f"  工具返回: {message.content}")
            print(f"  Tool Call ID: {message.tool_call_id}")

    print("\n========== 最终回答 ==========")
    if messages:
        print(_final_text(messages[-1]))

    errors = verify_trace(messages, expected, case_name)
    print("\n========== 工具验证 ==========")
    print("模型请求:", dict(requested))
    print("实际执行:", dict(executed))
    print("实验预期:", expected)
    if errors:
        print("验证结果: ❌ 未通过")
        for error in errors:
            print(f"  - {error}")
    else:
        print("验证结果: ✅ 全部检查通过（次数 / ID / 状态 / 参数 / 结果）")
    return not errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 02 Multi-Tool Agent")
    parser.add_argument(
        "--case",
        choices=[*SCENARIOS, "all"],
        default="combined",
        help="要运行的实验场景（默认 combined；all 会产生多次 API 调用）",
    )
    args = parser.parse_args()

    agent = build_agent()
    cases = list(SCENARIOS) if args.case == "all" else [args.case]

    failed_cases = []
    try:
        for case_name in cases:
            scenario = SCENARIOS[case_name]
            print(f"\n{'=' * 16} {case_name} {'=' * 16}")
            print("用户问题:", scenario["question"])
            result = agent.invoke(
                {"messages": [{"role": "user", "content": scenario["question"]}]},
                config={
                    "run_name": f"chapter02-{case_name}",
                    "tags": ["zrLangChain", "chapter-02", case_name],
                    "metadata": {"chapter": "02-multi-tool-agent", "scenario": case_name},
                },
            )
            if not show_trace(result["messages"], scenario["expected"], case_name):
                failed_cases.append(case_name)
    finally:
        # 结束前等待 LangSmith 上传，减少短进程丢失 Trace 的可能。
        wait_for_all_tracers()

    if failed_cases:
        raise SystemExit(f"验证失败的实验: {', '.join(failed_cases)}")


if __name__ == "__main__":
    main()
