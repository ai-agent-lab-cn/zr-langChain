"""Chapter 02: Multi-Tool Agent — Gemini + LangChain + LangSmith.

Run from the zrLangChain repository root:
    uv run python 02-multi-tool-agent/main.py
    uv run python 02-multi-tool-agent/main.py --case all
"""

import argparse
import math
import os
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


# 验证要求：各工具至少实际执行多少次。
# 这些是实验预期，并非对大模型行为的保证。
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


def show_trace(messages, expected: dict[str, int]) -> None:
    """Display model tool requests, actual tool execution, and verification summary."""
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
    for block in messages[-1].content_blocks:
        if block.get("type") == "text":
            print(block["text"])

    print("\n========== 工具验证 ==========")
    print("模型请求:", dict(requested))
    print("实际执行:", dict(executed))
    print("实验预期:", expected)
    passed = all(executed.get(name, 0) >= count for name, count in expected.items())
    print("验证结果:", "✅ 符合预期" if passed else "⚠️ 未达到预期，请检查 Trace")


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
            show_trace(result["messages"], scenario["expected"])
    finally:
        # 结束前等待 LangSmith 上传，减少短进程丢失 Trace 的可能。
        wait_for_all_tracers()


if __name__ == "__main__":
    main()
