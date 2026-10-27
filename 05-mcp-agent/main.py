"""Chapter 05: MCP Agent — first-party LangChain MCPAdapter + FastMCP.

在 zrLangChain 仓库根目录执行：
    uv add 'langchain[mcp]>=1.4.6'
    uv run python 05-mcp-agent/main.py --self-test
    uv run python 05-mcp-agent/main.py --case all

--self-test 使用真实的 MCP stdio 子进程，但不调用 Gemini。
"""

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import tracing_context

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_SCRIPT = Path(__file__).resolve().with_name("server.py")

CASES = {
    "weather": (
        "请务必使用 MCP 工具查询南京的模拟天气，并明确说明数据是模拟的，"
        "不是真实天气。"
    ),
    "math": "请务必使用 MCP 计算工具计算 125 × 8，给出计算结果。",
    "combined": (
        "请分别调用 MCP 的模拟天气工具和计算工具："
        "查询上海的模拟天气、计算 12 + 8，并分别回答，注明天气为模拟数据。"
    ),
}

SYSTEM_PROMPT = (
    "你是一个谨慎的中文 AI 助手。用户要求查天气或做计算时，"
    "必须调用提供的 MCP 工具，不得自行编造工具结果。"
    "本章天气工具返回教学模拟数据，绝不能声称是真实天气。"
    "工具报告不支持城市时，说明支持范围，不编造结果。"
)


def content_text(content: object) -> str:
    """兼容普通文本、模型的 content blocks 和 MCP 工具内容。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    if hasattr(content, "content"):
        return content_text(getattr(content, "content"))
    return str(content)


def tool_names(tools: list) -> set[str]:
    return {tool.name for tool in tools}


def find_tool(tools: list, suffix: str):
    """单 Server 下通常无前缀；也兼容命名空间工具名。"""
    matching = [t for t in tools if t.name == suffix or t.name.endswith("_" + suffix)]
    if len(matching) != 1:
        raise RuntimeError(f"未找到唯一工具 {suffix}；当前工具：{sorted(tool_names(tools))}")
    return matching[0]


async def self_test(tools: list) -> bool:
    """测试真实 stdio MCP 握手/发现/调用；不使用模型、无需 API Key。"""
    with tracing_context(enabled=False):
        return await offline_checks(tools)


async def offline_checks(tools: list) -> bool:
    """在关闭追踪的上下文中调用本地 MCP 工具。"""
    names = tool_names(tools)
    print("发现 MCP Tools:", ", ".join(sorted(names)))
    try:
        for required in ("get_mock_weather", "calculate"):
            find_tool(tools, required)
    except RuntimeError as exc:
        print(f"❌ 工具发现不符合预期：{exc}")
        return False
    print("✅ MCP stdio 连接与工具发现")

    checks = [
        ("get_mock_weather", {"city": "南京"}, ("南京", "22", "模拟")),
        ("calculate", {"operation": "multiply", "a": 125, "b": 8}, ("1000",)),
        ("calculate", {"operation": "add", "a": 12, "b": 8}, ("20",)),
        ("get_mock_weather", {"city": "不存在的城市"}, ("仅支持",)),
    ]
    for name, args, fragments in checks:
        tool = find_tool(tools, name)
        response = await tool.ainvoke(args)
        result_text = content_text(response)
        if not all(fragment in result_text for fragment in fragments):
            print(f"❌ {name}({args}) 输出不符合预期：{result_text}")
            return False
        print(f"✅ {name}({args}) → {result_text}")
    print("self-test: ✅ 全部通过（真实 MCP stdio，不调用 Gemini）")
    return True


def check_online(case: str, messages: list) -> list[str]:
    """对真实 Tool Call 和 ToolMessage 验收；避免仅凭最终回答宣称调用成功。"""
    issues = []
    calls = [
        call
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
    tool_responses = [m for m in messages if isinstance(m, ToolMessage)]
    # 错误正文可能重复请求中的数字，不能作为成功工具结果参与内容验收。
    if any(m.status != "success" for m in tool_responses):
        issues.append("MCP 工具执行失败，不能以错误正文作为成功结果")
    called = [call.get("name", "") for call in calls]
    output = content_text(messages[-1]) if messages else ""
    tool_text = "\n".join(content_text(m) for m in tool_responses if m.status == "success")

    required = {
        "weather": ("get_mock_weather",),
        "math": ("calculate",),
        "combined": ("get_mock_weather", "calculate"),
    }[case]
    for suffix in required:
        if not any(name == suffix or name.endswith("_" + suffix) for name in called):
            issues.append(f"缺少实际 MCP 工具调用：{suffix}")
        matching_call_ids = {
            c.get("id")
            for c in calls
            if c.get("name", "") == suffix or c.get("name", "").endswith("_" + suffix)
        }
        if not any(m.tool_call_id in matching_call_ids for m in tool_responses):
            issues.append(f"缺少与 {suffix} 的 Tool Call ID 对应的 ToolMessage")

    fragments = {
        "weather": ("南京", "22", "模拟"),
        "math": ("1000",),
        "combined": ("上海", "24", "20", "模拟"),
    }[case]
    for fragment in fragments:
        if fragment not in tool_text:
            issues.append(f"MCP 工具结果未包含预期数据：{fragment}")
    # 终端还会打印最终回答，判断 Agent 是否使用工具结果。
    reply_fragments = {"weather": ("南京", "模拟"), "math": ("1000",), "combined": ("上海", "20", "模拟")}[case]
    for fragment in reply_fragments:
        if fragment not in output:
            issues.append(f"Agent 最终回答未包含：{fragment}")

    if len(tool_responses) < len(calls):
        issues.append("部分 Tool Call 缺少返回的 ToolMessage")
    return issues


async def run_case(agent, name: str) -> bool:
    prompt = CASES[name]
    print(f"\n============ {name} ============")
    print("用户:", prompt)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={
            "run_name": f"chapter05-{name}",
            "tags": ["zrLangChain", "chapter-05", name, "mcp", "stdio"],
            "metadata": {"chapter": "05-mcp-agent", "scenario": name, "transport": "stdio"},
        },
    )
    messages = result.get("messages", [])
    calls = [tc for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls]
    responses = [m for m in messages if isinstance(m, ToolMessage)]
    print("\nAgent:", content_text(messages[-1]) if messages else "（没有回答）")
    print("Tool Calls:", [(c.get("name"), c.get("args")) for c in calls])
    print("Tool Results:", [content_text(m) for m in responses])
    print("LLM AIMessage:", sum(isinstance(m, AIMessage) for m in messages))
    issues = check_online(name, messages)
    print("验证结果:", "✅ 通过" if not issues else "❌ 未通过：" + "; ".join(issues))
    return not issues


async def main_async(args: argparse.Namespace) -> int:
    if not args.self_test:
        load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    if not SERVER_SCRIPT.is_file():
        raise FileNotFoundError(f"找不到 MCP Server：{SERVER_SCRIPT}")
    # FastMCP 将 Path 目标作为本地 Python 脚本，以 stdio 子进程方式启动。
    async with MCPAdapter(SERVER_SCRIPT) as adapter:
        tools = await adapter.list_tools()
        if args.self_test:
            return 0 if await self_test(tools) else 1
        print("MCP Tools:", ", ".join(sorted(tool_names(tools))))
        for required in ("get_mock_weather", "calculate"):
            find_tool(tools, required)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY 未配置；请检查仓库根目录 .env。")
        model = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            api_key=api_key,
        )
        agent = create_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT)
        cases = list(CASES) if args.case == "all" else [args.case]
        success = True
        for name in cases:
            try:
                if not await run_case(agent, name):
                    success = False
            except Exception as exc:
                success = False
                print(f"❌ {name} 运行失败：{type(exc).__name__}: {exc}")
    print("\n========== 总体验收 ==========")
    print("✅ 所选实验全部通过" if success else "❌ 存在失败实验")
    return 0 if success else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Chapter 05: MCP Agent")
    parser.add_argument("--self-test", action="store_true", help="通过真实 MCP stdio 测试工具，零 Gemini 调用")
    parser.add_argument("--case", choices=[*CASES, "all"], default="all", help="选择在线实验场景")
    args = parser.parse_args()
    try:
        return asyncio.run(main_async(args))
    except Exception as exc:
        print(f"❌ 运行失败：{type(exc).__name__}: {exc}")
        return 1
    finally:
        if not args.self_test:
            wait_for_all_tracers()


if __name__ == "__main__":
    raise SystemExit(main())
