"""Chapter 04: Agent Memory — Short-Term Memory with LangChain + Gemini.

Run from the zrLangChain repository root:
    uv run python 04-agent-memory/main.py --self-test
    uv run python 04-agent-memory/main.py --case all
"""

import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import tracing_context

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = (
    "You are a concise assistant answering in Chinese. "
    "Answer using only the conversation history in the current thread. "
    "Remember user-provided facts in this thread and respect explicit corrections. "
    "If a fact has not been given in this thread, say '不知道' instead of guessing. "
    "Do not claim to remember facts from other threads or previous processes."
)


def thread_config(thread_id: str, run_name: str, case: str) -> dict:
    """Keep conversation identity in configurable; trace metadata is separate."""
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": run_name,
        "tags": ["zrLangChain", "chapter-04", case],
        "metadata": {"chapter": "04-agent-memory", "scenario": case},
    }


def content_as_text(message) -> str:
    """Read text without printing Gemini's non-text signature blocks."""
    blocks = getattr(message, "content_blocks", [])
    text = "\n".join(
        block.get("text", "") for block in blocks if block.get("type") == "text"
    ).strip()
    if text:
        return text
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else ""


def history(agent, config: dict) -> list:
    """Read the checkpoint; NOT merely the result of the last model call."""
    snapshot = agent.get_state(config)
    return list(snapshot.values.get("messages", []))


def user_messages(messages: list) -> list[str]:
    return [content_as_text(m) for m in messages if isinstance(m, HumanMessage)]


def ask(agent, case: str, step: str, thread_id: str, question: str) -> tuple[str, list]:
    config = thread_config(thread_id, f"chapter04-{case}-{step}", case)
    print(f"\n--- {case} / {step} ---")
    print("thread_id:", thread_id)
    print("用户:", question)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )
    answer = content_as_text(result["messages"][-1])
    messages = history(agent, config)
    print("Agent:", answer or "[无文本回答]")
    print(f"Checkpoint: {len(messages)} 条消息；其中 HumanMessage {len(user_messages(messages))} 条")
    return answer, messages


def check(case: str, conditions: dict[str, bool]) -> bool:
    """Print every criterion, not just a single model-answer-based verdict."""
    print("\n========== 验证结果 ==========")
    for title, passed in conditions.items():
        print(f"{'✅' if passed else '❌'} {title}")
    success = all(conditions.values())
    print(f"{case}: {'✅ 通过' if success else '❌ 未通过'}")
    return success


def run_remember(agent) -> bool:
    thread_id = "chapter04-remember-A"
    fact = "我叫张三，目前负责客服工单智能分类项目。请记住。"
    ask(agent, "remember", "introduce", thread_id, fact)
    answer, saved = ask(
        agent, "remember", "recall", thread_id,
        "我叫什么名字？负责哪个项目？请根据本会话的历史回答。",
    )
    originals = user_messages(saved)
    return check("remember", {
        "同一 thread 保留两轮用户消息": len(originals) == 2,
        "Checkpoint 保留第一轮事实": fact in originals,
        "Agent 回忆姓名": "张三" in answer,
        "Agent 回忆项目": "客服工单智能分类" in answer,
    })


def run_isolation(agent) -> bool:
    thread_a = "chapter04-isolation-A"
    thread_b = "chapter04-isolation-B"
    secret = "蓝鲸-42"
    ask(agent, "isolation", "set-A", thread_a, f"我的项目代号是{secret}，仅对当前会话有效。请记住。")
    answer_b, saved_b = ask(
        agent, "isolation", "ask-B", thread_b,
        "我以前告诉过你的项目代号是什么？如果本会话中没提供，请回答不知道。",
    )
    answer_a, saved_a = ask(
        agent, "isolation", "recall-A", thread_a,
        "我刚才告诉你的项目代号是什么？",
    )
    return check("isolation", {
        "A/B 使用不同 thread_id": thread_a != thread_b,
        "B 的 Checkpoint 没有 A 的代号": all(secret not in m for m in user_messages(saved_b)),
        "B 拒绝猜测未知代号": ("不知道" in answer_b or "未提供" in answer_b or "没有" in answer_b) and secret not in answer_b,
        "A 的 Checkpoint 保留代号": any(secret in m for m in user_messages(saved_a)),
        "A 可以回忆代号": secret in answer_a,
    })


def run_update(agent) -> bool:
    thread_id = "chapter04-update-A"
    first = "我的首选编程语言是 Python，请记住。"
    correction = "更正：我的首选编程语言现在是 Go，不是 Python。请以最新信息为准。"
    ask(agent, "update", "old", thread_id, first)
    ask(agent, "update", "correction", thread_id, correction)
    answer, saved = ask(
        agent, "update", "latest", thread_id,
        "我现在首选的编程语言是什么？请只回复语言名称。",
    )
    originals = user_messages(saved)
    return check("update", {
        "Checkpoint 保留三轮用户消息": len(originals) == 3,
        "历史中保留原信息和更正信息": first in originals and correction in originals,
        "最终回答以更正后的 Go 为准": bool(re.search(r"(?<![A-Za-z])Go(?![A-Za-z])", answer)) and "Python" not in answer,
    })


def self_test() -> bool:
    """Offline Agent + checkpoint integration test, without Gemini or LangSmith calls."""
    with tracing_context(enabled=False):
        return offline_checks()


def offline_checks() -> bool:
    """在关闭追踪的局部上下文中执行真实 Agent 和 Checkpoint 检查。"""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    saver = InMemorySaver()
    fake = FakeListChatModel(responses=["已记录", "我在同一会话中", "独立会话"])
    agent = create_agent(model=fake, tools=[], checkpointer=saver)
    a = {"configurable": {"thread_id": "chapter04-offline-A"}}
    b = {"configurable": {"thread_id": "chapter04-offline-B"}}

    agent.invoke({"messages": [{"role": "user", "content": "记住：测试标记甲"}]}, config=a)
    agent.invoke({"messages": [{"role": "user", "content": "这是第二轮"}]}, config=a)
    agent.invoke({"messages": [{"role": "user", "content": "另一个线程"}]}, config=b)

    saved_a = user_messages(history(agent, a))
    saved_b = user_messages(history(agent, b))

    # A different saver (simulating a fresh process) cannot retrieve old data.
    fresh_agent = create_agent(
        model=FakeListChatModel(responses=["新的进程"]),
        tools=[], checkpointer=InMemorySaver(),
    )
    fresh = history(fresh_agent, a)

    return check("self-test", {
        "同一线程自动累积两轮对话": saved_a == ["记住：测试标记甲", "这是第二轮"],
        "不同线程仅包含自己的消息": saved_b == ["另一个线程"],
        "新的 InMemorySaver 没有旧数据": len(fresh) == 0,
        "get_state() 可直接读取 Checkpoint": len(history(agent, a)) == 4,
    })


def build_agent():
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 未配置；请检查项目根目录的 .env。")
    model = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        api_key=api_key,
    )
    return create_agent(
        model=model,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Chapter 04 Agent Memory")
    parser.add_argument(
        "--case", choices=["remember", "isolation", "update", "all"],
        default="all", help="在线记忆实验，默认 all",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="离线测试 Checkpointer，不调用 Gemini API",
    )
    args = parser.parse_args()

    if args.self_test:
        try:
            return 0 if self_test() else 1
        except Exception as exc:
            print(f"❌ 离线自检失败：{type(exc).__name__}: {exc}")
            return 1

    runner = {
        "remember": run_remember,
        "isolation": run_isolation,
        "update": run_update,
    }
    try:
        agent = build_agent()
        cases = list(runner) if args.case == "all" else [args.case]
        results: dict[str, bool] = {}
        for case in cases:
            try:
                print(f"\n{'=' * 16} {case} {'=' * 16}")
                results[case] = runner[case](agent)
            except Exception as exc:
                print(f"❌ {case} 运行失败：{type(exc).__name__}: {exc}")
                results[case] = False
        print("\n========== 总体验收 ==========")
        for case, passed in results.items():
            print(f"{'✅' if passed else '❌'} {case}")
        passed_all = all(results.values())
        print("✅ 所选实验全部通过" if passed_all else "❌ 存在未通过的实验")
        return 0 if passed_all else 1
    except Exception as exc:
        print(f"❌ 初始化失败：{type(exc).__name__}: {exc}")
        return 1
    finally:
        wait_for_all_tracers()


if __name__ == "__main__":
    raise SystemExit(main())
