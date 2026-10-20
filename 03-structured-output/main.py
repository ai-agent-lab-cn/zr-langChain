"""Chapter 03: Structured Output — LangChain + Gemini + Pydantic + LangSmith.

Run from the zrLangChain repository root:
    uv run python 03-structured-output/main.py --self-test
    uv run python 03-structured-output/main.py --case all
    uv run python 03-structured-output/main.py --case all --strategy provider
"""

import argparse
import os
import re
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, ValidationError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


class ContactInfo(BaseModel):
    """Contact details explicitly provided by a user."""

    name: str = Field(description="联系人姓名")
    email: str | None = Field(description="邮箱；原文未提供时必须为 null")
    phone: str | None = Field(description="电话号码；原文未提供时必须为 null")


class ReviewAnalysis(BaseModel):
    """Structured review with a bounded rating and enumerated sentiment."""

    rating: int = Field(ge=1, le=5, description="明确给出的 1–5 分评分")
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="评论整体情绪，只能从三个枚举值中选择"
    )
    key_points: list[str] = Field(
        min_length=1, max_length=4, description="原文提及的 1–4 个评价要点"
    )


class ActionItem(BaseModel):
    """One action item extracted from meeting notes."""

    task: str = Field(description="待完成的具体任务")
    owner: str = Field(description="责任人姓名")
    priority: Literal["high", "medium", "low"] = Field(
        description="优先级：高=high，中=medium，低=low"
    )
    due_date: str = Field(description="截止日期，格式 YYYY-MM-DD")


class MeetingActions(BaseModel):
    """Meeting topic and its action items."""

    topic: str = Field(description="会议讨论的项目主题")
    tasks: list[ActionItem] = Field(min_length=1, description="明确分配的任务清单")


SCENARIOS = {
    "contact": {
        "schema": ContactInfo,
        "question": (
            "从下面的信息中提取联系人，原文没有提供的字段填写 null，不要编造。"
            "\n客户：张三；邮箱：zhangsan@example.com；未提供电话。"
        ),
    },
    "review": {
        "schema": ReviewAnalysis,
        "question": (
            "分析下面的用户评价，提取评分、整体情绪和评价要点，只依据原文。"
            "\n评论：这款耳机音质很好，物流也很快，整体非常满意。评分为 5 分（满分 5 分）。"
        ),
    },
    "meeting": {
        "schema": MeetingActions,
        "question": (
            "从下面的会议纪要中提取项目主题和任务列表，日期必须使用 YYYY-MM-DD。"
            "\n会议主题：客服工单智能分类。"
            "张三负责在 2026-10-25 前提交需求清单，优先级高；"
            "李四负责在 2026-10-28 前完成演示文稿，优先级中。"
        ),
    },
}

SYSTEM_PROMPT = (
    "You extract structured facts from the user's text. "
    "Use only information explicitly stated. Never invent missing contact fields, "
    "ratings, names, dates, or tasks. Use null for missing nullable fields. "
    "Use the requested enum values and ISO date strings (YYYY-MM-DD). "
    "Return data matching the specified structured output schema."
)


def build_agent(schema: type[BaseModel], strategy: str):
    """Reuse the working Gemini + create_agent architecture from earlier chapters."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 未配置；请检查仓库根目录 .env。")

    model = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        api_key=api_key,
    )
    response_format = (
        ToolStrategy(schema, handle_errors=True)
        if strategy == "tool"
        else ProviderStrategy(schema)
    )
    return create_agent(
        model=model,
        tools=[],  # 本章聚焦输出结构；ToolStrategy 的 schema tool 不是业务工具。
        system_prompt=SYSTEM_PROMPT,
        response_format=response_format,
    )


def validate_result(case_name: str, value: object) -> list[str]:
    """Validate schema AND known facts; a well-formed JSON object can still be wrong."""
    issues: list[str] = []
    schema = SCENARIOS[case_name]["schema"]

    if not isinstance(value, schema):
        return [f"structured_response 类型错误：需要 {schema.__name__}，实际 {type(value).__name__}"]

    # JSON 往返验证：检查序列化后仍满足 Pydantic Schema。
    try:
        schema.model_validate_json(value.model_dump_json())
    except ValidationError as exc:
        return [f"Pydantic 校验失败：{exc}"]

    if case_name == "contact":
        if value.name != "张三":
            issues.append("姓名不是张三")
        if value.email != "zhangsan@example.com":
            issues.append("邮箱与原文不一致")
        if value.phone is not None:
            issues.append("原文没有电话，phone 必须为 null")

    elif case_name == "review":
        if value.rating != 5:
            issues.append("评分应为 5")
        if value.sentiment != "positive":
            issues.append("整体情绪应为 positive")
        if not value.key_points or not all(point.strip() for point in value.key_points):
            issues.append("评价要点不能包含空值")

    elif case_name == "meeting":
        if "客服工单" not in value.topic:
            issues.append("会议主题与原文不符")
        if len(value.tasks) != 2:
            issues.append("任务数应为 2")
        expected = {
            ("张三", "high", "2026-10-25"),
            ("李四", "medium", "2026-10-28"),
        }
        actual = {(item.owner, item.priority, item.due_date) for item in value.tasks}
        if actual != expected:
            issues.append("责任人、优先级或截止日期与原文不一致")
        if any(not re.fullmatch(r"\d{4}-\d{2}-\d{2}", item.due_date) for item in value.tasks):
            issues.append("日期不是 YYYY-MM-DD 格式")

    return issues


def self_test() -> bool:
    """Offline schema and business-check tests; no API keys or network required."""
    fixtures = {
        "contact": ContactInfo(name="张三", email="zhangsan@example.com", phone=None),
        "review": ReviewAnalysis(rating=5, sentiment="positive", key_points=["音质好", "物流快"]),
        "meeting": MeetingActions(
            topic="客服工单智能分类",
            tasks=[
                ActionItem(task="提交需求清单", owner="张三", priority="high", due_date="2026-10-25"),
                ActionItem(task="完成演示文稿", owner="李四", priority="medium", due_date="2026-10-28"),
            ],
        ),
    }
    for name, obj in fixtures.items():
        issues = validate_result(name, obj)
        print(f"{name}: {'✅ 通过' if not issues else '❌ ' + '; '.join(issues)}")
        if issues:
            return False

    invalid_samples = [
        (ReviewAnalysis, {"rating": 6, "sentiment": "positive", "key_points": ["好"]}),
        (ReviewAnalysis, {"rating": 5, "sentiment": "unknown", "key_points": ["好"]}),
        (ContactInfo, {"name": "张三", "email": "zhangsan@example.com"}),
        (MeetingActions, {"topic": "客服工单", "tasks": []}),
    ]
    for schema, bad in invalid_samples:
        try:
            schema.model_validate(bad)
        except ValidationError:
            continue
        print(f"❌ 未拦截非法数据：{schema.__name__}: {bad}")
        return False

    # 即使满足 Pydantic 类型，也必须拦截原文不存在的电话。
    if not validate_result(
        "contact", ContactInfo(name="张三", email="zhangsan@example.com", phone="13800000000")
    ):
        print("❌ 业务校验未拦截幻觉电话")
        return False

    print("离线自检：✅ 全部通过（3 个正常样例 + 5 个异常检查）")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Chapter 03 Structured Output")
    parser.add_argument(
        "--case", choices=[*SCENARIOS, "all"], default="contact",
        help="运行场景；all 会调用多次 Gemini API",
    )
    parser.add_argument(
        "--strategy", choices=["tool", "provider"], default="tool",
        help="结构化输出策略：tool=工具调用，provider=Gemini 原生结构化输出",
    )
    parser.add_argument("--self-test", action="store_true", help="仅运行本地 Schema/业务校验，不调用 API")
    args = parser.parse_args()

    if args.self_test:
        return 0 if self_test() else 1

    cases = list(SCENARIOS) if args.case == "all" else [args.case]
    success = True
    try:
        for case_name in cases:
            scenario = SCENARIOS[case_name]
            print(f"\n{'=' * 12} {case_name} / {args.strategy} {'=' * 12}")
            print("用户问题:", scenario["question"])
            try:
                agent = build_agent(scenario["schema"], args.strategy)
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": scenario["question"]}]},
                    config={
                        "run_name": f"chapter03-{case_name}-{args.strategy}",
                        "tags": ["zrLangChain", "chapter-03", case_name, args.strategy],
                        "metadata": {
                            "chapter": "03-structured-output",
                            "scenario": case_name,
                            "strategy": args.strategy,
                        },
                    },
                )
                output = result.get("structured_response")
                print("\n========== Structured Response ==========")
                if isinstance(output, BaseModel):
                    print(output.model_dump_json(indent=2))
                else:
                    print(repr(output))

                issues = validate_result(case_name, output)
                messages = result.get("messages", [])
                model_messages = sum(isinstance(m, AIMessage) for m in messages)
                schema_tool_calls = sum(
                    len(m.tool_calls) for m in messages if isinstance(m, AIMessage)
                )
                tool_messages = sum(isinstance(m, ToolMessage) for m in messages)
                print("\n========== 追踪摘要 ==========")
                print(f"AIMessage: {model_messages}; Tool Calls: {schema_tool_calls}; ToolMessage: {tool_messages}")
                print("说明：ToolStrategy 可能通过 schema tool 完成输出；它不是实际业务工具。")
                print("\n========== 输出校验 ==========")
                if issues:
                    success = False
                    for issue in issues:
                        print("❌", issue)
                    print("验证结果: ❌ 未通过")
                else:
                    print("验证结果: ✅ 符合预期")
            except Exception as exc:
                success = False
                print(f"❌ {case_name} 运行失败：{type(exc).__name__}: {exc}")
    finally:
        wait_for_all_tracers()

    print("\n========== 总体验收 ==========")
    print("✅ 所选实验全部通过" if success else "❌ 存在失败实验；请检查输出及 LangSmith Trace")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
