import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tracers.langchain import wait_for_all_tracers

# 加载环境变量
load_dotenv(dotenv_path=".env")

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY is not configured")

# 定义工具
def get_weather(city: str) -> str:
    """Get simulated weather for a given city."""
    return f"It's always sunny in {city}! (Simulated data)"


# 创建 Agent
agent = create_agent(
    model=f"google_genai:{os.getenv('GEMINI_MODEL', 'gemini-3.6-flash')}",
    tools=[get_weather],
    system_prompt=(
        "You are a helpful AI assistant. "
        "Use get_weather for weather questions. "
        "Clearly state that weather data is simulated."
    ),
)


# 调用 Agent
def main():
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather in San Francisco?",
                }
            ]
        }
    )
    
    wait_for_all_tracers()

    # print(result["messages"][-1].content_blocks)

    print("\n========== Agent 执行过程 ==========")

    tool_called = False

    for i, message in enumerate(result["messages"], start=1):
        print(f"\nStep {i}: {message.type}")

        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(f"调用工具: {call['name']}")
                print(f"调用参数: {call['args']}")

        elif isinstance(message, ToolMessage):
            tool_called = True
            print(f"工具名称: {message.name}")
            print(f"工具返回: {message.content}")

    print("\n========== 最终回答 ==========")

    final_message = result["messages"][-1]

    for block in final_message.content_blocks:
        if block.get("type") == "text":
            print(block["text"])

    print("\nTool Calling 验证:", "成功" if tool_called else "未执行")

if __name__ == "__main__":
    main()