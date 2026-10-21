"""第五章：本地教学 MCP Server（stdio）。

由 main.py 中的 MCPAdapter(Path(...)) 自动启动。
注意：stdio 模式的 stdout 留给 JSON-RPC 协议，调试请用 stderr。
"""

from typing import Literal

from fastmcp import FastMCP

mcp = FastMCP("zrLangChain-Chapter05-Demo")

_MOCK_WEATHER = {
    "南京": ("晴", 22),
    "上海": ("多云", 24),
    "北京": ("小雨", 18),
}


@mcp.tool
def get_mock_weather(city: str) -> str:
    """查询指定城市的*教学模拟*天气。仅支持南京、上海、北京；不是真实天气。"""
    city = city.strip()
    if city not in _MOCK_WEATHER:
        return f"未找到城市「{city}」。演示仅支持：南京、上海、北京。"
    condition, temperature = _MOCK_WEATHER[city]
    return (
        f"{city}模拟天气：{condition}，温度 {temperature}℃。"
        "注意：以上为教学模拟数据，不代表实时天气。"
    )


@mcp.tool
def calculate(operation: Literal["add", "multiply"], a: int, b: int) -> str:
    """执行整数加法或乘法。operation 只能是 add 或 multiply。"""
    if operation == "add":
        return f"计算结果：{a} + {b} = {a + b}"
    return f"计算结果：{a} × {b} = {a * b}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
