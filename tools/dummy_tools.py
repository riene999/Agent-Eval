"""示例/测试用的无副作用工具:加、乘。

用于验证 @traced_tool 记录是否正确,以及为 ReAct Agent 提供一组最小工具集
(配合 MathTask 跑 "(3+5)*2" 这类算术题)。
"""

from __future__ import annotations

from tools.wrapper import traced_tool


@traced_tool
def add(a: float, b: float) -> float:
    """返回 a 与 b 的和。"""
    return a + b


@traced_tool
def multiply(a: float, b: float) -> float:
    """返回 a 与 b 的乘积。"""
    return a * b
