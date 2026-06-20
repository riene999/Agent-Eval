"""MathTask:Step 3 用的算术任务,验证 ReAct 工具调用链路。

要求 Agent 必须用 add / multiply 工具算出 (3+5)*2,judge 检查最终回答是否含 16。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from proxy.recorder import Event
from tasks.base import Task
from tools.dummy_tools import add, multiply


class MathTask(Task):
    task_id = "math"

    def get_prompt(self) -> str:
        return (
            "请计算 (3 + 5) * 2 的结果。你必须使用提供的 add 和 multiply 工具来计算,"
            "不要心算。得到结果后用一句话给出最终数字答案。"
        )

    def get_tools(self) -> List[Callable[..., Any]]:
        return [add, multiply]

    def judge(self, final_output: str, trajectory: List[Event]) -> Dict[str, Any]:
        ok = "16" in final_output
        return {
            "success": ok,
            "score": 1.0 if ok else 0.0,
            "reason": f"最终回答{'包含' if ok else '不包含'} 16",
        }
