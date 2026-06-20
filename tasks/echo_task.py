"""EchoTask:Step 1 用的最小任务,judge 永远成功。

仅用于打通空闭环,不涉及任何工具或真实判分逻辑。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from proxy.recorder import Event
from tasks.base import Task


class EchoTask(Task):
    task_id = "echo"

    def get_prompt(self) -> str:
        return "ping"

    def get_tools(self) -> List[Callable[..., Any]]:
        return []

    def judge(self, final_output: str, trajectory: List[Event]) -> Dict[str, Any]:
        return {"success": True, "score": 1.0, "reason": "echo 任务恒成功"}
