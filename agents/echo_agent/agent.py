"""EchoAgent:Step 1 用的最小 Agent。

不调用任何 LLM 或工具,直接把 prompt 回显出来,用于在没有外部依赖的情况下验证
"runner → agent → 判分 → final_output 事件"这条空闭环是否打通。
"""

from __future__ import annotations

from typing import Any, Dict

from agents.base import BaseAgent
from tasks.base import Task


class EchoAgent(BaseAgent):
    agent_id = "echo_agent_v1"

    def run(self, task: Task, run_id: str) -> Dict[str, Any]:
        prompt = task.get_prompt()
        return {"output": f"echo: {prompt}"}
