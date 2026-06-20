"""Agent 抽象基类(共享契约,非某个具体 agent 类型)。

所有链路类型(MVP 只有 ReAct)都实现 run(task, run_id) 这一个接口。Agent 只负责
"想办法完成任务",轨迹由代理层(LLM 调用)与工具 wrapper(工具调用)旁路自动记录,
判分与 final_output 事件由 runner 在 run() 之后统一处理——Agent 不自报成功。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from tasks.base import Task


class BaseAgent(ABC):
    # 子类必须定义,用于轨迹文件路径 trajectories/{agent_id}/...
    agent_id: str

    @abstractmethod
    def run(self, task: Task, run_id: str) -> Dict[str, Any]:
        """执行一个任务,返回至少包含 {"output": str} 的字典。"""
        raise NotImplementedError
