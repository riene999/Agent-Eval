"""Task 抽象基类。

一个 Task 封装"给 Agent 什么、能用什么工具、如何判分"三件事,且把具体数据集
(tau-bench / 未来的合成集)的格式与评分逻辑完全收敛在子类内部,主干不感知。
reference_path_length 是为合成任务集预留的扩展位:已知最优步数,用于后续计算
path_length_ratio 指标;tau-bench 任务留 None。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from proxy.recorder import Event

# 通用 system 提示;具体任务可通过覆盖 Task.system_prompt() 改成领域政策。
DEFAULT_SYSTEM_PROMPT = (
    "你是一个会使用工具的助手。通过调用提供的工具一步步完成用户的任务;"
    "当你已掌握足够信息时,直接用自然语言给出最终答案,不要再调用工具。"
)


class Task(ABC):
    task_id: str
    # 合成任务的已知最优步数;tau-bench 等无参考路径的任务保持 None。
    reference_path_length: Optional[int] = None

    @abstractmethod
    def get_prompt(self) -> str:
        """返回喂给 Agent 的初始 prompt。"""
        raise NotImplementedError

    def system_prompt(self) -> str:
        """返回给 Agent 的 system 提示(可选覆盖)。

        这是附加钩子而非抽象方法:不覆盖则用通用提示;tau-bench 等任务覆盖它
        返回领域政策(wiki),从而无需改动主干即可注入领域知识。
        """
        return DEFAULT_SYSTEM_PROMPT

    def goal_text(self) -> Optional[str]:
        """无副作用地返回"用户的目标/指令",供评测器(LLM-judge/归因)使用。

        与 get_prompt 不同:get_prompt 可能有副作用(如 tau 会驱动用户模拟器开场);
        本方法只返回静态文本,默认 None。
        """
        return None

    def reference_summary(self) -> Optional[str]:
        """可选的"标准路径"文字摘要,供错误归因参考(gold path 接口)。

        默认 None(无参考归因);需要时子类返回标准答案的步骤摘要。
        """
        return None

    def user_turn(self, message: str) -> Optional[str]:
        """对话式任务的"用户回合"钩子。

        当 Agent 输出一段自然语言(不调用工具)时,框架把它交给本方法:
        - 返回字符串:视为用户对该消息的回复,对话继续(tau-bench 用用户模拟器实现);
        - 返回 None:本任务没有交互用户,这段文本即最终答案,运行结束。
        默认 None(非对话任务,如算术题)。
        """
        return None

    @abstractmethod
    def get_tools(self) -> List[Callable[..., Any]]:
        """返回本任务可用的工具列表(每个都应已被 @traced_tool 装饰)。"""
        raise NotImplementedError

    @abstractmethod
    def judge(self, final_output: str, trajectory: List[Event]) -> Dict[str, Any]:
        """对一次运行判分,返回 {success: bool, score: float, reason: str}。"""
        raise NotImplementedError
