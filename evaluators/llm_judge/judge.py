"""LLM-as-judge:从 LLM 维度给质量分,和规则 judge 互补(无参考打分)。

只看任务目标 + 交互轨迹 + 规则判分结果,不喂标准答案;输出结构化分数与理由。
"""

from __future__ import annotations

from typing import Any, Dict, List

from evaluators.base import Evaluator
from evaluators.transcript import render
from proxy.recorder import Event

_SYS = (
    "你是严格的对话式 Agent 评测员。根据任务目标与完整交互轨迹,对 Agent 的表现打分。"
    "只输出一个 JSON 对象,不要任何多余文字或解释。"
)

_SCHEMA = (
    '{\n'
    '  "overall": 0~1 的总分(float),\n'
    '  "dimensions": {"task_completion": 0~1, "policy_adherence": 0~1, "efficiency": 0~1},\n'
    '  "reason": "简要中文理由"\n'
    '}'
)


class LlmJudge(Evaluator):
    name = "llm_judge"
    role = "judge"

    def evaluate(
        self, task: Any, events: List[Event], verdict: Dict[str, Any]
    ) -> Dict[str, Any]:
        goal = (task.goal_text() if hasattr(task, "goal_text") else None) or "(未提供)"
        user = (
            f"# 任务目标\n{goal}\n\n"
            f"# 交互轨迹(按步号)\n{render(events)}\n\n"
            f"# 规则判分(仅供参考,不要照搬)\n"
            f"success={verdict.get('success')} reason={verdict.get('reason')}\n\n"
            f"# 严格按以下 JSON 输出\n{_SCHEMA}"
        )
        result = self._chat_json(_SYS, user)
        result["model"] = self.model
        return result
