"""错误归因:分析整条轨迹,定位"从哪一步(seq)开始偏离",并给出类别/根因/修复建议。

默认无参考(task.reference_summary() 为 None);若任务提供标准路径则一并喂入。
"""

from __future__ import annotations

from typing import Any, Dict, List

from evaluators.base import Evaluator
from evaluators.transcript import render
from proxy.recorder import Event

_SYS = (
    "你是 Agent 轨迹的错误归因专家。给定任务目标、带步号(seq)的完整轨迹与规则判分,"
    "找出 Agent 从哪一步(引用轨迹里的 seq)开始偏离正确路径并归因。只输出一个 JSON 对象。"
)

# 固定错误类别枚举,便于跨运行聚合统计
_CATEGORIES = (
    "param_hallucination, constraint_forgetting, wrong_tool, missing_step, "
    "premature_stop, misread_tool_output, policy_violation, other"
)

_SCHEMA = (
    '{\n'
    '  "deviation_seq": 整数(从哪一步 seq 开始偏离),\n'
    '  "error_category": "主错误类别(从给定枚举里选一个)",\n'
    '  "error_categories_all": ["命中的所有类别"],\n'
    '  "confidence": 0~1 的置信度(float),\n'
    '  "recoverable": true/false(偏离后是否还有机会自我纠正),\n'
    '  "summary": "一句话说清在第几步发生了什么错",\n'
    '  "root_cause_hypothesis": "对根因的推测",\n'
    '  "fix_suggestion": "具体可执行的修复建议"\n'
    '}'
)


class Attributor(Evaluator):
    name = "attribution"
    role = "attributor"

    def evaluate(
        self, task: Any, events: List[Event], verdict: Dict[str, Any]
    ) -> Dict[str, Any]:
        goal = (task.goal_text() if hasattr(task, "goal_text") else None) or "(未提供)"
        reference = task.reference_summary() if hasattr(task, "reference_summary") else None
        ref_block = f"\n\n# 标准路径(参考)\n{reference}" if reference else ""
        user = (
            f"# 任务目标\n{goal}\n\n"
            f"# 可选错误类别枚举\n{_CATEGORIES}\n\n"
            f"# 带步号(seq)的完整轨迹\n{render(events)}\n\n"
            f"# 规则判分\nsuccess={verdict.get('success')} reason={verdict.get('reason')}"
            f"{ref_block}\n\n"
            f"# 严格按以下 JSON 输出\n{_SCHEMA}"
        )
        result = self._chat_json(_SYS, user)
        result["model"] = self.model
        return result
