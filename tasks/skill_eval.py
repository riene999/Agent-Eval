"""企业知识问答的 Skill 专项评测任务包装器。

包装器不修改原任务题面、标准轨迹和答案判分，只根据启用的 Skill 限制工具范围，
并在原判分结果上补充 Skill 选择、边界识别和跨 Skill 工具调用指标。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from proxy.recorder import Event
from skills.registry import get_skill, list_skills, resolve_tools
from tasks.base import Task


class SkillEvalTask(Task):
    """把一个已有任务变成可由 Skill Router 执行的任务。"""

    def __init__(
        self,
        base: Task,
        active_skill_ids: list[str],
        *,
        eval_mode: str = "multi",
    ) -> None:
        if not active_skill_ids:
            raise ValueError("Skill 评测至少需要启用一个 Skill")
        self.base = base
        self.eval_mode = eval_mode
        self.task_id = base.task_id
        self.reference_path_length = base.reference_path_length
        self.skill_specs = [get_skill(skill_id) for skill_id in active_skill_ids]
        self.gold_skill_id = self._find_gold_skill()
        active_ids = {spec.skill_id for spec in self.skill_specs}
        self.expected_skill_id = self.gold_skill_id if self.gold_skill_id in active_ids else None
        self.case_type = "in_scope" if self.expected_skill_id else "out_of_scope"

    def _find_gold_skill(self) -> Optional[str]:
        explicit = getattr(self.base, "expected_skill", None)
        if explicit:
            return str(explicit)
        domain = str(getattr(self.base, "domain", "") or "").strip().lower()
        for spec in list_skills(include_disabled=False):
            if domain and domain in spec.domains:
                return spec.skill_id
        return None

    def get_prompt(self) -> str:
        return self.base.get_prompt()

    def system_prompt(self) -> str:
        return self.base.system_prompt()

    def goal_text(self) -> Optional[str]:
        return self.base.goal_text()

    def reference_summary(self) -> Optional[str]:
        return self.base.reference_summary()

    def user_turn(self, message: str) -> Optional[str]:
        return self.base.user_turn(message)

    def get_tools(self) -> List[Callable[..., Any]]:
        unique: dict[str, Callable[..., Any]] = {}
        for spec in self.skill_specs:
            for tool in resolve_tools(spec):
                unique[tool.__name__] = tool
        return list(unique.values())

    def tools_for_skill(self, skill_id: str) -> List[Callable[..., Any]]:
        for spec in self.skill_specs:
            if spec.skill_id == skill_id:
                return resolve_tools(spec)
        return []

    def judge(self, final_output: str, trajectory: List[Event]) -> Dict[str, Any]:
        verdict = dict(self.base.judge(final_output, trajectory))
        answer_success = bool(verdict.get("success"))
        route_events = [event for event in trajectory if event.event_type == "skill_route"]
        selected = route_events[-1].data.get("selected_skill") if route_events else None
        selected = selected if selected not in {"", "none", None} else None
        routing_correct = selected == self.expected_skill_id
        if self.eval_mode == "single" and self.case_type == "out_of_scope":
            verdict["success"] = routing_correct
            verdict["score"] = 1.0 if routing_correct else 0.0
            verdict["reason"] = (
                "正确识别为能力范围外"
                if routing_correct
                else f"范围外问题被错误路由到 {selected}"
            )

        expected_tools: set[str] = set()
        if self.gold_skill_id:
            try:
                expected_tools = set(get_skill(self.gold_skill_id).tools)
            except KeyError:
                expected_tools = set()
        calls = [
            event.data.get("tool_name")
            for event in trajectory
            if event.event_type == "tool_call"
        ]
        wrong_calls = [name for name in calls if expected_tools and name not in expected_tools]
        cross_skill_tool_rate = len(wrong_calls) / len(calls) if calls else 0.0

        verdict.update(
            {
                "skill_expected": self.expected_skill_id,
                "skill_gold": self.gold_skill_id,
                "skill_selected": selected,
                "skill_routing_correct": routing_correct,
                "skill_scope_correct": routing_correct,
                "skill_case_type": self.case_type,
                "cross_skill_tool_rate": cross_skill_tool_rate,
                "skill_answer_success": answer_success,
            }
        )
        return verdict
