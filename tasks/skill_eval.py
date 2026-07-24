"""企业知识问答的 Skill 提示词包装器。

Skill 只作为可插拔提示词追加到原任务的 system prompt，不负责路由，也不改变
Agent 链路、任务工具集合和原始判分；因此同一 Skill 可公平用于任意 Agent。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from proxy.recorder import Event
from skills.registry import get_skill, list_skills
from tasks.base import Task


class SkillEvalTask(Task):
    """为已有任务追加一组 Skill 提示词。"""

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
        self.active_skill_ids = [spec.skill_id for spec in self.skill_specs]
        self.gold_skill_id = self._find_gold_skill()
        self.prompt_matches_task = self.gold_skill_id in set(self.active_skill_ids)
        self.case_type = "in_scope" if self.prompt_matches_task else "out_of_scope"

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
        blocks = []
        for spec in self.skill_specs:
            tools = "、".join(spec.tools)
            blocks.append(
                f"## {spec.name}（{spec.skill_id}）\n"
                f"{spec.description}\n"
                f"{spec.instructions}\n"
                f"相关工具：{tools}"
            )
        return (
            f"{self.base.system_prompt()}\n\n"
            "# 已启用的 Skill 提示词\n"
            "以下内容是处理任务时可参考的补充规则；请结合用户问题选择适用规则，"
            "不适用的 Skill 不要强行套用。\n\n"
            + "\n\n".join(blocks)
        )

    def goal_text(self) -> Optional[str]:
        return self.base.goal_text()

    def reference_summary(self) -> Optional[str]:
        return self.base.reference_summary()

    def user_turn(self, message: str) -> Optional[str]:
        return self.base.user_turn(message)

    def get_tools(self) -> List[Callable[..., Any]]:
        return self.base.get_tools()

    def judge(self, final_output: str, trajectory: List[Event]) -> Dict[str, Any]:
        verdict = dict(self.base.judge(final_output, trajectory))
        verdict.update(
            {
                "skills_enabled": self.active_skill_ids,
                "skill_gold": self.gold_skill_id,
                "skill_prompt_active": self.prompt_matches_task,
                "skill_case_type": self.case_type,
            }
        )
        return verdict
