"""单 Skill 与 N+1 提示词评测的聚合指标。

新数据只统计 Skill 提示词命中与任务效果；旧版路由轨迹仍按原字段聚合，保证历史
报告可读取，但新评测不再把 Skill 当成独立 Agent 或模型路由。
"""

from __future__ import annotations

from typing import Any


def summarize_skill_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    skill_rows = [row for row in rows if row.get("skill_case_type") is not None]
    if not skill_rows:
        return {}

    in_scope = [row for row in skill_rows if row.get("skill_case_type") == "in_scope"]
    out_scope = [row for row in skill_rows if row.get("skill_case_type") == "out_of_scope"]

    def average(items: list[dict[str, Any]], key: str) -> float | None:
        values = [float(item[key]) for item in items if item.get(key) is not None]
        return sum(values) / len(values) if values else None

    result = {
        "skill_task_count": len(skill_rows),
        "in_scope_count": len(in_scope),
        "out_of_scope_count": len(out_scope),
        "in_scope_success_rate": average(in_scope, "accuracy"),
        "out_of_scope_success_rate": average(out_scope, "accuracy"),
    }
    has_router_data = any(
        row.get("skill_routing_correct") is not None for row in skill_rows
    )
    if has_router_data:
        confusion = sum(
            1
            for row in in_scope
            if row.get("skill_selected") not in {None, row.get("skill_expected")}
        )
        result.update(
            {
                "routing_accuracy": average(skill_rows, "skill_routing_correct"),
                "boundary_accuracy": average(out_scope, "skill_scope_correct"),
                "skill_confusion_rate": confusion / len(in_scope) if in_scope else None,
                "cross_skill_tool_rate": average(skill_rows, "cross_skill_tool_rate"),
            }
        )
    return result


def compare_skill_runs(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = {row["task_id"]: row for row in baseline_rows}
    candidate = {row["task_id"]: row for row in candidate_rows}
    common = sorted(set(baseline) & set(candidate))
    gains = [
        task_id
        for task_id in common
        if not baseline[task_id].get("accuracy") and candidate[task_id].get("accuracy")
    ]
    regressions = [
        task_id
        for task_id in common
        if baseline[task_id].get("accuracy") and not candidate[task_id].get("accuracy")
    ]
    base_summary = summarize_skill_rows([baseline[task_id] for task_id in common])
    cand_summary = summarize_skill_rows([candidate[task_id] for task_id in common])
    return {
        "paired_task_count": len(common),
        "gained_tasks": gains,
        "regressed_tasks": regressions,
        "net_gain": len(gains) - len(regressions),
        "baseline": base_summary,
        "candidate": cand_summary,
    }
