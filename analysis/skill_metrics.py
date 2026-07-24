"""单 Skill 与多 Skill 评测的聚合指标。

本模块只消费通用每题指标，不参与 Agent 执行。旧报告没有 Skill 字段时返回空结果，
因此不会影响 τ-bench、普通企业知识问答或历史轨迹。
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

    confusion = sum(
        1
        for row in in_scope
        if row.get("skill_selected") not in {None, row.get("skill_expected")}
    )
    return {
        "skill_task_count": len(skill_rows),
        "in_scope_count": len(in_scope),
        "out_of_scope_count": len(out_scope),
        "in_scope_success_rate": average(in_scope, "accuracy"),
        "routing_accuracy": average(skill_rows, "skill_routing_correct"),
        "boundary_accuracy": average(out_scope, "skill_scope_correct"),
        "skill_confusion_rate": confusion / len(in_scope) if in_scope else None,
        "cross_skill_tool_rate": average(skill_rows, "cross_skill_tool_rate"),
    }


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
