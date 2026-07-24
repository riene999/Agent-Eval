"""工具调用序列的离线对齐与差异分类。

本模块把两条轨迹视为两串工具动作，使用动态规划寻找总代价最小的配对方式。
输出不仅包含距离，还会区分参数错误、选错工具、遗漏调用和额外调用。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping

from proxy.recorder import rehydrate

Action = Dict[str, Any]


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_tool_actions(events: Iterable[Any]) -> List[Action]:
    """从 Event 对象或原始事件字典中提取按时序排列的工具调用。"""
    actions: List[Action] = []
    for event in events:
        event_type = event.get("event_type") if isinstance(event, Mapping) else event.event_type
        if event_type != "tool_call":
            continue
        data = event.get("data", {}) if isinstance(event, Mapping) else event.data
        seq = event.get("seq") if isinstance(event, Mapping) else event.seq
        hydrated = rehydrate(data)
        actions.append({
            "seq": seq,
            "tool_name": hydrated.get("tool_name", ""),
            "args": hydrated.get("args", {}) or {},
        })
    return actions


def _substitution(left: Action, right: Action) -> tuple[float, str]:
    if left["tool_name"] != right["tool_name"]:
        return 1.0, "wrong_tool"
    if _canonical(left.get("args", {})) != _canonical(right.get("args", {})):
        return 0.75, "argument_mismatch"
    return 0.0, "match"


def align_actions(baseline: List[Action], candidate: List[Action]) -> Dict[str, Any]:
    """用编辑距离式动态规划对齐两串动作，并回溯得到可读差异。"""
    n, m = len(baseline), len(candidate)
    costs = [[0.0] * (m + 1) for _ in range(n + 1)]
    choices: List[List[tuple[str, str] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0] = float(i)
        choices[i][0] = ("missing_call", "up")
    for j in range(1, m + 1):
        costs[0][j] = float(j)
        choices[0][j] = ("extra_call", "left")

    priority = {"match": 0, "argument_mismatch": 1, "wrong_tool": 2,
                "missing_call": 3, "extra_call": 4}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost, sub_type = _substitution(baseline[i - 1], candidate[j - 1])
            options = [
                (costs[i - 1][j - 1] + sub_cost, priority[sub_type], sub_type, "diag"),
                (costs[i - 1][j] + 1.0, priority["missing_call"], "missing_call", "up"),
                (costs[i][j - 1] + 1.0, priority["extra_call"], "extra_call", "left"),
            ]
            best = min(options, key=lambda item: (item[0], item[1]))
            costs[i][j] = best[0]
            choices[i][j] = (best[2], best[3])

    operations: List[Dict[str, Any]] = []
    i, j = n, m
    while i or j:
        operation, direction = choices[i][j] or ("match", "diag")
        if direction == "diag":
            operations.append({"type": operation, "baseline": baseline[i - 1], "candidate": candidate[j - 1]})
            i -= 1
            j -= 1
        elif direction == "up":
            operations.append({"type": operation, "baseline": baseline[i - 1], "candidate": None})
            i -= 1
        else:
            operations.append({"type": operation, "baseline": None, "candidate": candidate[j - 1]})
            j -= 1
    operations.reverse()

    counts = {name: 0 for name in ("match", "argument_mismatch", "wrong_tool", "missing_call", "extra_call")}
    for operation in operations:
        counts[operation["type"]] += 1
    first = next((op for op in operations if op["type"] != "match"), None)
    denominator = max(n, m, 1)
    return {
        "distance": round(costs[n][m], 4),
        "similarity": max(0.0, round(1.0 - costs[n][m] / denominator, 4)),
        "baseline_action_count": n,
        "candidate_action_count": m,
        "first_deviation_seq": (first.get("candidate") or first.get("baseline") or {}).get("seq") if first else None,
        "counts": counts,
        "operations": operations,
    }

