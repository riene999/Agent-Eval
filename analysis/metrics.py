"""离线轨迹分析:读 JSONL,算四项指标并聚合成 markdown 表。

指标定义:
- accuracy:final_output 事件的 success 为真记 1,否则 0;
- total_tokens:被测 Agent(role≠user_sim)的 llm_call 的 prompt+completion 之和,
  不含用户模拟器,衡量"Agent 自身"的 token 开销;
- tool_call_count:tool_call 事件数;
- redundant_call_rate:(总工具调用数 - 去重后调用数) / 总工具调用数,
  其中去重以"工具名 + 参数 JSON(排序后)"是否相等为准。
"""

from __future__ import annotations

import argparse
import glob
import json
from typing import Any, Dict, List

from proxy.recorder import Event


def _load_file(path: str) -> List[Event]:
    events: List[Event] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(Event.model_validate_json(line))
    events.sort(key=lambda e: e.seq)
    return events


def compute_metrics(events: List[Event]) -> Dict[str, Any]:
    agent_id = events[0].agent_id if events else ""
    task_id = events[0].task_id if events else ""
    run_id = events[0].run_id if events else ""

    success = False
    total_tokens = 0
    call_keys: List[str] = []
    for e in events:
        if e.event_type == "llm_call":
            # 只统计被测 Agent 的 token,排除用户模拟器(role=user_sim)
            if e.data.get("role") != "user_sim":
                total_tokens += int(e.data.get("prompt_tokens", 0) or 0)
                total_tokens += int(e.data.get("completion_tokens", 0) or 0)
        elif e.event_type == "tool_call":
            key = json.dumps(
                [e.data.get("tool_name"), e.data.get("args", {})],
                sort_keys=True,
                ensure_ascii=False,
            )
            call_keys.append(key)
        elif e.event_type == "final_output":
            success = bool(e.data.get("success", False))

    tool_call_count = len(call_keys)
    distinct = len(set(call_keys))
    redundant_call_rate = (
        (tool_call_count - distinct) / tool_call_count if tool_call_count else 0.0
    )
    return {
        "agent_id": agent_id,
        "task_id": task_id,
        "run_id": run_id,
        "accuracy": 1 if success else 0,
        "total_tokens": total_tokens,
        "tool_call_count": tool_call_count,
        "redundant_call_rate": redundant_call_rate,
    }


def collect(glob_pattern: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(glob_pattern, recursive=True)):
        events = _load_file(path)
        if events:
            rows.append(compute_metrics(events))
    return rows


def to_markdown(rows: List[Dict[str, Any]]) -> str:
    header = "| agent_id | task_id | accuracy | total_tokens | tool_call_count | redundant_call_rate |"
    sep = "| --- | --- | ---: | ---: | ---: | ---: |"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['agent_id']} | {r['task_id']} | {r['accuracy']} | "
            f"{r['total_tokens']} | {r['tool_call_count']} | {r['redundant_call_rate']:.1%} |"
        )
    if rows:
        n = len(rows)
        acc = sum(r["accuracy"] for r in rows) / n
        tok = sum(r["total_tokens"] for r in rows) / n
        tcc = sum(r["tool_call_count"] for r in rows) / n
        rcr = sum(r["redundant_call_rate"] for r in rows) / n
        lines.append(
            f"| **均值({n} 条)** |  | {acc:.2f} | {tok:.1f} | {tcc:.1f} | {rcr:.1%} |"
        )
    return "\n".join(lines)


def build_report(meta: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    """把一批运行的元信息 + 三轴指标表渲染成一份 markdown 报告。"""
    head = [f"# 评测报告 {meta.get('run_id', '')}", ""]
    for label, key in [("agent", "agent_id"), ("模型", "model"), ("split", "split"),
                       ("题数", "count"), ("时间", "timestamp")]:
        val = meta.get(key)
        if val is not None:
            head.append(f"- {label}: {val}")
    head += ["", "## 三轴指标", "", to_markdown(rows), ""]
    if meta.get("per_task"):
        head += ["## 每题结果", ""]
        for tid, verdict in meta["per_task"]:
            head.append(f"- `{tid}`: success={verdict.get('success')} — {verdict.get('reason', '')}")
    return "\n".join(head)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="聚合轨迹指标为 markdown 表")
    parser.add_argument("--glob", default="trajectories/**/*.jsonl", help="轨迹文件通配")
    args = parser.parse_args(argv)
    rows = collect(args.glob)
    if not rows:
        print(f"未找到匹配 {args.glob!r} 的轨迹文件")
        return
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
