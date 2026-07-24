"""历史轨迹的离线回放与双轨迹差异分析。

回放只读取已经落盘的 JSONL 和内容存储，不会再次请求模型或执行工具。
差异分析抽取工具动作并调用序列对齐算法，帮助定位第一次偏离及错误类型。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from analysis.alignment import align_actions, extract_tool_actions
from proxy.recorder import rehydrate


def load_raw_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(rehydrate(json.loads(line)))
    return sorted(events, key=lambda event: int(event.get("seq", 0)))


def compare_trace_files(baseline_path: Path, candidate_path: Path) -> Dict[str, Any]:
    baseline_events = load_raw_events(baseline_path)
    candidate_events = load_raw_events(candidate_path)
    return {
        "baseline": _trace_meta(baseline_events, baseline_path),
        "candidate": _trace_meta(candidate_events, candidate_path),
        "alignment": align_actions(
            extract_tool_actions(baseline_events),
            extract_tool_actions(candidate_events),
        ),
    }


def replay_payload(path: Path) -> Dict[str, Any]:
    events = load_raw_events(path)
    return {"trace": _trace_meta(events, path), "events": events}


def _trace_meta(events: List[Dict[str, Any]], path: Path) -> Dict[str, Any]:
    final = next((event for event in reversed(events) if event.get("event_type") == "final_output"), None)
    return {
        "agent_id": events[0].get("agent_id") if events else path.parts[-3],
        "task_id": events[0].get("task_id") if events else path.parts[-2],
        "run_id": events[0].get("run_id") if events else path.stem,
        "event_count": len(events),
        "success": (final or {}).get("data", {}).get("success"),
        "final_output": (final or {}).get("data", {}).get("output"),
    }
