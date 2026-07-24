"""两份评测报告的配对比较与回归门禁。

只比较两份报告中共同出现的任务，避免题目范围不同造成虚假的提升或退化。
支持准确率、Token、延迟、成本和重复工具调用五类门禁，并为差值计算自助法置信区间。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_THRESHOLDS: Dict[str, float] = {
    "accuracy_drop_max": 0.03,
    "token_increase_max": 0.20,
    "latency_p95_increase_max": 0.25,
    "cost_increase_max": 0.20,
    "redundant_call_rate_increase_max": 0.05,
}

METRICS = (
    "accuracy", "total_tokens", "tool_call_count", "redundant_call_rate",
    "latency_p95", "cost_usd", "tool_selection", "arg_correctness",
)


def _mean(values: Iterable[Any]) -> Optional[float]:
    usable = [float(value) for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def _task_rows(report: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    rows: Dict[str, Dict[str, Optional[float]]] = {}
    for task in report.get("tasks", []):
        trials = task.get("trials") if isinstance(task.get("trials"), list) else [task]
        rows[str(task.get("task_id", ""))] = {
            metric: _mean(trial.get(metric) for trial in trials) for metric in METRICS
        }
    return rows


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_delta(pairs: List[tuple[float, float]], samples: int = 1000) -> Optional[List[float]]:
    if not pairs:
        return None
    rng = random.Random(42)
    deltas: List[float] = []
    for _ in range(samples):
        picked = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        deltas.append(sum(candidate - baseline for baseline, candidate in picked) / len(picked))
    low, high = _percentile(deltas, 0.025), _percentile(deltas, 0.975)
    return [round(low or 0.0, 6), round(high or 0.0, 6)]


def _metric_result(name: str, baseline: Dict[str, Dict[str, Optional[float]]],
                   candidate: Dict[str, Dict[str, Optional[float]]], tasks: List[str]) -> Dict[str, Any]:
    pairs = [(baseline[task][name], candidate[task][name]) for task in tasks]
    usable = [(float(left), float(right)) for left, right in pairs if left is not None and right is not None]
    left_mean = _mean(left for left, _ in usable)
    right_mean = _mean(right for _, right in usable)
    delta = right_mean - left_mean if left_mean is not None and right_mean is not None else None
    relative = delta / left_mean if delta is not None and left_mean not in (None, 0) else None
    return {
        "baseline": left_mean,
        "candidate": right_mean,
        "delta": delta,
        "relative_change": relative,
        "paired_tasks": len(usable),
        "delta_ci95": _bootstrap_delta(usable),
    }


def compare_payloads(baseline_report: Dict[str, Any], candidate_report: Dict[str, Any],
                     thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """比较两份已加载的报告，并返回前端和命令行共用的结构化结果。"""
    gates = dict(DEFAULT_THRESHOLDS)
    gates.update(thresholds or {})
    baseline = _task_rows(baseline_report)
    candidate = _task_rows(candidate_report)
    common = sorted(set(baseline) & set(candidate))
    metrics = {name: _metric_result(name, baseline, candidate, common) for name in METRICS}

    checks = [
        _gate("accuracy", "准确率下降", metrics["accuracy"].get("delta"), -gates["accuracy_drop_max"], "min", "绝对变化"),
        _gate("total_tokens", "Token 增长", metrics["total_tokens"].get("relative_change"), gates["token_increase_max"], "max", "相对变化"),
        _gate("latency_p95", "p95 延迟增长", metrics["latency_p95"].get("relative_change"), gates["latency_p95_increase_max"], "max", "相对变化"),
        _gate("cost_usd", "成本增长", metrics["cost_usd"].get("relative_change"), gates["cost_increase_max"], "max", "相对变化"),
        _gate("redundant_call_rate", "重复调用比例增长", metrics["redundant_call_rate"].get("delta"), gates["redundant_call_rate_increase_max"], "max", "绝对变化"),
    ]
    evaluated = [check for check in checks if check["status"] != "skipped"]
    status = "passed" if common and all(check["status"] == "passed" for check in evaluated) else "failed"
    if not common:
        status = "failed"

    return {
        "status": status,
        "paired_task_count": len(common),
        "baseline_only_count": len(set(baseline) - set(candidate)),
        "candidate_only_count": len(set(candidate) - set(baseline)),
        "baseline": _report_meta(baseline_report),
        "candidate": _report_meta(candidate_report),
        "thresholds": gates,
        "metrics": metrics,
        "checks": checks,
        "tasks": [{"task_id": task, "baseline": baseline[task], "candidate": candidate[task]} for task in common],
    }


def _gate(metric: str, label: str, value: Optional[float], threshold: float,
          direction: str, value_kind: str) -> Dict[str, Any]:
    if value is None:
        status = "skipped"
    elif direction == "min":
        status = "passed" if value >= threshold else "failed"
    else:
        status = "passed" if value <= threshold else "failed"
    return {"metric": metric, "label": label, "value": value, "threshold": threshold,
            "direction": direction, "value_kind": value_kind, "status": status}


def _report_meta(report: Dict[str, Any]) -> Dict[str, Any]:
    meta = report.get("meta", {})
    return {key: meta.get(key) for key in ("run_id", "agent_id", "model", "timestamp", "trials")}


def compare_files(baseline_path: Path, candidate_path: Path,
                  thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    return compare_payloads(
        json.loads(baseline_path.read_text(encoding="utf-8")),
        json.loads(candidate_path.read_text(encoding="utf-8")),
        thresholds,
    )


def to_markdown(result: Dict[str, Any]) -> str:
    labels = {
        "accuracy": "准确率", "total_tokens": "平均 Token", "tool_call_count": "平均工具调用",
        "redundant_call_rate": "重复调用比例", "latency_p95": "p95 延迟", "cost_usd": "平均成本",
        "tool_selection": "工具选择正确率", "arg_correctness": "参数正确率",
    }
    lines = [
        f"# 评测对比 {result['baseline'].get('run_id')} → {result['candidate'].get('run_id')}", "",
        f"- 门禁：{'✅ 通过' if result['status'] == 'passed' else '❌ 未通过'}",
        f"- 共同题目：{result['paired_task_count']}", "",
        "| 指标 | 基线 | 候选 | 变化 |", "| --- | ---: | ---: | ---: |",
    ]
    for name, metric in result["metrics"].items():
        if metric["baseline"] is None:
            continue
        lines.append(f"| {labels[name]} | {metric['baseline']:.4g} | {metric['candidate']:.4g} | {metric['delta']:+.4g} |")
    lines += ["", "## 回归门禁", "", "| 检查项 | 结果 | 实际变化 | 阈值 |", "| --- | :---: | ---: | ---: |"]
    for check in result["checks"]:
        icon = {"passed": "✅", "failed": "❌", "skipped": "⏭️"}[check["status"]]
        actual = "-" if check["value"] is None else f"{check['value']:+.1%}"
        lines.append(f"| {check['label']} | {icon} | {actual} | {check['threshold']:+.1%} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="比较两次评测并执行回归门禁")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out")
    parser.add_argument("--max-accuracy-drop", type=float, default=0.03,
                        help="允许准确率最多下降多少，0.03 表示 3 个百分点")
    parser.add_argument("--max-token-increase", type=float, default=0.20)
    parser.add_argument("--max-latency-increase", type=float, default=0.25)
    parser.add_argument("--max-cost-increase", type=float, default=0.20)
    parser.add_argument("--max-redundant-increase", type=float, default=0.05)
    args = parser.parse_args()
    result = compare_files(Path(args.baseline), Path(args.candidate), {
        "accuracy_drop_max": args.max_accuracy_drop,
        "token_increase_max": args.max_token_increase,
        "latency_p95_increase_max": args.max_latency_increase,
        "cost_increase_max": args.max_cost_increase,
        "redundant_call_rate_increase_max": args.max_redundant_increase,
    })
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        out.with_suffix(".md").write_text(to_markdown(result), encoding="utf-8")
    print(to_markdown(result))
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
