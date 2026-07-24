"""Skill 专项评测入口。

单 Skill 模式直接运行一组“能力内 + 能力外”企业任务；N+1 模式固定模型、题目和
采样参数，依次运行原 Skill 集合与“原集合 + 新 Skill”，最后生成增益/退化报告。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

from analysis.skill_metrics import compare_skill_runs
from proxy.recorder import PROJECT_ROOT, load_env
from skills.registry import get_skill


def _task_ids(start: int, count: int) -> list[str]:
    path = PROJECT_ROOT / "data" / "enterprise_kb" / "tasks.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row["id"] for row in rows[start:start + count]]


def _run_command(
    *,
    run_id: str,
    model: str,
    task_ids: list[str],
    skills: list[str],
    mode: str,
    trials: int,
    concurrency: int,
    temperature: float,
    seed: Optional[int],
) -> None:
    command = [
        sys.executable,
        "-m",
        "runner.run",
        "--agent",
        "skill_router",
        "--model",
        model,
        "--tasks",
        ",".join(task_ids),
        "--run-id",
        run_id,
        "--skills",
        ",".join(skills),
        "--skill-mode",
        mode,
        "--trials",
        str(trials),
        "--concurrency",
        str(concurrency),
        "--temperature",
        str(temperature),
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _flat_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in report.get("tasks", []):
        if "trials" not in item:
            result.append(item)
            continue
        trials = item.get("trials") or []
        if not trials:
            continue
        row = dict(trials[0])
        row["task_id"] = item["task_id"]
        row["accuracy"] = sum(float(trial.get("accuracy", 0)) for trial in trials) / len(trials)
        result.append(row)
    return result


def _write_n_plus_one_report(
    run_id: str,
    baseline_id: str,
    candidate_id: str,
    candidate_skill: str,
) -> None:
    baseline = json.loads(
        (PROJECT_ROOT / "reports" / f"{baseline_id}.json").read_text(encoding="utf-8")
    )
    candidate = json.loads(
        (PROJECT_ROOT / "reports" / f"{candidate_id}.json").read_text(encoding="utf-8")
    )
    baseline_rows = _flat_rows(baseline)
    candidate_rows = _flat_rows(candidate)
    comparison = compare_skill_runs(baseline_rows, candidate_rows)
    n = len(candidate_rows)
    accuracy = (
        sum(float(row.get("accuracy", 0)) for row in candidate_rows) / n if n else 0.0
    )
    payload = {
        "meta": {
            "run_id": run_id,
            "agent_id": "skill_router_v1",
            "model": candidate.get("meta", {}).get("model"),
            "kind": "skill_n_plus_one",
            "candidate_skill": candidate_skill,
            "baseline_run_id": baseline_id,
            "candidate_run_id": candidate_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "summary": {
            "n": n,
            "accuracy": accuracy,
            "gained_count": len(comparison["gained_tasks"]),
            "regressed_count": len(comparison["regressed_tasks"]),
            "net_gain": comparison["net_gain"],
            **comparison["candidate"],
        },
        "tasks": candidate_rows,
        "skill_evaluation": comparison,
    }
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{run_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"# N+1 Skill 评测报告 {run_id}",
        "",
        f"> 新增 Skill: `{candidate_skill}` · 配对 {comparison['paired_task_count']} 题",
        "",
        "## 结论",
        "",
        f"- 新增后做对、原来做错: **{len(comparison['gained_tasks'])}** 题",
        f"- 原来做对、新增后做错: **{len(comparison['regressed_tasks'])}** 题",
        f"- 净收益: **{comparison['net_gain']:+d}** 题",
        "",
        "## 新增后做对的题",
        "",
        *([f"- `{task_id}`" for task_id in comparison["gained_tasks"]] or ["- 无"]),
        "",
        "## 被新增 Skill 影响失败的旧题",
        "",
        *([f"- `{task_id}`" for task_id in comparison["regressed_tasks"]] or ["- 无"]),
        "",
        f"> 基线报告: `{baseline_id}` · 候选报告: `{candidate_id}`",
    ]
    (report_dir / f"{run_id}.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    load_env()
    parser = argparse.ArgumentParser(description="单 Skill / N+1 Skill 效果评测")
    parser.add_argument("--mode", choices=["single", "n_plus_one"], required=True)
    parser.add_argument("--skill", required=True, help="单测或待新增的 Skill ID")
    parser.add_argument("--baseline-skills", default="", help="N+1 的原有 Skill，逗号分隔")
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    get_skill(args.skill)
    task_ids = _task_ids(args.start, args.count)
    if not task_ids:
        raise SystemExit("所选范围没有企业知识问答任务")

    if args.mode == "single":
        _run_command(
            run_id=args.run_id,
            model=args.model,
            task_ids=task_ids,
            skills=[args.skill],
            mode="single",
            trials=args.trials,
            concurrency=args.concurrency,
            temperature=args.temperature,
            seed=args.seed,
        )
        return

    baseline_skills = [
        item.strip() for item in args.baseline_skills.split(",") if item.strip()
    ]
    if not baseline_skills:
        raise SystemExit("N+1 评测必须提供 --baseline-skills")
    for skill_id in baseline_skills:
        get_skill(skill_id)
    candidate_skills = list(dict.fromkeys([*baseline_skills, args.skill]))
    baseline_id = f"{args.run_id}_baseline"
    candidate_id = f"{args.run_id}_plus"
    _run_command(
        run_id=baseline_id,
        model=args.model,
        task_ids=task_ids,
        skills=baseline_skills,
        mode="multi",
        trials=args.trials,
        concurrency=args.concurrency,
        temperature=args.temperature,
        seed=args.seed,
    )
    _run_command(
        run_id=candidate_id,
        model=args.model,
        task_ids=task_ids,
        skills=candidate_skills,
        mode="multi",
        trials=args.trials,
        concurrency=args.concurrency,
        temperature=args.temperature,
        seed=args.seed,
    )
    _write_n_plus_one_report(args.run_id, baseline_id, candidate_id, args.skill)
    print(f"N+1 Skill 报告已写入: {PROJECT_ROOT / 'reports' / f'{args.run_id}.md'}")


if __name__ == "__main__":
    main()
