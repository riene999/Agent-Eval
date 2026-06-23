"""主入口:跑一组 (agent, task) 组合。

按名字解析 agent 与 task,在 run_context 内执行 agent.run(),随后用 task.judge()
判分并补写 final_output 事件。判分与成功标记由本层统一负责,Agent 不自报成功。
注册表用延迟导入,使尚未实现/缺依赖的链路不影响已有链路运行。
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Sequence
from uuid import uuid4

from agents.base import BaseAgent
from proxy.recorder import (
    PROJECT_ROOT,
    append_event,
    load_env,
    read_events,
    run_context,
    trajectory_path,
)
from tasks.base import Task

logger = logging.getLogger(__name__)


def make_agent(name: str, model: Optional[str] = None) -> BaseAgent:
    if name == "echo":
        from agents.echo_agent import EchoAgent

        return EchoAgent()  # echo 不调模型,忽略 model
    if name == "react":
        from agents.react_agent import ReactAgent

        return ReactAgent(model=model)
    raise SystemExit(f"未知 agent: {name!r}(可选:echo, react)")


def make_task(task_id: str) -> Task:
    if task_id == "echo":
        from tasks.echo_task import EchoTask

        return EchoTask()
    if task_id == "math":
        from tasks.math_task import MathTask

        return MathTask()
    if task_id.startswith("tau_"):
        from tasks.tau_bench import load_tau_task

        return load_tau_task(task_id)
    raise SystemExit(f"未知 task: {task_id!r}(可选:echo, math, tau_retail_XXX)")


def _should_attribute(mode: Optional[str], in_sample: bool, success: bool) -> bool:
    if mode == "all":
        return True
    if mode == "failed_only":
        return not success
    if mode == "sample":
        return in_sample
    return False


def _run_evaluator(
    kind: str,
    model: Optional[str],
    agent: BaseAgent,
    task: Task,
    run_id: str,
    verdict: Dict[str, Any],
) -> None:
    """跑一个评测器,结果追加为同名事件;best-effort,失败不拖垮主流程。"""
    events = read_events(agent.agent_id, task.task_id, run_id)
    try:
        if kind == "llm_judge":
            from evaluators.llm_judge import LlmJudge

            data = LlmJudge(model).evaluate(task, events, verdict)
        else:
            from evaluators.attributor import Attributor

            data = Attributor(model).evaluate(task, events, verdict)
    except Exception as e:
        logger.warning("%s 评测失败: %r", kind, e)
        data = {"error": repr(e)}
    append_event(
        agent_id=agent.agent_id, task_id=task.task_id, run_id=run_id, event_type=kind, data=data
    )


def run_one(
    agent: BaseAgent,
    task: Task,
    run_id: str,
    *,
    llm_judge: bool = False,
    attribution_mode: Optional[str] = None,
    in_sample: bool = False,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """执行一次 (agent, task),判分,并按需跑可选的 LLM 评测/归因。"""
    logger.info("开始运行 agent=%s task=%s run_id=%s", agent.agent_id, task.task_id, run_id)
    with run_context(run_id, agent.agent_id, task.task_id):
        result = agent.run(task, run_id)
        output = str(result.get("output", ""))
        events = read_events(agent.agent_id, task.task_id, run_id)
        verdict = task.judge(output, events)
        append_event(
            agent_id=agent.agent_id,
            task_id=task.task_id,
            run_id=run_id,
            event_type="final_output",
            data={
                "output": output,
                "success": bool(verdict.get("success", False)),
                "error": result.get("error"),
            },
        )
        if llm_judge:
            _run_evaluator("llm_judge", judge_model, agent, task, run_id, verdict)
        if attribution_mode and _should_attribute(
            attribution_mode, in_sample, bool(verdict.get("success"))
        ):
            _run_evaluator("attribution", judge_model, agent, task, run_id, verdict)
    return verdict


def resolve_task_ids(args: argparse.Namespace) -> list[str]:
    """按参数决定要跑哪些 task_id:单题 / 显式多题 / tau 批量(split+范围)。"""
    if args.task:
        return [args.task]
    if args.tasks:
        return [t.strip() for t in args.tasks.split(",") if t.strip()]
    if args.count is not None:
        from tasks.tau_bench import tau_task_id

        return [tau_task_id(args.split, i) for i in range(args.start, args.start + args.count)]
    raise SystemExit("请指定 --task <id> / --tasks a,b,c / --count N 三者之一")


def main(argv: Optional[Sequence[str]] = None) -> None:
    load_env()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="运行一组或一批 (agent, task)")
    parser.add_argument("--agent", required=True, help="agent 名:echo | react")
    parser.add_argument("--model", default=None,
                        help="被测 agent 使用的模型;不填则用 .env 的 AGENT_MODEL")
    parser.add_argument("--task", default=None, help="单题 id:echo | math | tau_retail_XXX")
    parser.add_argument("--tasks", default=None, help="逗号分隔的多个 task id,批量跑")
    parser.add_argument("--split", default="test", choices=["test", "train", "dev"],
                        help="tau split(配合 --count)")
    parser.add_argument("--start", type=int, default=0, help="tau 起始题号(配合 --count)")
    parser.add_argument("--count", type=int, default=None,
                        help="从 --start 起批量跑多少道 tau 题")
    parser.add_argument("--run-id", default=None, help="不指定则自动生成;批量时作为本批共同标签")
    parser.add_argument("--llm-judge", action="store_true", help="开启 LLM-as-judge 质量打分")
    parser.add_argument("--attribution", action="store_true", help="开启错误归因")
    parser.add_argument("--attribution-mode", default="failed_only",
                        help="failed_only(默认) | all | sample_N")
    parser.add_argument("--judge-model", default=None,
                        help="评测/归因用模型;默认复用 --model / AGENT_MODEL")
    args = parser.parse_args(argv)

    task_ids = resolve_task_ids(args)
    agent = make_agent(args.agent, args.model)
    run_id = args.run_id or uuid4().hex[:12]

    # 归因策略:failed_only / all / sample_N(随机抽 N 道做归因)
    judge_model = args.judge_model or args.model
    attribution_mode: Optional[str] = None
    sampled: set[str] = set()
    if args.attribution:
        m = args.attribution_mode
        if m in ("failed_only", "all"):
            attribution_mode = m
        elif m.startswith("sample_"):
            try:
                n = int(m.split("_", 1)[1])
            except ValueError:
                raise SystemExit(f"--attribution-mode 形如 sample_N,得到: {m!r}")
            import random

            attribution_mode = "sample"
            sampled = set(random.sample(task_ids, min(n, len(task_ids))))
        else:
            raise SystemExit(f"未知 --attribution-mode: {m!r}")

    results: list[tuple[str, Dict[str, Any]]] = []
    for tid in task_ids:
        task = make_task(tid)
        # 同 run_id 重跑同题会把事件追加到旧文件,先清掉保证干净
        path = trajectory_path(agent.agent_id, task.task_id, run_id)
        if path.exists():
            path.unlink()
        verdict = run_one(
            agent, task, run_id,
            llm_judge=args.llm_judge,
            attribution_mode=attribution_mode,
            in_sample=(tid in sampled),
            judge_model=judge_model,
        )
        results.append((task.task_id, verdict))

    from analysis.metrics import build_report, compute_metrics, to_markdown

    rows = []
    for tid, _ in results:
        events = read_events(agent.agent_id, tid, run_id)
        if events:
            rows.append(compute_metrics(events))

    if len(results) == 1:
        tid, verdict = results[0]
        print(
            f"[done] agent={agent.agent_id} task={tid} run_id={run_id} "
            f"success={verdict.get('success')} score={verdict.get('score')} "
            f"reason={verdict.get('reason')!r}\n  轨迹: {trajectory_path(agent.agent_id, tid, run_id)}"
        )
    else:
        print(f"\n[批量完成] agent={agent.agent_id} run_id={run_id} 共 {len(results)} 题:")
        for tid, verdict in results:
            print(f"  [{tid}] success={verdict.get('success')} reason={verdict.get('reason')!r}")
        print("\n" + to_markdown(rows))

    # 写报告文件:本批共享 run_id,故 reports/<run_id>.md 恰好对应"这一次"的题目
    meta = {
        "run_id": run_id,
        "agent_id": agent.agent_id,
        "model": getattr(agent, "model", None),
        "split": args.split if args.count is not None else None,
        "count": len(results),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "per_task": results,
    }
    report_path = PROJECT_ROOT / "reports" / f"{run_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(meta, rows), encoding="utf-8")
    print(f"\n报告已写入: {report_path}")


if __name__ == "__main__":
    main()
