"""主入口:跑一组 (agent, task) 组合。

按名字解析 agent 与 task,在 run_context 内执行 agent.run(),随后用 task.judge()
判分并补写 final_output 事件。判分与成功标记由本层统一负责,Agent 不自报成功。
注册表用延迟导入,使尚未实现/缺依赖的链路不影响已有链路运行。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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


def make_agent(name: str, model: Optional[str] = None, temperature: float = 0.0) -> BaseAgent:
    if name == "echo":
        from agents.echo_agent import EchoAgent

        return EchoAgent()  # echo 不调模型,忽略 model/temperature
    if name == "react":
        from agents.react_agent import ReactAgent

        return ReactAgent(model=model, temperature=temperature)
    if name == "plan_solve":
        from agents.plan_solve import PlanSolveAgent

        return PlanSolveAgent(model=model, temperature=temperature)
    if name == "skill_router":
        from agents.skill_router import SkillRouterAgent

        return SkillRouterAgent(model=model, temperature=temperature)
    raise SystemExit(f"未知 agent: {name!r}(可选:echo, react, plan_solve, skill_router)")


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
    if task_id.startswith("ekb_"):  # 自建数据集 data/enterprise_kb(自包含)
        import sys

        kb_root = str(PROJECT_ROOT / "data")
        if kb_root not in sys.path:
            sys.path.insert(0, kb_root)
        from enterprise_kb.task import load_task

        return load_task(task_id)
    raise SystemExit(f"未知 task: {task_id!r}(可选:echo, math, tau_retail_XXX, ekb_XXX)")


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
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """执行一次 (agent, task),判分,并按需跑可选的 LLM 评测/归因。"""
    logger.info("开始运行 agent=%s task=%s run_id=%s", agent.agent_id, task.task_id, run_id)
    with run_context(run_id, agent.agent_id, task.task_id, seed=seed):
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
                # 路径保真(对 gold);非 tau 任务为 None,分析端自动跳过
                "tool_selection": verdict.get("tool_selection"),
                "arg_correctness": verdict.get("arg_correctness"),
                "skill_expected": verdict.get("skill_expected"),
                "skill_gold": verdict.get("skill_gold"),
                "skill_selected": verdict.get("skill_selected"),
                "skill_routing_correct": verdict.get("skill_routing_correct"),
                "skill_scope_correct": verdict.get("skill_scope_correct"),
                "skill_case_type": verdict.get("skill_case_type"),
                "cross_skill_tool_rate": verdict.get("cross_skill_tool_rate"),
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
        if args.tasks.strip() == "all-ekb":  # 跑整个自建数据集 enterprise_kb
            import sys

            sys.path.insert(0, str(PROJECT_ROOT / "data"))
            from enterprise_kb.task import _load_specs

            return list(_load_specs().keys())
        return [t.strip() for t in args.tasks.split(",") if t.strip()]
    if args.count is not None:
        from tasks.tau_bench import tau_task_id

        return [tau_task_id(args.split, i) for i in range(args.start, args.start + args.count)]
    raise SystemExit("请指定 --task <id> / --tasks a,b,c / --count N 三者之一")


def _preflight_proxy(timeout: float = 3.0) -> None:
    """开跑前探一下代理 /health;连不上就立刻退出,别让整批题对着死代理 churn。"""
    import urllib.error
    import urllib.request

    base = os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1").rstrip("/")
    # 去掉末尾的 /v1 再接 /health(代理的 /health 在根路径,不在 /v1 下)
    root = base[:-3] if base.endswith("/v1") else base
    health = root.rstrip("/") + "/health"
    # 探活这一跳同样要绕开系统/VPN 代理(直连 localhost)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(health, timeout=timeout) as r:
            if r.status == 200:
                return
        raise SystemExit(f"代理 {health} 返回异常状态,无法开跑。")
    except (urllib.error.URLError, OSError, ConnectionError) as e:
        raise SystemExit(
            f"连不上代理({health}):{e}\n"
            f"请先在另一个终端启动代理:  uv run python -m proxy.server\n"
            f"(确认 models.json 已配好对应模型的上游与 key)"
        )


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
    parser.add_argument("--concurrency", type=int, default=1,
                        help="并发数(有界线程池,跨 题×试验);默认 1=串行")
    parser.add_argument("--trials", type=int, default=1,
                        help="每题跑几次(>1 出 pass@k 与方差);默认 1")
    parser.add_argument("--seed", type=int, default=None,
                        help="基准随机种子,第 t 次试验用 seed+t(注入 LLM 请求,可复现)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度;测多试验多样性建议 >0")
    parser.add_argument(
        "--skills",
        default=None,
        help="逗号分隔的 Skill ID；填写后把企业知识任务包装为 Skill 专项任务",
    )
    parser.add_argument(
        "--skill-mode",
        default=None,
        choices=["single", "multi"],
        help="写入报告的 Skill 评测类型",
    )
    args = parser.parse_args(argv)

    _preflight_proxy()  # 代理没起就别开跑,免得对着死代理 churn 一整批
    task_ids = resolve_task_ids(args)
    agent = make_agent(args.agent, args.model, args.temperature)
    run_id = args.run_id or uuid4().hex[:12]
    trials = max(1, args.trials)

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

    # 熔断:跑到一半代理挂了/连不上时,连续多次连接错误就中止整批,别 churn 完剩下的题
    _abort = threading.Event()
    _conn_fails = [0]
    _fail_lock = threading.Lock()
    _CONN_FAIL_LIMIT = int(os.getenv("CONN_FAIL_LIMIT", "5"))

    def _is_conn_error(e: Exception) -> bool:
        return "Connection" in type(e).__name__ or "APIConnectionError" in repr(e)

    def _worker(tid: str, trial: int) -> tuple[str, str, Dict[str, Any]]:
        # 多试验:每次试验单独 run_id 与 seed,各写各的轨迹文件
        trial_run_id = run_id if trials == 1 else f"{run_id}_t{trial}"
        if _abort.is_set():  # 已熔断:后面的题直接跳过,不再发请求
            return tid, trial_run_id, {"success": False, "score": 0.0, "reason": "批量已中止(代理不可达)"}
        seed = None if args.seed is None else args.seed + trial
        task = make_task(tid)
        if args.skills:
            if not tid.startswith("ekb_"):
                raise ValueError("Skill 评测当前只支持 enterprise_kb 任务")
            from tasks.skill_eval import SkillEvalTask

            skill_ids = [item.strip() for item in args.skills.split(",") if item.strip()]
            task = SkillEvalTask(task, skill_ids, eval_mode=args.skill_mode or "multi")
        path = trajectory_path(agent.agent_id, task.task_id, trial_run_id)
        if path.exists():
            path.unlink()
        try:
            verdict = run_one(
                agent, task, trial_run_id,
                llm_judge=args.llm_judge,
                attribution_mode=attribution_mode,
                in_sample=(tid in sampled),
                judge_model=judge_model,
                seed=seed,
            )
        except Exception as e:  # 单个试验异常不拖垮整批
            logger.warning("任务 %s 第 %d 次异常: %r", tid, trial, e)
            if _is_conn_error(e):
                with _fail_lock:
                    _conn_fails[0] += 1
                    if _conn_fails[0] >= _CONN_FAIL_LIMIT and not _abort.is_set():
                        _abort.set()
                        logger.error("连续 %d 次连不上代理,已中止本批(请检查 proxy.server 是否在跑)。",
                                     _conn_fails[0])
            verdict = {"success": False, "score": 0.0, "reason": f"运行异常: {e!r}"}
        return task.task_id, trial_run_id, verdict

    # 工作单元 = 题 × 试验;各单元独立(各自数据库/轨迹文件,记录层按线程隔离、
    # blob 原子写,天然并发安全),有界并发;结果按提交顺序回填
    units = [(tid, t) for tid in task_ids for t in range(trials)]
    raw: list = [None] * len(units)
    pool = ThreadPoolExecutor(max_workers=max(1, args.concurrency))
    futures = {pool.submit(_worker, tid, t): i for i, (tid, t) in enumerate(units)}
    try:
        # 带超时的轮询等待:主线程每 0.3s 醒一次,Ctrl+C 才能被及时接住
        # (阻塞在 as_completed 时,在飞请求要等到 read timeout 才返回,中断会被拖住)。
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=0.3, return_when=FIRST_COMPLETED)
            for fut in done:
                raw[futures[fut]] = fut.result()
    except KeyboardInterrupt:
        # 在飞的 HTTP 请求最长要等 read timeout 才结束,普通退出会被 atexit 强制 join 线程
        # 而卡住;直接 os._exit 硬退出,跳过线程 join,保证一按 Ctrl+C 立刻死。
        logger.warning("收到中断,强制退出。")
        os._exit(130)
    pool.shutdown(wait=False, cancel_futures=True)

    from analysis.metrics import (
        build_report,
        build_trials_report,
        compute_metrics,
        pass_at_k,
        report_json,
        to_markdown,
        trials_json,
    )

    report_path = PROJECT_ROOT / "reports" / f"{run_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if trials == 1:
        # —— 单次:维持原有行为 ——
        results = [(task_id, verdict) for (task_id, _rid, verdict) in raw]
        rows = []
        for task_id, trial_run_id, _v in raw:
            events = read_events(agent.agent_id, task_id, trial_run_id)
            if events:
                rows.append(compute_metrics(events))
        if len(results) == 1:
            task_id, verdict = results[0]
            print(
                f"[done] agent={agent.agent_id} task={task_id} run_id={run_id} "
                f"success={verdict.get('success')} score={verdict.get('score')} "
                f"reason={verdict.get('reason')!r}\n  轨迹: "
                f"{trajectory_path(agent.agent_id, task_id, run_id)}"
            )
        else:
            print(f"\n[批量完成] agent={agent.agent_id} run_id={run_id} 共 {len(results)} 题:")
            for task_id, verdict in results:
                print(f"  [{task_id}] success={verdict.get('success')} reason={verdict.get('reason')!r}")
            print("\n" + to_markdown(rows))
        meta = {
            "run_id": run_id,
            "agent_id": agent.agent_id,
            "model": getattr(agent, "model", None),
            "skills": args.skills.split(",") if args.skills else None,
            "skill_mode": args.skill_mode,
            "split": args.split if args.count is not None else None,
            "count": len(results),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "per_task": results,
        }
        report_path.write_text(build_report(meta, rows), encoding="utf-8")
        report_path.with_suffix(".json").write_text(
            json.dumps(report_json(meta, rows), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        # —— 多试验:按 task_id 归组,算 pass@k + 方差 ——
        per_task: Dict[str, list] = {}
        for task_id, trial_run_id, _v in raw:
            events = read_events(agent.agent_id, task_id, trial_run_id)
            if events:
                per_task.setdefault(task_id, []).append(compute_metrics(events))
        p1s = [pass_at_k(len(rs), sum(r["accuracy"] for r in rs), 1) for rs in per_task.values()]
        pNs = [pass_at_k(len(rs), sum(r["accuracy"] for r in rs), len(rs)) for rs in per_task.values()]
        print(f"\n[多试验完成] agent={agent.agent_id} run_id={run_id} {len(per_task)} 题 × {trials} 次")
        if p1s:
            import statistics

            print(f"  平均 pass@1={statistics.mean(p1s):.1%}  平均 pass@{trials}={statistics.mean(pNs):.1%}")
        meta = {
            "run_id": run_id,
            "agent_id": agent.agent_id,
            "model": getattr(agent, "model", None),
            "skills": args.skills.split(",") if args.skills else None,
            "skill_mode": args.skill_mode,
            "split": args.split if args.count is not None else None,
            "trials": trials,
            "seed": args.seed,
            "temperature": args.temperature,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        items = list(per_task.items())
        report_path.write_text(build_trials_report(meta, items), encoding="utf-8")
        report_path.with_suffix(".json").write_text(
            json.dumps(trials_json(meta, items), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"\n报告已写入: {report_path}(同名 .json 一并导出)")


if __name__ == "__main__":
    main()
