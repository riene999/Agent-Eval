"""把 agent-eval 的评测产物导出为后训练数据(SFT / DPO)。

- SFT:从数据集的 gold_trajectory 渲染"正确的多轮 function-calling 对话"(执行只读工具补全
  工具返回),教模型何时调哪个工具、参数怎么填。无需跑模型。
  产 sft.jsonl(每行 {messages:[...]})+ tools.json(工具函数定义,供训练框架声明工具)。
- DPO:对某次评测里**失败**的题:prompt=系统+问题,chosen=gold 正确动作,rejected=模型实际
  (错误)动作,产 {prompt, chosen, rejected}(verl 的偏好/RM 三列、TRL DPO 文本格式通用)。

用法:
  python -m analysis.export sft --out data/train/sft.jsonl
  python -m analysis.export dpo --run-id exp1 exp2 [--agent react_agent_v1] [--mode failed_only|all] --out data/train/dpo.jsonl
  python -m analysis.export dpo --run-id exp3 --append --out data/train/dpo.jsonl   # 后续实验增量并入(自动去重)
"""

from __future__ import annotations

import argparse
import glob
import inspect
import json
import sys
import typing
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from analysis.metrics import _load_file
from proxy.recorder import PROJECT_ROOT, rehydrate

sys.path.insert(0, str(PROJECT_ROOT / "data"))
from enterprise_kb.task import EnterpriseKBTask, _load_specs  # noqa: E402
from enterprise_kb.tools import TOOLS, TOOLS_BY_NAME  # noqa: E402

_PYTYPE_TO_JSON = {int: "integer", float: "number", str: "string", bool: "boolean",
                   list: "array", dict: "object"}


def _derive_schema(fn: Callable[..., Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    props, required = {}, []
    for name, p in sig.parameters.items():
        if name == "self" or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        props[name] = {"type": _PYTYPE_TO_JSON.get(hints.get(name), "string")}
        if p.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "function", "function": {
        "name": fn.__name__, "description": (inspect.getdoc(fn) or "").strip(),
        "parameters": {"type": "object", "properties": props, "required": required}}}


def _args_str(args: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in (args or {}).items())


def _gold_completion(gold: List[Dict[str, Any]]) -> str:
    """把 gold_trajectory 渲染成可读的"动作序列"(DPO 的 chosen)。"""
    lines = []
    for s in gold:
        if "tool" in s:
            lines.append(f"调用工具 {s['tool']}({_args_str(s.get('args', {}))})")
        elif "say" in s:
            lines.append(f"回复用户:{s['say']}")
        elif "final" in s:
            lines.append(f"最终回答:{s['final']}")
    return "\n".join(lines)


def _model_completion(events: List[Any]) -> str:
    """从模型轨迹渲染它实际做的动作序列(DPO 的 rejected)。"""
    lines = []
    for e in sorted(events, key=lambda x: x.seq):
        d = rehydrate(e.data)
        if e.event_type == "tool_call":
            lines.append(f"调用工具 {d.get('tool_name')}({_args_str(d.get('args', {}))})")
        elif e.event_type == "final_output":
            lines.append(f"最终回答:{d.get('output', '')}")
    return "\n".join(lines)


def export_sft(out_path: Path) -> int:
    """从 gold 渲染多轮 function-calling SFT;另写 tools.json(工具定义)。"""
    (out_path.parent / "tools.json").write_text(
        json.dumps([_derive_schema(t) for t in TOOLS], ensure_ascii=False, indent=2),
        encoding="utf-8")
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for spec in _load_specs().values():
            task = EnterpriseKBTask(spec)
            msgs: List[Dict[str, Any]] = [
                {"role": "system", "content": task.system_prompt()},
                {"role": "user", "content": spec["question"]},
            ]
            for i, s in enumerate(spec["gold_trajectory"]):
                if "tool" in s:
                    fn = TOOLS_BY_NAME.get(s["tool"])
                    try:
                        result = fn(**s.get("args", {})) if fn else f"Error: 未知工具 {s['tool']}"
                    except Exception as e:
                        result = f"Error: {e!r}"
                    cid = f"call_{i}"
                    msgs.append({"role": "assistant", "content": "", "tool_calls": [
                        {"id": cid, "type": "function", "function": {
                            "name": s["tool"],
                            "arguments": json.dumps(s.get("args", {}), ensure_ascii=False)}}]})
                    msgs.append({"role": "tool", "tool_call_id": cid,
                                 "content": result if isinstance(result, str)
                                 else json.dumps(result, ensure_ascii=False)})
                elif "say" in s:
                    msgs.append({"role": "assistant", "content": s["say"]})
                elif "final" in s:
                    msgs.append({"role": "assistant", "content": s["final"]})
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            n += 1
    return n


def export_dpo(run_ids: List[str], agent: str, mode: str, out_path: Path,
               append: bool = False) -> Tuple[int, int]:
    """把一个或多个实验(run_ids)的评测轨迹,产出 {prompt, chosen(gold), rejected(模型)} 偏好对。

    - run_ids:传多个实验,会一并扫进同一个输出文件。
    - append=True:追加到已有文件(而不是覆盖),用于后续新实验增量并入。
    - 自动按 (题目, rejected) 去重:同一题同样的错误只收一次;同题不同的错法都保留(对 DPO 是好事)。
    """
    specs = _load_specs()
    seen: set = set()  # (task_id, rejected) 去重键
    if append and out_path.exists():  # 追加模式:先把已有的读进来,避免重复写
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                seen.add((r.get("task_id"), r.get("rejected")))
            except Exception:
                continue
    n, skipped = 0, 0
    with open(out_path, "a" if append else "w", encoding="utf-8") as f:
        for run_id in run_ids:
            # 精确匹配:本批文件名只能是 <run_id>.jsonl(单试验)或 <run_id>_t*.jsonl(多试验),
            # 不能用 <run_id>*.jsonl——否则前缀相同的别的实验(如 ekb_ds_100_react_qw)会被误吸进来。
            base = str(PROJECT_ROOT / "trajectories" / agent / "ekb_*")
            paths = glob.glob(f"{base}/{run_id}.jsonl") + glob.glob(f"{base}/{run_id}_t*.jsonl")
            for path in sorted(set(paths)):
                events = _load_file(path)
                if not events:
                    continue
                spec = specs.get(events[0].task_id)
                if not spec:
                    continue
                fin = [e for e in events if e.event_type == "final_output"]
                success = bool(fin[0].data.get("success")) if fin else False
                if mode == "failed_only" and success:  # 只要失败题
                    skipped += 1
                    continue
                task = EnterpriseKBTask(spec)
                chosen = _gold_completion(spec["gold_trajectory"])
                rejected = _model_completion(events)
                if chosen.strip() == rejected.strip():  # 模型做对且和 gold 一样,没偏好信号
                    skipped += 1
                    continue
                key = (events[0].task_id, rejected)
                if key in seen:  # 同题同错,已收录
                    skipped += 1
                    continue
                seen.add(key)
                f.write(json.dumps({
                    "prompt": f"{task.system_prompt()}\n\n# 用户问题\n{spec['question']}",
                    "chosen": chosen, "rejected": rejected,
                    "task_id": events[0].task_id, "success": success, "run_id": run_id,
                }, ensure_ascii=False) + "\n")
                n += 1
    return n, skipped


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="导出后训练数据(SFT/DPO)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("sft", help="从 gold 导出 SFT")
    sp.add_argument("--out", default="data/train/sft.jsonl")
    dp = sub.add_parser("dpo", help="从失败评测导出 DPO 偏好对")
    dp.add_argument("--run-id", dest="run_ids", nargs="+", required=True,
                    help="一个或多个实验 run-id(空格分隔),会一并并入同一输出")
    dp.add_argument("--agent", default="react_agent_v1")
    dp.add_argument("--mode", default="failed_only", choices=["failed_only", "all"],
                    help="failed_only=只挑失败题;all=连'做对但绕路'的也收(教简洁)")
    dp.add_argument("--append", action="store_true",
                    help="追加到已有文件并自动去重(后续新实验增量并入时用)")
    dp.add_argument("--out", default="data/train/dpo.jsonl")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.cmd == "sft":
        print(f"SFT 导出 {export_sft(out)} 条 -> {out}(工具定义 -> {out.parent / 'tools.json'})")
    else:
        n, sk = export_dpo(args.run_ids, args.agent, args.mode, out, append=args.append)
        verb = "追加" if args.append else "导出"
        print(f"DPO {verb} {n} 对(跳过 {sk})-> {out}")


if __name__ == "__main__":
    main()
