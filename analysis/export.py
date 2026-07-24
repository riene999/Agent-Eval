"""把 agent-eval 的评测产物导出为后训练数据(SFT / DPO)。

- SFT:从数据集的 gold_trajectory 渲染"正确的多轮 function-calling 对话"(执行只读工具补全
  工具返回),教模型何时调哪个工具、参数怎么填。无需跑模型。
  产 sft.jsonl(每行 {messages:[...]})+ tools.json(工具函数定义,供训练框架声明工具)。
- DPO:对某次评测里**失败**的题，找到 gold 与模型轨迹的首次分歧。公共前缀保留为
  OpenAI 多轮 function-calling messages，chosen/rejected 是同一上下文下更好/更差的
  下一条 assistant 消息。这样不会把工具调用降级成“调用工具 xxx”的普通文本。

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


def _json_text(value: Any) -> str:
    """OpenAI tool 消息的 content 必须是字符串。"""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _tool_call_message(name: str, args: Dict[str, Any], call_id: str) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args or {}, ensure_ascii=False, sort_keys=True),
            },
        }],
    }


def _final_message(content: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": content or ""}


def _gold_steps(gold: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any] | None]]:
    """把 gold 变成 [(assistant 决策, 可选 tool observation)]。"""
    steps: List[Tuple[Dict[str, Any], Dict[str, Any] | None]] = []
    for i, item in enumerate(gold):
        if "tool" in item:
            name = item["tool"]
            args = item.get("args", {}) or {}
            call_id = f"call_gold_{i}"
            fn = TOOLS_BY_NAME.get(name)
            try:
                result = fn(**args) if fn else f"Error: 未知工具 {name}"
            except Exception as exc:
                result = f"Error: {exc!r}"
            observation = {
                "role": "tool",
                "tool_call_id": call_id,
                "content": _json_text(result),
            }
            steps.append((_tool_call_message(name, args, call_id), observation))
        elif "say" in item:
            steps.append((_final_message(item["say"]), None))
        elif "final" in item:
            steps.append((_final_message(item["final"]), None))
    return steps


def _model_steps(events: List[Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any] | None]]:
    """从真实事件恢复模型执行过的结构化动作与工具返回。"""
    ordered = sorted(events, key=lambda event: event.seq)
    returns = {
        event.parent_seq: rehydrate(event.data)
        for event in ordered
        if event.event_type == "tool_return" and event.parent_seq is not None
    }
    steps: List[Tuple[Dict[str, Any], Dict[str, Any] | None]] = []
    for event in ordered:
        data = rehydrate(event.data)
        if event.event_type == "tool_call":
            name = data.get("tool_name", "")
            args = data.get("args", {}) or {}
            if not isinstance(args, dict):
                continue
            call_id = f"call_model_{event.seq}"
            returned = returns.get(event.seq)
            observation = None
            if returned is not None:
                result = returned.get("result")
                if returned.get("error"):
                    result = f"Error: {returned['error']}"
                observation = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _json_text(result),
                }
            steps.append((_tool_call_message(name, args, call_id), observation))
        elif event.event_type == "final_output":
            steps.append((_final_message(data.get("output", "")), None))
            break
    return steps


def _action_key(message: Dict[str, Any]) -> Tuple[Any, ...]:
    """只比较决策语义，忽略随机 tool_call_id。"""
    calls = message.get("tool_calls") or []
    if calls:
        fn = calls[0].get("function", {})
        raw_args = fn.get("arguments", "{}") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = raw_args
        return ("tool", fn.get("name", ""), json.dumps(args, ensure_ascii=False, sort_keys=True))
    return ("final", (message.get("content") or "").strip())


def _first_divergence(
    gold_steps: List[Tuple[Dict[str, Any], Dict[str, Any] | None]],
    model_steps: List[Tuple[Dict[str, Any], Dict[str, Any] | None]],
) -> int | None:
    """返回双方都存在下一动作的首次分歧位置；无法形成偏好对时返回 None。"""
    for idx in range(min(len(gold_steps), len(model_steps))):
        if _action_key(gold_steps[idx][0]) != _action_key(model_steps[idx][0]):
            return idx
    return None


def _pair_key(row: Dict[str, Any]) -> str:
    """结构化偏好对的稳定去重键。"""
    return json.dumps(
        [row.get("task_id"), row.get("messages"), row.get("rejected")],
        ensure_ascii=False,
        sort_keys=True,
    )


def export_sft(out_path: Path) -> int:
    """从 gold 渲染多轮 function-calling SFT;另写 tools.json(工具定义)。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    """导出“公共结构化前缀 + 首次分歧动作”的 DPO 偏好对。

    - run_ids:传多个实验,会一并扫进同一个输出文件。
    - append=True:追加到已有文件(而不是覆盖),用于后续新实验增量并入。
    - 自动按 (题目, 公共前缀, rejected) 去重。
    - chosen/rejected 只比较同一上下文中的下一次决策，避免把分叉后的 observation
      错当作共同 prompt。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tool_schemas = [_derive_schema(tool) for tool in TOOLS]
    (out_path.parent / "tools.json").write_text(
        json.dumps(tool_schemas, ensure_ascii=False, indent=2), encoding="utf-8")
    specs = _load_specs()
    seen: set[str] = set()
    if append and out_path.exists():  # 追加模式:先把已有的读进来,避免重复写
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                seen.add(_pair_key(r))
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
                gold_steps = _gold_steps(spec["gold_trajectory"])
                model_steps = _model_steps(events)
                divergence = _first_divergence(gold_steps, model_steps)
                if divergence is None:
                    skipped += 1
                    continue
                messages: List[Dict[str, Any]] = [
                    {"role": "system", "content": task.system_prompt()},
                    {"role": "user", "content": spec["question"]},
                ]
                for assistant_message, observation in gold_steps[:divergence]:
                    messages.append(assistant_message)
                    if observation is not None:
                        messages.append(observation)
                chosen = gold_steps[divergence][0]
                rejected = model_steps[divergence][0]
                row = {
                    "messages": messages,
                    "chosen": chosen,
                    "rejected": rejected,
                    "task_id": events[0].task_id,
                    "success": success,
                    "run_id": run_id,
                    "divergence_step": divergence,
                }
                key = _pair_key(row)
                if key in seen:  # 同题同错,已收录
                    skipped += 1
                    continue
                seen.add(key)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
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
