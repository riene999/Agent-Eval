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
from typing import Any, Dict, List, Optional

from proxy.recorder import Event

# 这些 role 是评测装置(用户模拟器/打分/归因),其 token 不计入"被测 Agent"
_HARNESS_ROLES = {"user_sim", "judge", "attributor"}


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
    llm_score: Optional[float] = None
    attribution: Optional[Dict[str, Any]] = None
    for e in events:
        if e.event_type == "llm_call":
            # 只统计被测 Agent 的 token,排除评测装置(用户模拟器/judge/归因)
            if e.data.get("role") not in _HARNESS_ROLES:
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
        elif e.event_type == "llm_judge":  # 可选事件,旧轨迹没有则保持 None
            llm_score = e.data.get("overall")
        elif e.event_type == "attribution":
            attribution = e.data

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
        "llm_score": llm_score,
        "attribution": attribution,
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


def _bar(value: float, vmax: float, width: int = 5) -> str:
    """把数值画成定宽的实心/空心方块条,便于一眼比大小。"""
    if vmax <= 0:
        return "▱" * width
    n = min(width, max(1, round(value / vmax * width)))
    return "▰" * n + "▱" * (width - n)


def _light(value: float, good: float, ok: float, higher_better: bool) -> str:
    """按阈值给红绿灯:higher_better 决定方向。"""
    if higher_better:
        return "🟢" if value >= good else ("🟡" if value >= ok else "🔴")
    return "🟢" if value <= good else ("🟡" if value <= ok else "🔴")


def build_report(meta: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    """渲染一份"外行也能一眼看好坏"的 markdown 报告:红绿灯总览 + 每题表 + 图例。"""
    title = f"# 评测报告 {meta.get('run_id', '')}"
    if not rows:
        return f"{title}\n\n(无数据)\n"

    n = len(rows)
    n_pass = sum(r["accuracy"] for r in rows)
    acc = n_pass / n
    avg_tok = sum(r["total_tokens"] for r in rows) / n
    avg_calls = sum(r["tool_call_count"] for r in rows) / n
    avg_red = sum(r["redundant_call_rate"] for r in rows) / n

    info = []
    for label, key in [("agent", "agent_id"), ("模型", "model"), ("split", "split"),
                       ("题数", "count"), ("时间", "timestamp")]:
        if meta.get(key) is not None:
            info.append(f"{label}={meta[key]}")

    lines = [title, ""]
    if info:
        lines += ["> " + " · ".join(info), ""]
    lines += [
        "## 总览",
        "",
        f"- 准确率 {_light(acc, 0.8, 0.5, True)} **{acc:.0%}**（{n_pass}/{n} 通过)",
        f"- 平均冗余率 {_light(avg_red, 0.05, 0.20, False)} {avg_red:.1%}",
        f"- 平均 token/题 {avg_tok:,.0f} · 平均工具/题 {avg_calls:.1f} "
        f"（成本轴:越低越省,需跨配置对比才见高下)",
        "",
        "## 每题指标",
        "",
        "| 题 | 结果 | token(本批相对) | 工具 | 冗余 |",
        "| --- | :---: | --- | ---: | :---: |",
    ]

    tokens = [r["total_tokens"] for r in rows]
    tmax = max(tokens) or 1
    lo, span = min(tokens), (max(tokens) - min(tokens)) or 1
    for r in rows:
        tok = r["total_tokens"]
        res = "✅" if r["accuracy"] else "❌"
        tok_light = _light(tok, lo + span / 3, lo + 2 * span / 3, False)
        red = r["redundant_call_rate"]
        red_light = _light(red, 0.05, 0.20, False)
        short = r["task_id"].replace("tau_retail_", "")
        lines.append(
            f"| {short} | {res} | {tok_light} {_bar(tok, tmax)} {tok / 1000:.0f}k "
            f"| {r['tool_call_count']} | {red_light} {red:.0%} |"
        )
    lines.append("")

    fails = [(tid, v) for tid, v in meta.get("per_task", []) if not v.get("success")]
    if fails:
        lines += ["## 失败的题(规则判分原因)", ""]
        lines += [f"- `{tid}`: {v.get('reason', '')}" for tid, v in fails]
        lines.append("")

    # 以下两段仅在开启了对应评测时出现(旧轨迹/未开启则自动跳过)
    judged = [r for r in rows if r.get("llm_score") is not None]
    if judged:
        lines += ["## LLM 评分", "", "| 题 | LLM 总分 | 规则 |", "| --- | ---: | :---: |"]
        for r in judged:
            short = r["task_id"].replace("tau_retail_", "")
            lines.append(f"| {short} | {r['llm_score']:.2f} | {'✅' if r['accuracy'] else '❌'} |")
        lines.append("")

    attributed = [r for r in rows if r.get("attribution")]
    if attributed:
        lines += ["## 失败归因(LLM)", ""]
        for r in attributed:
            a = r["attribution"]
            lines.append(
                f"- `{r['task_id']}` 第 {a.get('deviation_seq')} 步 "
                f"[{a.get('error_category')}] 信心 {a.get('confidence')}:{a.get('summary')}"
            )
            if a.get("fix_suggestion"):
                lines.append(f"    建议:{a['fix_suggestion']}")
        lines.append("")

    lines += [
        "## 图例",
        "- 结果:✅ 通过 / ❌ 失败",
        "- 准确率灯:🟢 ≥80% / 🟡 ≥50% / 🔴 <50%",
        "- token 条:越长越费;🟢🟡🔴 = 本批内相对(便宜/中等/偏贵),成本轴无绝对好坏",
        "- 冗余灯:🟢 ≤5% / 🟡 ≤20% / 🔴 >20%(相同参数重复调同一工具的占比)",
    ]
    return "\n".join(lines)


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
