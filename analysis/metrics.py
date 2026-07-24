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
import math
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from proxy.recorder import Event

# 这些 role 是评测装置(用户模拟器/打分/归因),其 token 不计入"被测 Agent"
_HARNESS_ROLES = {"user_sim", "judge", "attributor"}

# 每 100 万 token 的美元价格(粗略,可被项目根目录 prices.json 覆盖);未知模型成本记 None
_DEFAULT_PRICES = {
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
}


def _prices() -> Dict[str, Dict[str, float]]:
    from proxy.recorder import PROJECT_ROOT

    prices = dict(_DEFAULT_PRICES)
    path = PROJECT_ROOT / "prices.json"
    if path.exists():
        try:
            prices.update(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            pass
    return prices


def _cost(model: Optional[str], in_tokens: int, out_tokens: int) -> Optional[float]:
    p = _prices().get(model or "")
    if not p:
        return None
    return in_tokens / 1e6 * p.get("input", 0.0) + out_tokens / 1e6 * p.get("output", 0.0)


def _percentile(values: List[float], q: float) -> Optional[float]:
    """线性插值百分位;q=0.5 为中位数。空列表返回 None。"""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


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
    agent_in = 0
    agent_out = 0
    model: Optional[str] = None
    latencies: List[float] = []
    call_keys: List[str] = []
    llm_score: Optional[float] = None
    attribution: Optional[Dict[str, Any]] = None
    tool_selection: Optional[float] = None
    arg_correctness: Optional[float] = None
    skill_expected: Optional[str] = None
    skill_gold: Optional[str] = None
    skill_selected: Optional[str] = None
    skill_routing_correct: Optional[bool] = None
    skill_scope_correct: Optional[bool] = None
    skill_case_type: Optional[str] = None
    cross_skill_tool_rate: Optional[float] = None
    for e in events:
        if e.event_type == "llm_call":
            # 只统计被测 Agent,排除评测装置(用户模拟器/judge/归因)
            if e.data.get("role") not in _HARNESS_ROLES:
                pin = int(e.data.get("prompt_tokens", 0) or 0)
                pout = int(e.data.get("completion_tokens", 0) or 0)
                total_tokens += pin + pout
                agent_in += pin
                agent_out += pout
                if model is None:
                    model = e.data.get("model")
                lat = e.data.get("latency_ms")
                if isinstance(lat, (int, float)):
                    latencies.append(float(lat))
        elif e.event_type == "tool_call":
            key = json.dumps(
                [e.data.get("tool_name"), e.data.get("args", {})],
                sort_keys=True,
                ensure_ascii=False,
            )
            call_keys.append(key)
        elif e.event_type == "final_output":
            success = bool(e.data.get("success", False))
            # 这两项由 task 层(对 gold 比对)写入,非 tau 任务为 None
            tool_selection = e.data.get("tool_selection")
            arg_correctness = e.data.get("arg_correctness")
            skill_expected = e.data.get("skill_expected")
            skill_gold = e.data.get("skill_gold")
            skill_selected = e.data.get("skill_selected")
            skill_routing_correct = e.data.get("skill_routing_correct")
            skill_scope_correct = e.data.get("skill_scope_correct")
            skill_case_type = e.data.get("skill_case_type")
            cross_skill_tool_rate = e.data.get("cross_skill_tool_rate")
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
        "latency_p50": _percentile(latencies, 0.5),
        "latency_p95": _percentile(latencies, 0.95),
        "cost_usd": _cost(model, agent_in, agent_out),
        "tool_selection": tool_selection,
        "arg_correctness": arg_correctness,
        "llm_score": llm_score,
        "attribution": attribution,
        "skill_expected": skill_expected,
        "skill_gold": skill_gold,
        "skill_selected": skill_selected,
        "skill_routing_correct": skill_routing_correct,
        "skill_scope_correct": skill_scope_correct,
        "skill_case_type": skill_case_type,
        "cross_skill_tool_rate": cross_skill_tool_rate,
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


def _failure_counter(rows: List[Dict[str, Any]]) -> "Counter[str]":
    """统计各 error_category 出现次数(来自归因事件,无归因则空)。"""
    counter: "Counter[str]" = Counter()
    for r in rows:
        a = r.get("attribution")
        if a and a.get("error_category"):
            counter[a["error_category"]] += 1
    return counter


def _failure_dist_section(rows: List[Dict[str, Any]]) -> List[str]:
    counter = _failure_counter(rows)
    if not counter:
        return []
    out = ["## 失败类型分布(按归因)", "", "| 错误类别 | 次数 |", "| --- | ---: |"]
    out += [f"| {cat} | {cnt} |" for cat, cnt in counter.most_common()]
    out.append("")
    return out


def report_json(meta: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """单次/批量结果的机器可读 JSON:meta + 汇总 + 每题指标。"""
    n = len(rows)
    summary: Dict[str, Any] = {"n": n}
    if n:
        summary.update({
            "accuracy": sum(r["accuracy"] for r in rows) / n,
            "avg_total_tokens": sum(r["total_tokens"] for r in rows) / n,
            "avg_tool_calls": sum(r["tool_call_count"] for r in rows) / n,
            "failure_distribution": dict(_failure_counter(rows)),
        })
        from analysis.skill_metrics import summarize_skill_rows

        summary.update(summarize_skill_rows(rows))
    return {"meta": meta, "summary": summary, "tasks": rows}


def trials_json(
    meta: Dict[str, Any], per_task: List[Tuple[str, List[Dict[str, Any]]]]
) -> Dict[str, Any]:
    """多试验结果的机器可读 JSON:每题 pass@k + 全部试验明细。"""
    tasks = []
    for task_id, rows in per_task:
        n = len(rows)
        c = sum(r["accuracy"] for r in rows)
        tasks.append({
            "task_id": task_id, "n": n,
            "pass_at_1": pass_at_k(n, c, 1), "pass_at_n": pass_at_k(n, c, n),
            "trials": rows,
        })
    flat = [r for _, rows in per_task for r in rows]
    p1 = [t["pass_at_1"] for t in tasks]
    pn = [t["pass_at_n"] for t in tasks]
    summary = {
        "tasks": len(tasks),
        "avg_pass_at_1": (sum(p1) / len(p1)) if p1 else None,
        "avg_pass_at_n": (sum(pn) / len(pn)) if pn else None,
        "failure_distribution": dict(_failure_counter(flat)),
    }
    from analysis.skill_metrics import summarize_skill_rows

    summary.update(summarize_skill_rows(flat))
    return {"meta": meta, "summary": summary, "tasks": tasks}


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
    def _avg(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    lines += [
        "## 总览",
        "",
        f"- 准确率 {_light(acc, 0.8, 0.5, True)} **{acc:.0%}**（{n_pass}/{n} 通过)",
        f"- 平均冗余率 {_light(avg_red, 0.05, 0.20, False)} {avg_red:.1%}",
        f"- 平均 token/题 {avg_tok:,.0f} · 平均工具/题 {avg_calls:.1f} "
        f"（成本轴:越低越省,需跨配置对比才见高下)",
    ]
    p50, p95, costv = _avg("latency_p50"), _avg("latency_p95"), _avg("cost_usd")
    tsel, argc = _avg("tool_selection"), _avg("arg_correctness")
    if p50 is not None:
        lines.append(f"- 平均延迟 p50 {p50:.0f}ms · p95 {(p95 or 0):.0f}ms（该并发下观测)")
    if costv is not None:
        lines.append(f"- 平均成本/题 ${costv:.4f}")
    if tsel is not None:
        lines.append(f"- 平均选工具准确率 {tsel:.0%} · 参数正确率 {(argc or 0):.0%}(对标准答案)")
    from analysis.skill_metrics import summarize_skill_rows

    skill_summary = summarize_skill_rows(rows)
    if skill_summary:
        route = skill_summary.get("routing_accuracy")
        boundary = skill_summary.get("boundary_accuracy")
        confusion = skill_summary.get("skill_confusion_rate")
        lines.append(
            f"- Skill 路由准确率 {(route or 0):.0%}"
            + (
                f" · 边界识别率 {boundary:.0%}"
                if boundary is not None
                else ""
            )
            + (
                f" · Skill 混淆率 {confusion:.0%}"
                if confusion is not None
                else ""
            )
        )
    lines += [
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

    # 性能/成本/路径保真:仅在有数据时出现(旧轨迹/非 tau 自动跳过)
    perf = [r for r in rows if any(
        r.get(k) is not None for k in ("latency_p50", "cost_usd", "tool_selection", "arg_correctness")
    )]
    if perf:
        def _cell(v: Optional[float], kind: str) -> str:
            if v is None:
                return "-"
            return {"ms": f"{v:.0f}", "money": f"${v:.4f}", "pct": f"{v:.0%}"}[kind]

        lines += [
            "## 性能 · 成本 · 路径保真(对标准答案)",
            "",
            "| 题 | p50 ms | p95 ms | cost($) | 选工具 | 参数 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in perf:
            short = r["task_id"].replace("tau_retail_", "")
            lines.append(
                f"| {short} | {_cell(r.get('latency_p50'), 'ms')} | {_cell(r.get('latency_p95'), 'ms')} "
                f"| {_cell(r.get('cost_usd'), 'money')} | {_cell(r.get('tool_selection'), 'pct')} "
                f"| {_cell(r.get('arg_correctness'), 'pct')} |"
            )
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

    skill_rows = [r for r in rows if r.get("skill_case_type") is not None]
    if skill_rows:
        lines += [
            "## Skill 路由与能力边界",
            "",
            "| 题 | 类型 | 应选 Skill | 实际选择 | 路由 | 跨 Skill 工具 |",
            "| --- | --- | --- | --- | :---: | ---: |",
        ]
        for r in skill_rows:
            lines.append(
                f"| {r['task_id']} | {'适用' if r.get('skill_case_type') == 'in_scope' else '范围外'} "
                f"| {r.get('skill_expected') or 'none'} | {r.get('skill_selected') or 'none'} "
                f"| {'✅' if r.get('skill_routing_correct') else '❌'} "
                f"| {(r.get('cross_skill_tool_rate') or 0):.0%} |"
            )
        lines.append("")

    lines += _failure_dist_section(rows)

    lines += [
        "## 图例",
        "- 结果:✅ 通过 / ❌ 失败",
        "- 准确率灯:🟢 ≥80% / 🟡 ≥50% / 🔴 <50%",
        "- token 条:越长越费;🟢🟡🔴 = 本批内相对(便宜/中等/偏贵),成本轴无绝对好坏",
        "- 冗余灯:🟢 ≤5% / 🟡 ≤20% / 🔴 >20%(相同参数重复调同一工具的占比)",
    ]
    return "\n".join(lines)


def pass_at_k(n: int, c: int, k: int) -> float:
    """n 次试验中 c 次成功,随机取 k 次至少一次成功的概率(Codex 无偏估计)。"""
    k = min(k, n)
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def build_trials_report(
    meta: Dict[str, Any], per_task: List[Tuple[str, List[Dict[str, Any]]]]
) -> str:
    """渲染多试验报告:每题 pass@1 / pass@N + token/工具 均值±std,外加总览。"""
    trials = meta.get("trials")
    lines = [f"# 评测报告 {meta.get('run_id', '')}(多试验)", ""]
    info = []
    for label, key in [("agent", "agent_id"), ("模型", "model"), ("split", "split"),
                       ("trials", "trials"), ("seed", "seed"),
                       ("temperature", "temperature"), ("时间", "timestamp")]:
        if meta.get(key) is not None:
            info.append(f"{label}={meta[key]}")
    if info:
        lines += ["> " + " · ".join(str(x) for x in info), ""]

    p1s: List[float] = []
    pNs: List[float] = []
    body: List[str] = []
    for task_id, rows in per_task:
        if not rows:
            continue
        n = len(rows)
        c = sum(r["accuracy"] for r in rows)
        p1, pN = pass_at_k(n, c, 1), pass_at_k(n, c, n)
        p1s.append(p1)
        pNs.append(pN)
        toks = [r["total_tokens"] for r in rows]
        tools = [r["tool_call_count"] for r in rows]
        short = task_id.replace("tau_retail_", "")
        body.append(
            f"| {short} | {c}/{n} | {p1:.0%} | {pN:.0%} | "
            f"{statistics.mean(toks) / 1000:.0f}k ± {statistics.pstdev(toks) / 1000:.1f}k | "
            f"{statistics.mean(tools):.1f} ± {statistics.pstdev(tools):.1f} |"
        )

    lines += ["## 总览", ""]
    if p1s:
        lines += [
            f"- 题数 {len(p1s)} × 每题 {trials} 次",
            f"- 平均 pass@1 **{statistics.mean(p1s):.1%}** · "
            f"平均 pass@{trials} **{statistics.mean(pNs):.1%}**",
        ]
    else:
        lines.append("- (无数据)")
    lines += [
        "",
        "## 每题(多次试验)",
        "",
        "| 题 | 成功 | pass@1 | pass@N | token 均值±std | 工具 均值±std |",
        "| --- | :---: | ---: | ---: | ---: | ---: |",
        *body,
        "",
    ]

    # 跨所有试验的附加维度(开了对应开关才有:LLM 评分 / 性能成本 / 失败归因)
    all_rows = [r for _, rows in per_task for r in rows]

    def _avg(rs: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [x[key] for x in rs if x.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    if any(r.get("llm_score") is not None for r in all_rows):
        lines += ["## LLM 评分(每题均值)", "", "| 题 | LLM 均值 | 成功 |", "| --- | ---: | :---: |"]
        for task_id, rows in per_task:
            s = _avg(rows, "llm_score")
            if s is None:
                continue
            lines.append(
                f"| {task_id.replace('tau_retail_', '')} | {s:.2f} | "
                f"{sum(r['accuracy'] for r in rows)}/{len(rows)} |"
            )
        lines.append("")

    if any(
        r.get(k) is not None
        for r in all_rows
        for k in ("latency_p50", "cost_usd", "tool_selection", "arg_correctness")
    ):
        lines += [
            "## 性能 · 成本 · 路径保真(每题均值)",
            "",
            "| 题 | p50 ms | p95 ms | cost($) | 选工具 | 参数 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for task_id, rows in per_task:
            def _c(key: str, kind: str, rs: List[Dict[str, Any]] = rows) -> str:
                v = _avg(rs, key)
                if v is None:
                    return "-"
                return {"ms": f"{v:.0f}", "money": f"${v:.4f}", "pct": f"{v:.0%}"}[kind]

            lines.append(
                f"| {task_id.replace('tau_retail_', '')} | {_c('latency_p50', 'ms')} | "
                f"{_c('latency_p95', 'ms')} | {_c('cost_usd', 'money')} | "
                f"{_c('tool_selection', 'pct')} | {_c('arg_correctness', 'pct')} |"
            )
        lines.append("")

    attributed = [r for r in all_rows if r.get("attribution")]
    if attributed:
        lines += ["## 失败归因(LLM)", ""]
        for r in attributed:
            a = r["attribution"]
            lines.append(
                f"- `{r['task_id']}` [{r['run_id']}] 第 {a.get('deviation_seq')} 步 "
                f"[{a.get('error_category')}]:{a.get('summary')}"
            )
        lines.append("")

    lines += _failure_dist_section(all_rows)

    lines += [
        "## 说明",
        "- pass@1 = 平均成功率;pass@N = N 次中至少成功一次;std = 跨试验标准差(稳定性)。",
        "- temperature=0 时多样性低、pass@k 可能退化;测稳定性建议 temperature>0。",
    ]
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="聚合轨迹指标为 markdown 表 / JSON")
    parser.add_argument("--glob", default="trajectories/**/*.jsonl", help="轨迹文件通配")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出而非 markdown 表")
    args = parser.parse_args(argv)
    rows = collect(args.glob)
    if not rows:
        print(f"未找到匹配 {args.glob!r} 的轨迹文件")
        return
    if args.json:
        print(json.dumps(report_json({"glob": args.glob}, rows), ensure_ascii=False, indent=2))
    else:
        print(to_markdown(rows))


if __name__ == "__main__":
    main()
