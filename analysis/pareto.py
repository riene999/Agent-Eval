"""ASCII Pareto 前沿:对比多个配置在 (token 成本, 准确率) 上的权衡,零依赖。

每个 --point "标签=轨迹glob" 聚合成一个点(平均准确率 vs 平均 token),画文本散点并
高亮 Pareto 前沿(没被任何配置"更准且更省"全面碾压的点)。需 ≥2 个配置才有对比意义。
"""

from __future__ import annotations

import argparse
import glob
from typing import List, Optional, Set, Tuple

from analysis.metrics import _load_file, compute_metrics

# 一个配置点:(标签, 平均准确率, 平均 token, 样本数)
Point = Tuple[str, float, float, int]


def _aggregate(label: str, pattern: str) -> Optional[Point]:
    accs: List[float] = []
    toks: List[float] = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        events = _load_file(path)
        if events:
            m = compute_metrics(events)
            accs.append(m["accuracy"])
            toks.append(m["total_tokens"])
    if not accs:
        return None
    return label, sum(accs) / len(accs), sum(toks) / len(toks), len(accs)


def _frontier(points: List[Point]) -> Set[int]:
    """非支配集:想要 高准确率 + 低 token。p 被 q 支配 = q 不差且至少一维更好。"""
    front: Set[int] = set()
    for i, (_, ai, ti, _) in enumerate(points):
        dominated = any(
            j != i and aj >= ai and tj <= ti and (aj > ai or tj < ti)
            for j, (_, aj, tj, _) in enumerate(points)
        )
        if not dominated:
            front.add(i)
    return front


def _scatter(points: List[Point], front: Set[int], width: int = 48, height: int = 14) -> str:
    xs = [t for _, _, t, _ in points]
    ys = [a for _, a, _, _ in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xspan = (xmax - xmin) or 1.0
    yspan = (ymax - ymin) or 1.0

    grid = [[" "] * width for _ in range(height)]
    for i, (_, a, t, _) in enumerate(points):
        col = int((t - xmin) / xspan * (width - 1))
        row = height - 1 - int((a - ymin) / yspan * (height - 1))
        grid[row][col] = chr(ord("A") + i)  # 多点重叠时后者覆盖,够用

    out = ["准确率 ↑(越高越好)"]
    for r in range(height):
        out.append("|" + "".join(grid[r]))
    out.append("+" + "-" * width + "→ token(越左越省)")
    return "\n".join(out)


def render(points: List[Point]) -> str:
    front = _frontier(points)
    lines = [_scatter(points, front), "", "图例(★ = Pareto 前沿,即没被任何配置又准又省地碾压):"]
    for i, (label, a, t, n) in enumerate(points):
        star = " ★" if i in front else ""
        lines.append(f"  {chr(ord('A') + i)}  {label:<16} accuracy={a:.2f}  token={t:,.0f}  (n={n}){star}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="ASCII Pareto 前沿对比")
    parser.add_argument(
        "--point", action="append", required=True,
        help='形如 "标签=轨迹glob",可多次;如 "ReAct=trajectories/react_agent_v1/*/run.jsonl"',
    )
    args = parser.parse_args(argv)

    points: List[Point] = []
    for spec in args.point:
        if "=" not in spec:
            raise SystemExit(f'--point 形如 "标签=glob",得到: {spec!r}')
        label, pattern = spec.split("=", 1)
        pt = _aggregate(label.strip(), pattern.strip())
        if pt is None:
            print(f"(跳过:{label.strip()} 无匹配轨迹)")
        else:
            points.append(pt)
    if not points:
        print("无数据可画")
        return
    print(render(points))


if __name__ == "__main__":
    main()
