"""把事件轨迹 rehydrate 成带 seq 步号的可读转录,喂给评测器 LLM。

只渲染 Agent 自身的 4 类事件;llm_judge/attribution 等评测事件会被忽略(避免自指)。
长内容截断以控制 prompt 体积,但保留 seq 步号——归因要靠它定位"第几步偏离"。
"""

from __future__ import annotations

from typing import Any, List

from proxy.recorder import Event, rehydrate

_MAX = 600


def _clip(value: Any, n: int = _MAX) -> str:
    s = value if isinstance(value, str) else str(value)
    return s if len(s) <= n else s[:n] + f"…(截断,共 {len(s)} 字)"


def render(events: List[Event]) -> str:
    lines: List[str] = []
    for e in events:
        d = rehydrate(e.data)
        if e.event_type == "llm_call":
            choices = (d.get("response") or {}).get("choices") or [{}]
            content = (choices[0].get("message") or {}).get("content")
            if not content:
                continue
            who = "客户" if d.get("role") == "user_sim" else "Agent对客户说"
            lines.append(f"[{e.seq}] {who}: {_clip(content)}")
        elif e.event_type == "tool_call":
            lines.append(f"[{e.seq}] 调用工具 {d.get('tool_name')}({d.get('args')})")
        elif e.event_type == "tool_return":
            tag = "错误" if d.get("error") else "返回"
            lines.append(f"[{e.seq}]   └─{tag}: {_clip(d.get('error') or d.get('result'))}")
        elif e.event_type == "final_output":
            lines.append(
                f"[{e.seq}] 最终输出(success={d.get('success')}): {_clip(d.get('output'))}"
            )
    return "\n".join(lines)
