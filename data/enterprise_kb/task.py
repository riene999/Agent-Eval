"""企业知识库问答任务:单轮(问→查工具→答),无状态。

每条数据 = question + gold_trajectory(正确轨迹,含 final 答案)+ 可选 reference_outputs。
judge 默认用 reference_outputs 关键词判答案对错(可换 --llm-judge),并对 gold_trajectory 的
工具步算选工具/参数正确率;gold_trajectory 一并留存,供后续 SFT/DPO 数据导出当 chosen。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from proxy.recorder import Event, rehydrate
from tasks.base import Task

from .tools import TOOLS

_TASKS_FILE = Path(__file__).resolve().parent / "tasks.jsonl"

_SYSTEM = (
    "你是企业知识助手,覆盖 HR、IT、财务等域。必须先用提供的工具查证,再据查到的内容回答;"
    "不得编造制度、数字或流程。回答要具体(给出数字/步骤/依据),无法查到时如实说明。"
)


class EnterpriseKBTask(Task):
    def __init__(self, spec: Dict[str, Any]) -> None:
        self.task_id = spec["id"]
        self.domain = spec.get("domain")
        self._question = spec["question"]
        self.gold_trajectory: List[Dict[str, Any]] = spec.get("gold_trajectory", [])
        self._reference_outputs: List[str] = spec.get("reference_outputs", [])
        self.reference_path_length = spec.get("optimal_steps")

    def get_prompt(self) -> str:
        return self._question

    def get_tools(self) -> List[Callable[..., Any]]:
        return TOOLS

    def system_prompt(self) -> str:
        return _SYSTEM

    def goal_text(self) -> str:
        return self._question

    def reference_summary(self) -> Optional[str]:
        # 把 gold 轨迹渲染成"标准路径"文字,供错误归因参考
        steps = []
        for s in self.gold_trajectory:
            if "tool" in s:
                steps.append(f"调用 {s['tool']}({s.get('args', {})})")
            elif "say" in s:
                steps.append(f"对用户说:{s['say']}")
            elif "final" in s:
                steps.append(f"最终回答:{s['final']}")
        return "标准路径:\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(steps)) if steps else None

    def judge(self, final_output: str, trajectory: List[Event]) -> Dict[str, Any]:
        # 1) 答案正确性(默认关键词;无 reference_outputs 则提示改用 LLM 判定)
        if self._reference_outputs:
            hay = final_output.lower().replace(",", "").replace(",", "")
            missing = [o for o in self._reference_outputs if o.lower() not in hay]
            answer_ok = not missing
            reason = "answer_ok" if answer_ok else f"缺关键信息 {missing}"
        else:
            answer_ok = False
            missing = []
            reason = "未提供 reference_outputs,请用 --llm-judge 或补充"

        # 2) 路径保真:对 gold_trajectory 的工具步比 选工具 / 参数正确率
        gold = [(s["tool"], s.get("args", {})) for s in self.gold_trajectory if "tool" in s]
        agent_calls = [
            (e.data.get("tool_name"), rehydrate(e.data.get("args")))
            for e in trajectory
            if e.event_type == "tool_call"
        ]
        agent_names = {n for n, _ in agent_calls}

        def _key(d: Any) -> str:
            try:
                return json.dumps(d, sort_keys=True, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(d)

        tool_selection: Optional[float] = None
        arg_correctness: Optional[float] = None
        if gold:
            hit = [g for g in gold if g[0] in agent_names]
            tool_selection = len(hit) / len(gold)
            exact = sum(1 for gn, ga in gold if any(n == gn and _key(a) == _key(ga) for n, a in agent_calls))
            arg_correctness = (exact / len(hit)) if hit else 0.0

        return {
            "success": bool(answer_ok),
            "score": 1.0 if answer_ok else 0.0,
            "reason": reason,
            "tool_selection": tool_selection,
            "arg_correctness": arg_correctness,
        }


def _load_specs() -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {}
    if _TASKS_FILE.exists():
        for line in _TASKS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                spec = json.loads(line)
                specs[spec["id"]] = spec
    return specs


def load_task(task_id: str) -> EnterpriseKBTask:
    specs = _load_specs()
    if task_id not in specs:
        raise SystemExit(f"enterprise_kb 无此任务: {task_id!r}(共 {len(specs)} 条)")
    return EnterpriseKBTask(specs[task_id])
