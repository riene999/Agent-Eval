"""Plan-Solve Agent:先规划、后执行的手写链路(自包含,不依赖其它 agent)。

策略 = 两段式:先让模型基于政策与开场请求产出一份步骤计划(一次 LLM 调用,不调工具、
不回复用户),再把计划注入 system,进入 function-calling 执行循环。与 ReAct 的差异只在
"多了显式规划阶段";其余条件(模型、工具、任务钩子、温度、max_steps、代理记录)保持一致,
以便三轴对比只反映策略差异。本文件刻意自带全部辅助函数与循环,不与 ReAct 共享代码。
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import typing
from typing import Any, Callable, Dict, List, Optional

from agents.base import BaseAgent
from proxy.recorder import make_client
from tasks.base import Task

logger = logging.getLogger(__name__)

_PYTYPE_TO_JSON = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    dict: "object",
}

_PLAN_PROMPT = (
    "下面是用户的开场请求。请先**不要**调用任何工具、也不要回复用户,"
    "只用要点列出完成该任务的步骤计划(考虑需要先核实什么、查什么、改什么、如何确认):\n\n{request}"
)


def _derive_schema(fn: Callable[..., Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    props: Dict[str, Any] = {}
    required: List[str] = []
    for name, param in sig.parameters.items():
        if name == "self" or param.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            continue
        props[name] = {"type": _PYTYPE_TO_JSON.get(hints.get(name), "string")}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": (inspect.getdoc(fn) or "").strip(),
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


def _tool_schema(fn: Callable[..., Any]) -> Dict[str, Any]:
    return getattr(fn, "_openai_tool_schema", None) or _derive_schema(fn)


def _assistant_message(msg: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return out


def _to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(result)


class PlanSolveAgent(BaseAgent):
    agent_id = "plan_solve_v1"

    def __init__(
        self,
        model: Optional[str] = None,
        max_steps: Optional[int] = None,
        temperature: float = 0.0,
        client: Any = None,
    ) -> None:
        self.model = model or os.getenv("AGENT_MODEL", "deepseek-chat")
        self.max_steps = max_steps or int(os.getenv("AGENT_MAX_STEPS", "30"))
        self.temperature = temperature
        self._client = client

    def run(self, task: Task, run_id: str) -> Dict[str, Any]:
        client = self._client or make_client("agent")
        tools = task.get_tools()
        tool_map = {t.__name__: t for t in tools}
        tool_schemas = [_tool_schema(t) for t in tools]
        opening = task.get_prompt()

        # 规划阶段:一次 LLM 调用产出步骤计划(不调工具、不回复用户)
        plan_resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": task.system_prompt()},
                {"role": "user", "content": _PLAN_PROMPT.format(request=opening)},
            ],
        )
        plan = plan_resp.choices[0].message.content or ""
        logger.info("已生成计划(%d 字)", len(plan))

        # 执行阶段:把计划注入 system,进入 function-calling 循环
        system = (
            f"{task.system_prompt()}\n\n# 你已制定的计划\n{plan}\n\n"
            "请按计划逐步执行;若计划与政策或客户实际诉求冲突,以后者为准。"
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": opening},
        ]

        for step in range(self.max_steps):
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            if tool_schemas:
                kwargs["tools"] = tool_schemas
                kwargs["tool_choice"] = "auto"
            response = client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            messages.append(_assistant_message(msg))

            if not msg.tool_calls:
                text = msg.content or ""
                user_reply = task.user_turn(text)
                if user_reply is None:
                    logger.info("第 %d 步:得到最终答案", step)
                    return {"output": text}
                if "###STOP###" in user_reply:
                    logger.info("第 %d 步:用户结束对话", step)
                    return {"output": text}
                messages.append({"role": "user", "content": user_reply})
                continue

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                logger.info("第 %d 步:调用工具 %s args=%s", step, name, args)
                fn = tool_map.get(name)
                if fn is None:
                    observation = f"Error: 未知工具 {name}"
                else:
                    try:
                        observation = _to_text(fn(**args))
                    except Exception as e:
                        observation = f"Error: {e!r}"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": observation})

        logger.warning("达到 max_steps=%d 仍未结束", self.max_steps)
        return {"output": "(达到最大步数,未给出最终回答)"}
