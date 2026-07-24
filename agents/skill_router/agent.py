"""Skill Router Agent:先选择能力包，再在该能力包内执行工具调用。

路由和执行都是手写的 OpenAI function-calling 流程。该 Agent 不复用 ReAct 或
Plan-Solve 的循环，使链路实现保持独立；单 Skill 与多 Skill 仅改变候选能力集合。
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import typing
from typing import Any, Callable, Dict, List, Optional

from agents.base import BaseAgent
from proxy.recorder import append_event, current_context, make_client
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


def _derive_schema(fn: Callable[..., Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, param in sig.parameters.items():
        if name == "self" or param.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            continue
        properties[name] = {"type": _PYTYPE_TO_JSON.get(hints.get(name), "string")}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": (inspect.getdoc(fn) or "").strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _tool_schema(fn: Callable[..., Any]) -> Dict[str, Any]:
    return getattr(fn, "_openai_tool_schema", None) or _derive_schema(fn)


def _assistant_message(message: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message.tool_calls
        ]
    return result


def _to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(result)


class SkillRouterAgent(BaseAgent):
    agent_id = "skill_router_v1"

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

    def _route(self, task: Task, opening: str, client: Any) -> tuple[Optional[str], str]:
        specs = getattr(task, "skill_specs", None)
        if not specs:
            raise TypeError("SkillRouterAgent 只能运行 SkillEvalTask")
        choices = "\n".join(
            f"- {spec.skill_id}: {spec.name}；{spec.description}" for spec in specs
        )
        enum = [spec.skill_id for spec in specs] + ["none"]
        route_tool = {
            "type": "function",
            "function": {
                "name": "select_skill",
                "description": "选择最适合当前问题的一个 Skill；都不适用时选择 none。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string", "enum": enum},
                        "reason": {"type": "string"},
                    },
                    "required": ["skill_id", "reason"],
                },
            },
        }
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是能力路由器。只根据能力边界选择一个 Skill，不回答问题。"
                        "没有任何能力适用时必须选择 none。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"可用 Skill:\n{choices}\n\n用户问题:\n{opening}",
                },
            ],
            tools=[route_tool],
            tool_choice="auto",
        )
        message = response.choices[0].message
        selected: Optional[str] = None
        reason = message.content or ""
        for call in message.tool_calls or []:
            if call.function.name != "select_skill":
                continue
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            candidate = arguments.get("skill_id")
            reason = str(arguments.get("reason") or reason)
            if candidate in enum:
                selected = None if candidate == "none" else candidate
                break
        if selected is None and reason:
            for candidate in enum:
                if candidate != "none" and candidate in reason:
                    selected = candidate
                    break
        return selected, reason

    def run(self, task: Task, run_id: str) -> Dict[str, Any]:
        client = self._client or make_client("agent")
        opening = task.get_prompt()
        selected, reason = self._route(task, opening, client)
        context = current_context()
        if context:
            append_event(
                agent_id=context.agent_id,
                task_id=context.task_id,
                run_id=context.run_id,
                event_type="skill_route",
                data={
                    "available_skills": [
                        spec.skill_id for spec in getattr(task, "skill_specs", [])
                    ],
                    "selected_skill": selected,
                    "reason": reason,
                },
            )

        selected_spec = next(
            (
                spec
                for spec in getattr(task, "skill_specs", [])
                if spec.skill_id == selected
            ),
            None,
        )
        tools = task.tools_for_skill(selected) if selected else []  # type: ignore[attr-defined]
        tool_map = {tool.__name__: tool for tool in tools}
        schemas = [_tool_schema(tool) for tool in tools]
        if selected_spec:
            skill_context = (
                f"\n\n# 当前启用的 Skill\n{selected_spec.name} ({selected_spec.skill_id})\n"
                f"{selected_spec.instructions}\n只能使用该 Skill 提供的工具。"
            )
        else:
            skill_context = (
                "\n\n# 能力边界\n当前可用 Skill 均不适合这个问题。"
                "不要调用工具，直接说明当前能力无法处理该请求。"
            )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": task.system_prompt() + skill_context},
            {"role": "user", "content": opening},
        ]

        for step in range(self.max_steps):
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            if schemas:
                kwargs["tools"] = schemas
                kwargs["tool_choice"] = "auto"
            response = client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            messages.append(_assistant_message(message))
            if not message.tool_calls:
                text = message.content or ""
                user_reply = task.user_turn(text)
                if user_reply is None or "###STOP###" in (user_reply or ""):
                    return {"output": text}
                messages.append({"role": "user", "content": user_reply})
                continue
            for call in message.tool_calls:
                name = call.function.name
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                tool = tool_map.get(name)
                if tool is None:
                    observation = f"Error: 未知或越权工具 {name}"
                else:
                    try:
                        observation = _to_text(tool(**arguments))
                    except Exception as exc:
                        observation = f"Error: {exc!r}"
                logger.info("第 %d 步:Skill %s 调用 %s", step, selected, name)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": observation,
                    }
                )
        return {"output": "(达到最大步数，未给出最终回答)"}
