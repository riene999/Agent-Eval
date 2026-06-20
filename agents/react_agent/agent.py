"""ReAct Agent:手写的 OpenAI function-calling 循环。

每一步让模型基于对话历史决定"调用工具"还是"给出最终答案":有 tool_calls 就逐个
执行(工具已被 @traced_tool 包装,自动记录),把结果作为 tool 消息回灌;模型不再
调用工具时,其文本内容即最终答案。超过 max_steps 则强制结束。不依赖任何 Agent 框架。
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


def _derive_schema(fn: Callable[..., Any]) -> Dict[str, Any]:
    """没有自带 schema 的工具,从签名 + 类型注解 + docstring 推导 OpenAI schema。"""
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
        jtype = _PYTYPE_TO_JSON.get(hints.get(name), "string")
        props[name] = {"type": jtype}
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
    """把模型返回的 assistant 消息转回可回灌 API 的 dict(含 tool_calls)。"""
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


class ReactAgent(BaseAgent):
    agent_id = "react_agent_v1"

    def __init__(
        self,
        model: Optional[str] = None,
        max_steps: Optional[int] = None,
        temperature: float = 0.0,
        client: Any = None,
    ) -> None:
        # client 可注入(便于离线测试循环机制);默认用指向代理的追踪客户端。
        self.model = model or os.getenv("AGENT_MODEL", "deepseek-chat")
        self.max_steps = max_steps or int(os.getenv("AGENT_MAX_STEPS", "30"))
        self.temperature = temperature
        self._client = client

    def run(self, task: Task, run_id: str) -> Dict[str, Any]:
        tools = task.get_tools()
        tool_map = {t.__name__: t for t in tools}
        tool_schemas = [_tool_schema(t) for t in tools]
        client = self._client or make_client("agent")

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": task.system_prompt()},
            {"role": "user", "content": task.get_prompt()},
        ]

        final_output = ""
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
                # 没有工具调用:这段文本要么发给用户(对话任务),要么就是最终答案
                text = msg.content or ""
                user_reply = task.user_turn(text)
                if user_reply is None:
                    final_output = text
                    logger.info("第 %d 步:得到最终答案", step)
                    break
                if "###STOP###" in user_reply:
                    final_output = text
                    logger.info("第 %d 步:用户结束对话", step)
                    break
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
                    except Exception as e:  # 工具异常回灌给模型,让它自行纠错
                        observation = f"Error: {e!r}"
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": observation}
                )
        else:
            final_output = final_output or "(达到最大步数,未给出最终回答)"
            logger.warning("达到 max_steps=%d 仍未结束", self.max_steps)

        return {"output": final_output}
