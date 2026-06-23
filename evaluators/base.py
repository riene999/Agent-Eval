"""评测器(Evaluator)抽象与共用工具。

评测器是任务跑完后对轨迹做的可选后处理(LLM-as-judge 打分、错误归因)。它们自身的
LLM 调用走代理、用独立 role 标记(judge/attributor),既被记录、又不计入被测 agent 的
token。client 可注入,便于离线验证编排逻辑(不访问真实 LLM)。
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from proxy.recorder import Event, make_client


def extract_json(text: str) -> Dict[str, Any]:
    """从模型输出里尽量稳地抠出 JSON 对象;失败则返回带 _parse_error 的占位。"""
    if not text:
        return {"_parse_error": True, "raw": ""}
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        candidate = text[start : end + 1] if start != -1 and end > start else text
    try:
        return json.loads(candidate)
    except (ValueError, TypeError):
        return {"_parse_error": True, "raw": text[:2000]}


class Evaluator(ABC):
    name: str
    role: str  # 走代理时的 X-Llm-Role,用于和被测 agent 区分

    def __init__(
        self, model: Optional[str] = None, temperature: float = 0.0, client: Any = None
    ) -> None:
        self.model = model or os.getenv("AGENT_MODEL", "deepseek-chat")
        self.temperature = temperature
        self._client = client

    def _chat_json(self, system: str, user: str) -> Dict[str, Any]:
        client = self._client or make_client(self.role)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
        )
        return extract_json(resp.choices[0].message.content or "")

    @abstractmethod
    def evaluate(
        self, task: Any, events: List[Event], verdict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """对一次运行做评测,返回结构化结果(会作为事件 data 落库)。"""
        raise NotImplementedError
