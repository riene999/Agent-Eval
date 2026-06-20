"""OpenAI 兼容的本地反向代理。

接收 Agent 用 OpenAI 协议发来的 /v1/chat/completions 请求,转发到真实上游
(DeepSeek/OpenAI 等),透传响应,并把每次调用记为一条 llm_call 轨迹事件。
通过请求 header X-Run-Id / X-Agent-Id / X-Task-Id 关联到具体轨迹文件,X-Llm-Role
区分被测 agent 与用户模拟器。--mock 模式不访问上游,返回可计 token 的假响应,
用于无 key 的离线联调。
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any, Dict, Optional

import httpx
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse

from proxy.recorder import append_event, load_env

logger = logging.getLogger(__name__)

# 上游连接较慢的模型也要留足时间
_UPSTREAM_TIMEOUT_S = 120.0


def _estimate_tokens(text: str) -> int:
    # 粗略估算:约 4 字符 1 token,仅供 mock 模式让 token 指标有非零值
    return max(1, len(text) // 4)


def _mock_completion(payload: Dict[str, Any]) -> Dict[str, Any]:
    model = payload.get("model", "mock-model")
    messages = payload.get("messages", [])
    prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
    content = "(mock) 这是离线 mock 代理返回的固定回复。"
    prompt_tokens = _estimate_tokens(str(prompt_chars * "x"))
    completion_tokens = _estimate_tokens(content)
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _trace_headers(request: Request) -> Optional[Dict[str, str]]:
    """从请求 header 提取追踪标识;缺少 run_id 视为不追踪。"""
    run_id = request.headers.get("X-Run-Id")
    if not run_id:
        return None
    return {
        "run_id": run_id,
        "agent_id": request.headers.get("X-Agent-Id", "unknown_agent"),
        "task_id": request.headers.get("X-Task-Id", "unknown_task"),
        "role": request.headers.get("X-Llm-Role", "agent"),
    }


def _record_llm_call(
    trace: Dict[str, str],
    payload: Dict[str, Any],
    response_body: Dict[str, Any],
    latency_ms: float,
) -> None:
    usage = response_body.get("usage") or {}
    choices = response_body.get("choices") or [{}]
    message = choices[0].get("message") if choices else None
    append_event(
        agent_id=trace["agent_id"],
        task_id=trace["task_id"],
        run_id=trace["run_id"],
        event_type="llm_call",
        data={
            "role": trace["role"],
            "model": payload.get("model"),
            "messages": payload.get("messages"),
            "response": message,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "latency_ms": round(latency_ms, 3),
        },
    )


def _forward_upstream(payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    base = os.environ.get("UPSTREAM_BASE_URL", "https://api.deepseek.com").rstrip("/")
    api_key = os.environ.get("UPSTREAM_API_KEY", "")
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=_UPSTREAM_TIMEOUT_S) as client:
        resp = client.post(url, json=payload, headers=headers)
    try:
        body = resp.json()
    except Exception:
        body = {"error": {"message": resp.text, "status": resp.status_code}}
    return resp.status_code, body


def create_app(mock: bool = False) -> FastAPI:
    app = FastAPI(title="agent-eval proxy")
    app.state.mock = mock

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "mock": app.state.mock}

    @app.post("/v1/chat/completions")
    def chat_completions(
        request: Request, payload: Dict[str, Any] = Body(...)
    ) -> JSONResponse:
        trace = _trace_headers(request)
        start = time.perf_counter()
        if app.state.mock:
            status, body = 200, _mock_completion(payload)
        else:
            status, body = _forward_upstream(payload)
        latency_ms = (time.perf_counter() - start) * 1000.0

        if trace is not None:
            try:
                _record_llm_call(trace, payload, body, latency_ms)
            except Exception:  # 记录失败不应影响转发本身
                logger.exception("记录 llm_call 事件失败")
        return JSONResponse(status_code=status, content=body)

    return app


def main() -> None:
    load_env()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="OpenAI 兼容反向代理")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--mock", action="store_true", help="不访问上游,返回假响应")
    args = parser.parse_args()

    import uvicorn

    logger.info("代理启动 host=%s port=%d mock=%s", args.host, args.port, args.mock)
    uvicorn.run(create_app(mock=args.mock), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
