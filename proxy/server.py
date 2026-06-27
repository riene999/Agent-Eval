"""OpenAI 兼容的本地反向代理。

接收 Agent 用 OpenAI 协议发来的 /v1/chat/completions 请求,转发到真实上游
(DeepSeek/OpenAI 等),透传响应,并把每次调用记为一条 llm_call 轨迹事件。
通过请求 header X-Run-Id / X-Agent-Id / X-Task-Id 关联到具体轨迹文件,X-Llm-Role
区分被测 agent 与用户模拟器。--mock 模式不访问上游,返回可计 token 的假响应,
用于无 key 的离线联调。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse

from proxy.recorder import PROJECT_ROOT, append_event, blobify, load_env

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
    # 完整请求体:逐条消息 + tools schema 大则外置为 blob,其余字段(model/温度等)内联
    request = dict(payload)
    if isinstance(request.get("messages"), list):
        request["messages"] = [blobify(m) for m in request["messages"]]
    if "tools" in request:
        request["tools"] = blobify(request["tools"])
    append_event(
        agent_id=trace["agent_id"],
        task_id=trace["task_id"],
        run_id=trace["run_id"],
        event_type="llm_call",
        data={
            "role": trace["role"],
            "model": payload.get("model"),
            "request": request,
            "response": blobify(response_body),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "latency_ms": round(latency_ms, 3),
        },
    )


def _load_routes() -> Dict[str, Any]:
    """读 models.json:{模型名: {base_url, api_key}}。不存在/坏了则返回空(走 .env 回退)。"""
    path = Path(os.getenv("MODELS_CONFIG") or (PROJECT_ROOT / "models.json"))
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("models.json 解析失败,本次回退到 .env 的 UPSTREAM_*")
        return {}


def _resolve_upstream(model: str) -> tuple[str, str]:
    """按模型名路由到 (base_url, api_key);models.json 未配则回退 .env 的 UPSTREAM_*。"""
    route = _load_routes().get(model)
    if route:
        return route.get("base_url", "").rstrip("/"), route.get("api_key", "")
    return (
        os.environ.get("UPSTREAM_BASE_URL", "https://api.deepseek.com").rstrip("/"),
        os.environ.get("UPSTREAM_API_KEY", ""),
    )


def _forward_upstream(payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    base, api_key = _resolve_upstream(payload.get("model", ""))
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=_UPSTREAM_TIMEOUT_S) as client:
            resp = client.post(url, json=payload, headers=headers)
    except Exception as e:
        # 连不上上游(DNS/TLS/超时等):返回清晰的 502,而不是裸 500
        logger.warning("连接上游失败 %s: %r", url, e)
        return 502, {
            "error": {
                "message": f"无法连接上游 {url}: {e!r}",
                "type": "upstream_connection_error",
            }
        }
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
        seed = request.headers.get("X-Seed")
        if seed is not None:  # 由运行上下文注入,用于可复现/多试验
            try:
                payload["seed"] = int(seed)
            except ValueError:
                pass
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
