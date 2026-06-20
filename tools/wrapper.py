"""@traced_tool 装饰器:旁路记录工具调用,不改变函数行为。

调用时从 contextvar 读取 run_id/agent_id/task_id,在调用前后各追加一条
tool_call 与 tool_return 事件(tool_return.parent_seq 指向对应 tool_call)。
不在运行上下文中(如单测直接调函数)则完全透明、不记录。
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import time
from typing import Any, Callable, Dict, Tuple, TypeVar

from proxy.recorder import append_event, current_context

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _json_safe(value: Any) -> Any:
    """事件要落 JSON,这里把不可序列化的返回值降级为字符串。"""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _bind_args(fn: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """把实参规整成 {参数名: 值} 的扁平字典;**kwargs 会被摊平。"""
    try:
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        out: Dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name not in bound.arguments:
                continue
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                out.update(bound.arguments[name] or {})
            elif param.kind is inspect.Parameter.VAR_POSITIONAL:
                out[name] = list(bound.arguments[name])
            else:
                out[name] = bound.arguments[name]
        return _json_safe(out)
    except (TypeError, ValueError):
        return {"args": _json_safe(list(args)), "kwargs": _json_safe(dict(kwargs))}


def traced_tool(fn: F) -> F:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx = current_context()
        if ctx is None:
            return fn(*args, **kwargs)

        tool_name = getattr(fn, "__name__", "unknown")
        call_seq = append_event(
            agent_id=ctx.agent_id,
            task_id=ctx.task_id,
            run_id=ctx.run_id,
            event_type="tool_call",
            data={"tool_name": tool_name, "args": _bind_args(fn, args, kwargs)},
        )
        start = time.perf_counter()
        error = None
        result: Any = None
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as e:  # 记录异常后原样抛出,保持行为透明
            error = repr(e)
            raise
        finally:
            append_event(
                agent_id=ctx.agent_id,
                task_id=ctx.task_id,
                run_id=ctx.run_id,
                event_type="tool_return",
                data={
                    "tool_name": tool_name,
                    "result": _json_safe(result),
                    "error": error,
                    "latency_ms": round((time.perf_counter() - start) * 1000.0, 3),
                },
                parent_seq=call_seq,
            )

    return wrapper  # type: ignore[return-value]
