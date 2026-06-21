"""轨迹录制与追踪基础设施。

本模块是整个 harness 的"观测层"单一来源,负责:
- 定义统一的 Event 事件模型(见设计契约 §5.1);
- 解析轨迹文件路径 trajectories/{agent_id}/{task_id}/{run_id}.jsonl;
- 用 contextvar 在 agent 进程内传递 run_id/agent_id/task_id(供 @traced_tool 使用);
- 以"按行追加 + 行号即 seq"的方式跨 proxy/wrapper 两个写入方记录事件;
- 提供指向代理、且自动带追踪 header 的 OpenAI 客户端构造器。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

EventType = Literal["llm_call", "tool_call", "tool_return", "final_output"]

# 项目根目录:本文件位于 <root>/proxy/recorder.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 超过该字节数的事件字段会被外置为内容寻址 blob(见 blobify),其余内联
BLOB_THRESHOLD_BYTES = int(os.getenv("BLOB_THRESHOLD_BYTES", "1024"))


class Event(BaseModel):
    """一条轨迹事件;data 的结构随 event_type 不同而不同(见 §5.1)。"""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    agent_id: str
    task_id: str
    timestamp: str
    event_type: EventType
    seq: int
    parent_seq: Optional[int] = None
    data: Dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 运行上下文:同一进程内(runner + agent + wrapper)用 contextvar 传递标识
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunContext:
    run_id: str
    agent_id: str
    task_id: str


import contextvars

_ctx: "contextvars.ContextVar[Optional[RunContext]]" = contextvars.ContextVar(
    "agent_eval_run_context", default=None
)


@contextmanager
def run_context(run_id: str, agent_id: str, task_id: str) -> Iterator[RunContext]:
    """在 with 块内设置当前运行上下文,退出时自动还原。"""
    ctx = RunContext(run_id=run_id, agent_id=agent_id, task_id=task_id)
    token = _ctx.set(ctx)
    try:
        yield ctx
    finally:
        _ctx.reset(token)


def current_context() -> Optional[RunContext]:
    return _ctx.get()


# --------------------------------------------------------------------------- #
# 路径与时间
# --------------------------------------------------------------------------- #


def traj_root() -> Path:
    """轨迹根目录;两个进程都从同一个环境变量解析以保证写到同一处。"""
    raw = os.getenv("TRAJ_DIR")
    return Path(raw).resolve() if raw else (PROJECT_ROOT / "trajectories")


def trajectory_path(agent_id: str, task_id: str, run_id: str) -> Path:
    return traj_root() / agent_id / task_id / f"{run_id}.jsonl"


def _now_iso() -> str:
    # 形如 2026-06-13T10:00:00.123Z(毫秒精度 + Z 时区后缀)
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


# --------------------------------------------------------------------------- #
# 追加写入:行号即 seq
# --------------------------------------------------------------------------- #
#
# proxy 与 wrapper 是两个独立进程,但同一个 run 内它们严格串行(agent 阻塞等
# 待代理 HTTP 响应时不会同时写工具事件,反之亦然),因此用"读取当前行数 → seq
# = 行数 → 追加一行"即可得到时序递增、无空洞的 seq。进程内用全局锁、跨进程用
# sidecar 锁文件兜底,防止极端竞态。

_GLOBAL_WRITE_LOCK = threading.Lock()
_LOCK_TIMEOUT_S = 10.0


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _GLOBAL_WRITE_LOCK:
        deadline = time.monotonic() + _LOCK_TIMEOUT_S
        fd: Optional[int] = None
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                # 超时则强行接管,避免崩溃进程残留的锁文件造成死等
                if time.monotonic() > deadline:
                    logger.warning("锁文件超时,强制接管: %s", lock_path)
                    try:
                        os.remove(lock_path)
                    except OSError:
                        pass
                    continue
                time.sleep(0.01)
        try:
            yield
        finally:
            if fd is not None:
                os.close(fd)
            try:
                os.remove(lock_path)
            except OSError:
                pass


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def append_event(
    *,
    agent_id: str,
    task_id: str,
    run_id: str,
    event_type: EventType,
    data: Dict[str, Any],
    parent_seq: Optional[int] = None,
) -> int:
    """把一条事件追加到对应轨迹文件,返回分配到的 seq。"""
    path = trajectory_path(agent_id, task_id, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(path):
        seq = _count_lines(path)
        event = Event(
            run_id=run_id,
            agent_id=agent_id,
            task_id=task_id,
            timestamp=_now_iso(),
            event_type=event_type,
            seq=seq,
            parent_seq=parent_seq,
            data=data,
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
    logger.debug("记录事件 %s seq=%d -> %s", event_type, seq, path)
    return seq


def read_events(agent_id: str, task_id: str, run_id: str) -> List[Event]:
    """读取一份轨迹文件,按 seq 升序返回事件列表。"""
    path = trajectory_path(agent_id, task_id, run_id)
    if not path.exists():
        return []
    events: List[Event] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(Event.model_validate_json(line))
    events.sort(key=lambda e: e.seq)
    return events


# --------------------------------------------------------------------------- #
# 内容寻址 blob:大字段外置存盘,按 sha256 去重(重复的 wiki/历史消息只存一份)
# --------------------------------------------------------------------------- #


def blob_root() -> Path:
    return traj_root() / "blobs"


def _blob_path(digest: str) -> Path:
    # 前 2 位 fan-out,避免单目录文件过多
    return blob_root() / digest[:2] / f"{digest}.json"


def put_blob(content: Any) -> Dict[str, Any]:
    """按 sha256 把内容写入 blob store(已存在则跳过),返回引用。

    内容寻址 → 相同内容同一路径,写入天然幂等;用临时文件 + 原子 rename 落盘,
    因而无需像 jsonl 那样加锁,proxy 与 wrapper 两个进程可安全并发写。
    """
    raw = json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    path = _blob_path(digest)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{digest}.{uuid4().hex}.tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)
    return {"$blob": digest, "bytes": len(raw)}


def blobify(value: Any, threshold: int = BLOB_THRESHOLD_BYTES) -> Any:
    """超过阈值的值外置为 blob 引用,否则原样内联。"""
    try:
        size = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        return value
    return put_blob(value) if size > threshold else value


def _load_blob(digest: str) -> Any:
    return json.loads(_blob_path(digest).read_text(encoding="utf-8"))


def rehydrate(obj: Any) -> Any:
    """递归把 {"$blob": ...} 引用还原成原始内容;非引用原样返回(对旧轨迹透明)。"""
    if isinstance(obj, dict):
        digest = obj.get("$blob")
        if isinstance(digest, str):
            return rehydrate(_load_blob(digest))
        return {k: rehydrate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rehydrate(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# 客户端构造:指向代理、自动携带追踪 header
# --------------------------------------------------------------------------- #


def make_client(role: str = "agent"):
    """构造一个指向本地代理的 OpenAI 客户端。

    代理通过这些 header 把每次 LLM 调用关联到具体轨迹;role 用于区分被测
    agent 的调用与用户模拟器的调用(都记为 llm_call,但 data.role 不同)。
    """
    import httpx
    from openai import OpenAI  # 延迟导入,避免无依赖时影响纯录制功能

    ctx = current_context()
    if ctx is None:
        raise RuntimeError("make_client 必须在 run_context(...) 内调用")
    headers = {
        "X-Run-Id": ctx.run_id,
        "X-Agent-Id": ctx.agent_id,
        "X-Task-Id": ctx.task_id,
        "X-Llm-Role": role,
    }
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
    api_key = os.getenv("OPENAI_API_KEY", "proxy-placeholder")
    # trust_env=False:agent→本地代理这一跳必须直连 localhost,绕开系统/VPN 代理
    # (HTTP_PROXY/HTTPS_PROXY),否则发往 127.0.0.1 的请求会被科学上网代理拦截转发而 502。
    http_client = httpx.Client(trust_env=False, timeout=120.0)
    return OpenAI(
        base_url=base_url, api_key=api_key, default_headers=headers, http_client=http_client
    )


# --------------------------------------------------------------------------- #
# 极简 .env 加载:避免引入 python-dotenv 依赖
# --------------------------------------------------------------------------- #


def load_env(path: Optional[Path] = None) -> None:
    """加载项目根目录下的 .env(已存在的真实环境变量优先,不覆盖)。"""
    env_path = path or (PROJECT_ROOT / ".env")
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
