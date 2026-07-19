"""Agent-Eval 可视化控制台的 FastAPI 服务。

提供数据集、模型、报告和轨迹的只读接口，并将现有 runner 命令包装成后台任务。
页面资源使用原生 HTML/CSS/JavaScript，避免为本地控制台引入额外前端依赖。
模型密钥不会通过任何接口返回。
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from proxy.recorder import PROJECT_ROOT, rehydrate

STATIC_DIR = Path(__file__).resolve().parent / "static"
REPORTS_DIR = PROJECT_ROOT / "reports"
TRAJECTORIES_DIR = PROJECT_ROOT / "trajectories"
JOBS_DIR = REPORTS_DIR / ".jobs"

app = FastAPI(title="Agent-Eval 评测控制台", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RunRequest(BaseModel):
    """网页发起评测时允许填写的现有 runner 参数。"""

    agent: Literal["react", "plan_solve"] = "react"
    model: str
    dataset: Literal["tau_retail", "enterprise_kb"] = "tau_retail"
    split: Literal["test", "train", "dev"] = "test"
    start: int = Field(default=0, ge=0)
    count: int = Field(default=5, ge=1, le=1000)
    run_id: Optional[str] = None
    trials: int = Field(default=1, ge=1, le=20)
    concurrency: int = Field(default=1, ge=1, le=32)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: Optional[int] = None
    llm_judge: bool = False
    attribution: bool = False
    attribution_mode: str = "failed_only"
    judge_model: Optional[str] = None


_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _model_configs() -> dict[str, dict[str, Any]]:
    data = _read_json(PROJECT_ROOT / "models.json", {})
    return data if isinstance(data, dict) else {}


def _provider_name(base_url: str) -> str:
    host = base_url.lower()
    if "deepseek" in host:
        return "DeepSeek"
    if "bigmodel" in host:
        return "智谱 AI"
    if "dashscope" in host or "aliyuncs" in host:
        return "阿里云百炼"
    if "siliconflow" in host:
        return "硅基流动"
    if "openai" in host:
        return "OpenAI"
    return "自定义接口"


def _normalize_report(path: Path) -> dict[str, Any]:
    payload = _read_json(path, {})
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    multi_trial = "avg_pass_at_1" in summary
    accuracy = summary.get("avg_pass_at_1") if multi_trial else summary.get("accuracy")
    task_count = summary.get("tasks") if multi_trial else summary.get("n")
    if task_count is None:
        task_count = meta.get("count") or len(payload.get("tasks", []))
    return {
        "run_id": str(meta.get("run_id") or path.stem),
        "agent_id": meta.get("agent_id") or "-",
        "model": meta.get("model") or "-",
        "timestamp": meta.get("timestamp") or datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "task_count": task_count or 0,
        "trials": meta.get("trials", 1),
        "accuracy": accuracy,
        "pass_at_n": summary.get("avg_pass_at_n"),
        "avg_total_tokens": summary.get("avg_total_tokens"),
        "avg_tool_calls": summary.get("avg_tool_calls"),
        "avg_cost_usd": summary.get("avg_cost_usd"),
        "latency_p95": summary.get("latency_p95"),
        "failure_distribution": summary.get("failure_distribution") or {},
        "kind": "多次试验" if multi_trial else "单次评测",
        "mtime": path.stat().st_mtime,
    }


def _all_reports() -> list[dict[str, Any]]:
    if not REPORTS_DIR.exists():
        return []
    reports = [_normalize_report(path) for path in REPORTS_DIR.glob("*.json")]
    return sorted(reports, key=lambda item: item["mtime"], reverse=True)


def _count_decorated_tools(path: Path) -> int:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return 0
    total = 0
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            name = decorator.id if isinstance(decorator, ast.Name) else ""
            if name == "traced_tool":
                total += 1
                break
    return total


def _enterprise_dataset() -> dict[str, Any]:
    root = PROJECT_ROOT / "data" / "enterprise_kb"
    task_file = root / "tasks.jsonl"
    tasks: list[dict[str, Any]] = []
    if task_file.exists():
        for line in task_file.read_text(encoding="utf-8").splitlines():
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    domains = sorted({str(item.get("domain")) for item in tasks if item.get("domain")})
    return {
        "id": "enterprise_kb",
        "name": "企业知识问答",
        "description": "无状态的企业 HR、IT、财务等工具调用任务",
        "status": "可用" if tasks else "未生成",
        "splits": {"all": len(tasks)},
        "task_count": len(tasks),
        "domains": domains,
        "document_count": len(list((root / "knowledge" / "docs").rglob("*.md"))),
        "tool_count": _count_decorated_tools(root / "tools.py"),
    }


def _tau_dataset() -> dict[str, Any]:
    counts: dict[str, int] = {}
    status = "未安装"
    message = "请将 tau-bench 克隆到 data/tau-bench"
    try:
        from tools.tau_tools import ALL_TOOLS, load_tasks

        counts = {split: len(load_tasks(split)) for split in ("test", "train", "dev")}
        status = "可用"
        message = "τ-bench Retail 官方任务与数据库状态判分"
        tool_count = len(ALL_TOOLS)
    except BaseException as exc:
        tool_count = 0
        message = str(exc).splitlines()[0][:180]
    return {
        "id": "tau_retail",
        "name": "τ-bench Retail",
        "description": message,
        "status": status,
        "splits": counts,
        "task_count": sum(counts.values()),
        "domains": ["Retail"],
        "document_count": 1 if status == "可用" else 0,
        "tool_count": tool_count,
    }


def _trace_files() -> list[Path]:
    if not TRAJECTORIES_DIR.exists():
        return []
    files = [path for path in TRAJECTORIES_DIR.glob("*/*/*.jsonl") if "blobs" not in path.parts]
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)


def _trace_row(path: Path) -> dict[str, Any]:
    agent_id, task_id, filename = path.parts[-3:]
    event_count = 0
    success: Optional[bool] = None
    event_types: set[str] = set()
    timestamp: Optional[str] = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event_count += 1
                event = json.loads(line)
                event_types.add(str(event.get("event_type", "")))
                timestamp = timestamp or event.get("timestamp")
                if event.get("event_type") == "final_output":
                    success = bool(event.get("data", {}).get("success"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        pass
    return {
        "agent_id": agent_id,
        "task_id": task_id,
        "run_id": Path(filename).stem,
        "event_count": event_count,
        "success": success,
        "has_judge": "llm_judge" in event_types,
        "has_attribution": "attribution" in event_types,
        "timestamp": timestamp or datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "mtime": path.stat().st_mtime,
    }


def _event_summary(event: dict[str, Any]) -> str:
    kind = event.get("event_type")
    data = event.get("data", {})

    def number(value: Any) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    if kind == "llm_call":
        return f"{data.get('model', '模型调用')} · {data.get('prompt_tokens', 0)} + {data.get('completion_tokens', 0)} tokens · {number(data.get('latency_ms')):.0f} ms"
    if kind == "tool_call":
        args = json.dumps(data.get("args", {}), ensure_ascii=False)
        return f"调用 {data.get('tool_name', '-')} · {args[:120]}"
    if kind == "tool_return":
        state = "异常" if data.get("error") else "返回成功"
        return f"{data.get('tool_name', '-')} {state} · {number(data.get('latency_ms')):.0f} ms"
    if kind == "final_output":
        state = "任务成功" if data.get("success") else "任务失败"
        return f"{state} · {str(data.get('output', ''))[:120]}"
    if kind == "llm_judge":
        score = data.get("overall_score", data.get("score", "-"))
        return f"辅助评分 {score} · {str(data.get('reason', data.get('summary', '')))[:120]}"
    if kind == "attribution":
        return f"{data.get('error_category', '错误归因')} · {str(data.get('summary', ''))[:120]}"
    return str(data)[:140]


def _safe_trace_path(agent_id: str, task_id: str, run_id: str) -> Path:
    allowed = re.compile(r"^[A-Za-z0-9_.-]+$")
    if not all(allowed.fullmatch(value) for value in (agent_id, task_id, run_id)):
        raise HTTPException(status_code=400, detail="轨迹标识含有非法字符")
    path = TRAJECTORIES_DIR / agent_id / task_id / f"{run_id}.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="未找到轨迹")
    return path


def _refresh_jobs() -> list[dict[str, Any]]:
    with _JOBS_LOCK:
        for job in _JOBS.values():
            process = job["process"]
            code = process.poll()
            if code is None:
                job["status"] = "运行中"
            elif code == 0:
                job["status"] = "已完成"
                job["finished_at"] = job.get("finished_at") or datetime.now().isoformat(timespec="seconds")
            else:
                job["status"] = "失败"
                job["finished_at"] = job.get("finished_at") or datetime.now().isoformat(timespec="seconds")
                job["exit_code"] = code
            if code is not None and not job.get("log_closed"):
                job["log_handle"].close()
                job["log_closed"] = True
        return [
            {key: value for key, value in job.items() if key not in {"process", "log_handle"}}
            for job in sorted(_JOBS.values(), key=lambda item: item["started_at"], reverse=True)
        ]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    reports = _all_reports()
    traces = _trace_files()
    models = _model_configs()
    return {
        "counts": {
            "reports": len(reports),
            "traces": len(traces),
            "models": len(models),
            "datasets": 2,
        },
        "latest_report": reports[0] if reports else None,
        "recent_reports": reports[:6],
    }


@app.get("/api/datasets")
def datasets() -> list[dict[str, Any]]:
    return [_tau_dataset(), _enterprise_dataset()]


@app.get("/api/models")
def models() -> list[dict[str, Any]]:
    result = []
    for name, config in _model_configs().items():
        base_url = str(config.get("base_url", ""))
        result.append({
            "name": name,
            "provider": _provider_name(base_url),
            "base_url": base_url,
            "configured": bool(base_url and config.get("api_key")),
        })
    return result


@app.get("/api/reports")
def reports() -> list[dict[str, Any]]:
    return _all_reports()


@app.get("/api/reports/{run_id}")
def report_detail(run_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise HTTPException(status_code=400, detail="报告标识含有非法字符")
    path = REPORTS_DIR / f"{run_id}.json"
    payload = _read_json(path)
    if payload is None:
        raise HTTPException(status_code=404, detail="未找到报告")
    markdown_path = path.with_suffix(".md")
    return {
        "report": payload,
        "markdown": markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else "",
    }


@app.get("/api/traces")
def traces(
    agent: str = "",
    task: str = "",
    run: str = "",
    result: str = "all",
    limit: int = Query(default=200, ge=1, le=500),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _trace_files():
        row = _trace_row(path)
        if agent and agent.lower() not in row["agent_id"].lower():
            continue
        if task and task.lower() not in row["task_id"].lower():
            continue
        if run and run.lower() not in row["run_id"].lower():
            continue
        if result == "success" and row["success"] is not True:
            continue
        if result == "failed" and row["success"] is not False:
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


@app.get("/api/traces/{agent_id}/{task_id}/{run_id}")
def trace_detail(agent_id: str, task_id: str, run_id: str) -> dict[str, Any]:
    path = _safe_trace_path(agent_id, task_id, run_id)
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            events.append({
                "seq": event.get("seq"),
                "parent_seq": event.get("parent_seq"),
                "timestamp": event.get("timestamp"),
                "event_type": event.get("event_type"),
                "summary": _event_summary(event),
            })
    return {"agent_id": agent_id, "task_id": task_id, "run_id": run_id, "events": events}


@app.get("/api/traces/{agent_id}/{task_id}/{run_id}/events/{seq}")
def event_detail(agent_id: str, task_id: str, run_id: str, seq: int) -> dict[str, Any]:
    path = _safe_trace_path(agent_id, task_id, run_id)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("seq") == seq:
                return rehydrate(event)
    raise HTTPException(status_code=404, detail="未找到事件")


@app.get("/api/proxy-health")
def proxy_health() -> dict[str, Any]:
    base = os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1").rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    try:
        with httpx.Client(timeout=1.5, trust_env=False) as client:
            response = client.get(root.rstrip("/") + "/health")
        return {"online": response.status_code == 200, "status_code": response.status_code}
    except httpx.HTTPError as exc:
        return {"online": False, "error": type(exc).__name__}


@app.post("/api/runs")
def create_run(request: RunRequest) -> dict[str, Any]:
    configs = _model_configs()
    if request.model not in configs:
        raise HTTPException(status_code=400, detail=f"models.json 中没有模型 {request.model!r}")
    if request.judge_model and request.judge_model not in configs:
        raise HTTPException(status_code=400, detail=f"models.json 中没有评分模型 {request.judge_model!r}")
    if request.attribution_mode not in {"failed_only", "all"} and not re.fullmatch(r"sample_\d+", request.attribution_mode):
        raise HTTPException(status_code=400, detail="归因范围应为 failed_only、all 或 sample_N")

    run_id = request.run_id or uuid4().hex[:12]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise HTTPException(status_code=400, detail="运行标识只能包含字母、数字、点、横线和下划线")
    _refresh_jobs()
    with _JOBS_LOCK:
        if run_id in _JOBS and _JOBS[run_id]["status"] == "运行中":
            raise HTTPException(status_code=409, detail="同名评测任务正在运行")

    command = [
        sys.executable, "-m", "runner.run",
        "--agent", request.agent,
        "--model", request.model,
        "--run-id", run_id,
        "--trials", str(request.trials),
        "--concurrency", str(request.concurrency),
        "--temperature", str(request.temperature),
    ]
    if request.seed is not None:
        command.extend(["--seed", str(request.seed)])
    if request.llm_judge:
        command.append("--llm-judge")
    if request.attribution:
        command.extend(["--attribution", "--attribution-mode", request.attribution_mode])
    if request.judge_model:
        command.extend(["--judge-model", request.judge_model])

    if request.dataset == "tau_retail":
        command.extend([
            "--split", request.split,
            "--start", str(request.start),
            "--count", str(request.count),
        ])
    else:
        task_file = PROJECT_ROOT / "data" / "enterprise_kb" / "tasks.jsonl"
        task_ids = []
        if task_file.exists():
            for line in task_file.read_text(encoding="utf-8").splitlines():
                try:
                    task_ids.append(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        selected = task_ids[request.start:request.start + request.count]
        if not selected:
            raise HTTPException(status_code=400, detail="所选范围内没有企业知识问答任务")
        command.extend(["--tasks", ",".join(selected)])

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = JOBS_DIR / f"{run_id}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    job = {
        "job_id": run_id,
        "run_id": run_id,
        "agent": request.agent,
        "model": request.model,
        "dataset": request.dataset,
        "count": request.count,
        "trials": request.trials,
        "status": "运行中",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "log_path": str(log_path),
        "process": process,
        "log_handle": log_handle,
    }
    with _JOBS_LOCK:
        _JOBS[run_id] = job
    return {key: value for key, value in job.items() if key not in {"process", "log_handle"}}


@app.get("/api/jobs")
def jobs() -> list[dict[str, Any]]:
    return _refresh_jobs()


@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def job_log(job_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id):
        raise HTTPException(status_code=400, detail="任务标识含有非法字符")
    path = JOBS_DIR / f"{job_id}.log"
    if not path.exists():
        raise HTTPException(status_code=404, detail="未找到任务日志")
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-40000:]


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 Agent-Eval 可视化控制台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    uvicorn.run("ui.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
