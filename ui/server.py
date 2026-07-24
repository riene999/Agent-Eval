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

from analysis.compare import compare_files, to_markdown as comparison_markdown
from analysis.export import export_dpo, export_sft
from analysis.replay import compare_trace_files, replay_payload
from proxy.recorder import PROJECT_ROOT, rehydrate
from skills.registry import available_tool_names, get_skill, import_skill, list_skills

STATIC_DIR = Path(__file__).resolve().parent / "static"
REPORTS_DIR = PROJECT_ROOT / "reports"
TRAJECTORIES_DIR = PROJECT_ROOT / "trajectories"
JOBS_DIR = REPORTS_DIR / ".jobs"
COMPARISONS_DIR = REPORTS_DIR / "comparisons"
TRAIN_DIR = PROJECT_ROOT / "data" / "train"

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


class CompareRequest(BaseModel):
    """两份报告与可调回归阈值。"""

    baseline_run_id: str
    candidate_run_id: str
    accuracy_drop_max: float = Field(default=0.03, ge=0.0, le=1.0)
    token_increase_max: float = Field(default=0.20, ge=0.0, le=10.0)
    latency_p95_increase_max: float = Field(default=0.25, ge=0.0, le=10.0)
    cost_increase_max: float = Field(default=0.20, ge=0.0, le=10.0)
    redundant_call_rate_increase_max: float = Field(default=0.05, ge=0.0, le=1.0)


class TraceRef(BaseModel):
    agent_id: str
    task_id: str
    run_id: str


class TraceCompareRequest(BaseModel):
    baseline: TraceRef
    candidate: TraceRef


class ExportRequest(BaseModel):
    """训练数据导出只允许写入 data/train，避免网页传入任意路径。"""

    kind: Literal["sft", "dpo"]
    filename: str
    run_ids: list[str] = Field(default_factory=list)
    agent: str = "react_agent_v1"
    mode: Literal["failed_only", "all"] = "failed_only"
    append: bool = False


class SkillImportRequest(BaseModel):
    """浏览器读取本地 JSON 后提交，后端只保存通过校验的能力包。"""

    skill: dict[str, Any]
    overwrite: bool = False


class SkillRunRequest(BaseModel):
    """单 Skill 或 N+1 Skill 评测参数。"""

    mode: Literal["single", "n_plus_one"]
    skill: str
    baseline_skills: list[str] = Field(default_factory=list)
    model: str
    run_id: Optional[str] = None
    start: int = Field(default=0, ge=0)
    count: int = Field(default=30, ge=1, le=1000)
    trials: int = Field(default=1, ge=1, le=20)
    concurrency: int = Field(default=1, ge=1, le=32)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: Optional[int] = None


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
    report_kind = meta.get("kind")
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
        "kind": (
            "N+1 Skill 评测"
            if report_kind == "skill_n_plus_one"
            else ("多次试验" if multi_trial else ("单 Skill 评测" if meta.get("skill_mode") else "单次评测"))
        ),
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
    if kind == "skill_route":
        selected = data.get("selected_skill") or "none"
        return f"选择 {selected} · {str(data.get('reason', ''))[:120]}"
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
            {
                **{key: value for key, value in job.items() if key not in {"process", "log_handle"}},
                "report_exists": (REPORTS_DIR / f"{job['run_id']}.json").exists(),
            }
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


def _safe_report_path(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise HTTPException(status_code=400, detail="报告标识含有非法字符")
    path = REPORTS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"未找到报告 {run_id}")
    return path


@app.delete("/api/reports/{run_id}")
def delete_report(run_id: str) -> dict[str, Any]:
    """真实删除一份报告的 JSON 与 Markdown，不触碰原始 trajectory。"""
    json_path = _safe_report_path(run_id)
    markdown_path = json_path.with_suffix(".md")
    deleted: list[str] = []
    try:
        if markdown_path.exists():
            markdown_path.unlink()
            deleted.append(markdown_path.name)
        json_path.unlink()
        deleted.append(json_path.name)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"删除报告失败：{exc}") from exc
    return {"run_id": run_id, "deleted": deleted, "trajectory_deleted": False}


@app.post("/api/comparisons")
def compare_reports(request: CompareRequest) -> dict[str, Any]:
    """对共同任务做配对比较，并把本次门禁结果保存为独立报告。"""
    baseline_path = _safe_report_path(request.baseline_run_id)
    candidate_path = _safe_report_path(request.candidate_run_id)
    if baseline_path == candidate_path:
        raise HTTPException(status_code=400, detail="基线与候选报告不能相同")
    thresholds = {
        "accuracy_drop_max": request.accuracy_drop_max,
        "token_increase_max": request.token_increase_max,
        "latency_p95_increase_max": request.latency_p95_increase_max,
        "cost_increase_max": request.cost_increase_max,
        "redundant_call_rate_increase_max": request.redundant_call_rate_increase_max,
    }
    result = compare_files(baseline_path, candidate_path, thresholds)
    comparison_id = f"{request.baseline_run_id}__{request.candidate_run_id}"
    COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    (COMPARISONS_DIR / f"{comparison_id}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (COMPARISONS_DIR / f"{comparison_id}.md").write_text(
        comparison_markdown(result), encoding="utf-8")
    result["comparison_id"] = comparison_id
    return result


@app.get("/api/traces")
def traces(
    agent: str = "",
    task: str = "",
    run: str = "",
    result: str = "all",
    limit: int = Query(default=200, ge=1, le=2000),
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


@app.post("/api/replay")
def replay_trace(reference: TraceRef) -> dict[str, Any]:
    """返回完整历史事件供浏览器逐步播放，不重新执行模型和工具。"""
    return replay_payload(_safe_trace_path(reference.agent_id, reference.task_id, reference.run_id))


@app.post("/api/trace-diff")
def trace_diff(request: TraceCompareRequest) -> dict[str, Any]:
    baseline = _safe_trace_path(
        request.baseline.agent_id, request.baseline.task_id, request.baseline.run_id)
    candidate = _safe_trace_path(
        request.candidate.agent_id, request.candidate.task_id, request.candidate.run_id)
    if baseline == candidate:
        raise HTTPException(status_code=400, detail="请选择两条不同轨迹")
    return compare_trace_files(baseline, candidate)


def _safe_export_name(filename: str) -> str:
    name = Path(filename).name
    if name != filename or not re.fullmatch(r"[A-Za-z0-9_.-]+\.jsonl", name):
        raise HTTPException(status_code=400, detail="文件名只能包含字母、数字、点、横线和下划线，并以 .jsonl 结尾")
    return name


def _export_row(path: Path) -> dict[str, Any]:
    rows = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            rows = sum(1 for line in handle if line.strip())
    except OSError:
        pass
    return {
        "filename": path.name,
        "rows": rows,
        "size_bytes": path.stat().st_size,
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "download_url": f"/api/exports/{path.name}",
    }


@app.get("/api/exports")
def exports() -> list[dict[str, Any]]:
    if not TRAIN_DIR.exists():
        return []
    paths = sorted(TRAIN_DIR.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [_export_row(path) for path in paths]


@app.get("/api/dpo-run-ids")
def dpo_run_ids(agent: str = "react_agent_v1") -> list[dict[str, Any]]:
    """列出可供企业知识问答 DPO 导出的逻辑运行批次。"""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", agent):
        raise HTTPException(status_code=400, detail="Agent 标识含有非法字符")
    root = TRAJECTORIES_DIR / agent
    grouped: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return []
    for path in root.glob("ekb_*/*.jsonl"):
        raw_run_id = path.stem
        run_id = re.sub(r"_t\d+$", "", raw_run_id)
        row = _trace_row(path)
        item = grouped.setdefault(run_id, {
            "run_id": run_id,
            "trajectory_count": 0,
            "task_ids": set(),
            "failed_count": 0,
            "success_count": 0,
            "latest_mtime": 0.0,
        })
        item["trajectory_count"] += 1
        item["task_ids"].add(row["task_id"])
        item["latest_mtime"] = max(item["latest_mtime"], path.stat().st_mtime)
        if row["success"] is True:
            item["success_count"] += 1
        elif row["success"] is False:
            item["failed_count"] += 1
    result = []
    for item in grouped.values():
        result.append({
            "run_id": item["run_id"],
            "trajectory_count": item["trajectory_count"],
            "task_count": len(item["task_ids"]),
            "failed_count": item["failed_count"],
            "success_count": item["success_count"],
            "updated_at": datetime.fromtimestamp(item["latest_mtime"]).isoformat(timespec="seconds"),
        })
    return sorted(result, key=lambda item: item["updated_at"], reverse=True)


@app.post("/api/exports")
def create_export(request: ExportRequest) -> dict[str, Any]:
    name = _safe_export_name(request.filename)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", request.agent):
        raise HTTPException(status_code=400, detail="Agent 标识含有非法字符")
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    out = TRAIN_DIR / name
    if request.kind == "sft":
        count = export_sft(out)
        skipped = 0
    else:
        if not request.run_ids:
            raise HTTPException(status_code=400, detail="DPO 导出至少需要一个运行标识")
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) for run_id in request.run_ids):
            raise HTTPException(status_code=400, detail="运行标识含有非法字符")
        count, skipped = export_dpo(
            request.run_ids, request.agent, request.mode, out, append=request.append)
    preview = []
    if out.exists():
        with out.open("r", encoding="utf-8") as handle:
            for _, line in zip(range(3), handle):
                preview.append(line.strip()[:4000])
    return {"file": _export_row(out), "written": count, "skipped": skipped, "preview": preview,
            "tools_file": "tools.json" if request.kind == "sft" else None}


@app.get("/api/exports/{filename}")
def download_export(filename: str) -> FileResponse:
    name = _safe_export_name(filename)
    path = TRAIN_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="未找到导出文件")
    return FileResponse(path, media_type="application/x-ndjson", filename=name)


def _skill_row(spec: Any) -> dict[str, Any]:
    return {
        **spec.model_dump(),
        "tool_count": len(spec.tools),
    }


@app.get("/api/skills")
def skills() -> dict[str, Any]:
    """列出已安装 Skill 以及导入时可引用的安全工具名。"""
    return {
        "skills": [_skill_row(spec) for spec in list_skills(include_disabled=True)],
        "available_tools": available_tool_names(),
    }


@app.get("/api/skills/{skill_id}")
def skill_detail(skill_id: str) -> dict[str, Any]:
    try:
        return _skill_row(get_skill(skill_id, include_disabled=True))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/skills")
def create_skill(request: SkillImportRequest) -> dict[str, Any]:
    """导入声明式 Skill；不接受 Python 文件，也不会执行上传内容。"""
    try:
        spec = import_skill(request.skill, overwrite=request.overwrite)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _skill_row(spec)


@app.post("/api/skill-runs")
def create_skill_run(request: SkillRunRequest) -> dict[str, Any]:
    configs = _model_configs()
    if request.model not in configs:
        raise HTTPException(status_code=400, detail=f"models.json 中没有模型 {request.model!r}")
    try:
        get_skill(request.skill)
        for skill_id in request.baseline_skills:
            get_skill(skill_id)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if request.mode == "n_plus_one" and not request.baseline_skills:
        raise HTTPException(status_code=400, detail="N+1 评测至少需要一个原有 Skill")
    if request.skill in request.baseline_skills:
        raise HTTPException(status_code=400, detail="待新增 Skill 不能已经存在于原有 Skill 集合")

    run_id = request.run_id or f"skill_{uuid4().hex[:10]}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise HTTPException(status_code=400, detail="运行标识只能包含字母、数字、点、横线和下划线")
    _refresh_jobs()
    with _JOBS_LOCK:
        if run_id in _JOBS and _JOBS[run_id]["status"] == "运行中":
            raise HTTPException(status_code=409, detail="同名 Skill 评测正在运行")

    command = [
        sys.executable,
        "-m",
        "runner.skill_eval",
        "--mode",
        request.mode,
        "--skill",
        request.skill,
        "--model",
        request.model,
        "--run-id",
        run_id,
        "--start",
        str(request.start),
        "--count",
        str(request.count),
        "--trials",
        str(request.trials),
        "--concurrency",
        str(request.concurrency),
        "--temperature",
        str(request.temperature),
    ]
    if request.baseline_skills:
        command.extend(["--baseline-skills", ",".join(request.baseline_skills)])
    if request.seed is not None:
        command.extend(["--seed", str(request.seed)])

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
        "agent": "skill_router",
        "model": request.model,
        "dataset": "Skill 专项评测",
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
