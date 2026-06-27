"""企业知识库数据集的只读工具(无状态:只查不改,读 knowledge/ 下的文档与结构化表)。

工具分两类:
- 文档检索:list_domains / search_knowledge / get_document —— 自动索引 knowledge/docs/ 下的
  全部 .md,新增文档"丢文件进去"即生效,无需改代码(支撑扩到上千条数据)。
- 结构化查询:员工/假期/节假日/IT 资产/工单/差旅标准/报销政策 —— 读 knowledge/data/*.json。
每个工具都被 @traced_tool 包装,返回 JSON 可序列化结果;查不到给出明确提示而非抛异常。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List

from tools.wrapper import traced_tool

_KB = Path(__file__).resolve().parent / "knowledge"
_DOCS = _KB / "docs"
_DATA = _KB / "data"

# 域别名 -> 文档子目录
_DOMAIN_ALIAS = {
    "hr": "hr", "人力": "hr", "人力资源": "hr",
    "it": "it", "信息": "it",
    "finance": "finance", "财务": "finance", "fin": "finance",
}


@lru_cache(maxsize=1)
def _load_docs() -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    if _DOCS.is_dir():
        for path in sorted(_DOCS.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            doc_id = path.relative_to(_DOCS).with_suffix("").as_posix()
            title = text.lstrip().splitlines()[0].lstrip("# ").strip() if text.strip() else doc_id
            docs.append({"doc_id": doc_id, "domain": doc_id.split("/")[0], "title": title, "text": text})
    return docs


@lru_cache(maxsize=None)
def _load_table(name: str) -> Dict[str, Any]:
    path = _DATA / f"{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _snippet(text: str, terms: List[str], width: int = 100) -> str:
    low = text.lower()
    for t in terms:
        i = low.find(t.lower())
        if i >= 0:
            start = max(0, i - 20)
            return text[start:start + width].replace("\n", " ").strip()
    return text.replace("\n", " ").strip()[:width]


@traced_tool
def list_domains() -> List[str]:
    """列出知识库覆盖的业务域(如 hr / it / finance)。"""
    return sorted({d["domain"] for d in _load_docs()})


@traced_tool
def search_knowledge(domain: str, query: str) -> List[Dict[str, str]]:
    """在指定域(或 'all')的知识文档里按关键词检索,返回最相关的若干篇(doc_id/标题/片段)。"""
    target = _DOMAIN_ALIAS.get(domain.strip().lower(), domain.strip().lower())
    terms = [t for t in query.split() if t]
    scored = []
    for d in _load_docs():
        if target not in ("all", "") and d["domain"] != target:
            continue
        hay = (d["title"] + " " + d["text"]).lower()
        score = sum(hay.count(t.lower()) for t in terms)
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return [{"doc_id": "", "title": "未找到相关文档", "snippet": ""}]
    return [
        {"doc_id": d["doc_id"], "title": d["title"], "snippet": _snippet(d["text"], terms)}
        for _, d in scored[:3]
    ]


@traced_tool
def get_document(doc_id: str) -> str:
    """按 doc_id(如 'hr/annual_leave')取文档全文。"""
    for d in _load_docs():
        if d["doc_id"] == doc_id:
            return d["text"]
    return f"Error: 未找到文档 {doc_id}"


@traced_tool
def find_employee_id(name: str) -> str:
    """按姓名查员工编号(找不到或重名时给出提示)。后续按编号查档案/假期/资产等。"""
    table = _load_table("employees")
    matches = [eid for eid, info in table.items() if info.get("name") == name]
    if not matches:
        return f"Error: 未找到姓名为「{name}」的员工"
    if len(matches) > 1:
        return f"Error: 姓名「{name}」有多位({matches}),请进一步区分"
    return matches[0]


@traced_tool
def get_employee(employee_id: str) -> Dict[str, Any]:
    """查员工档案(部门、职级、入职日期、汇报对象等)。"""
    return _load_table("employees").get(employee_id, {"error": f"未找到员工 {employee_id}"})


@traced_tool
def get_leave_balance(employee_id: str) -> Dict[str, Any]:
    """查员工当年假期余额(年假总数/已用/剩余、病假剩余)。"""
    return _load_table("leave_balances").get(employee_id, {"error": f"未找到 {employee_id} 的假期记录"})


@traced_tool
def list_holidays(year: int) -> Any:
    """列出指定年份的法定节假日。"""
    return _load_table("holidays").get(str(year), {"error": f"未找到 {year} 年节假日"})


@traced_tool
def get_it_asset(employee_id: str) -> Dict[str, Any]:
    """查员工名下的 IT 资产(笔记本/显示器/手机等)。"""
    return _load_table("it_assets").get(employee_id, {"error": f"未找到 {employee_id} 的资产记录"})


@traced_tool
def get_ticket(ticket_id: str) -> Dict[str, Any]:
    """查 IT 工单状态。"""
    return _load_table("it_tickets").get(ticket_id, {"error": f"未找到工单 {ticket_id}"})


@traced_tool
def get_expense_limit(level: str, category: str) -> Dict[str, Any]:
    """查某职级在某类目(meal/hotel/transport)的差旅每日报销上限。"""
    limits = _load_table("expense_limits").get(level)
    if not limits:
        return {"error": f"未找到职级 {level} 的差旅标准"}
    if category not in limits:
        return {"error": f"未知类目 {category}", "available": list(limits)}
    return {"level": level, "category": category, "daily_limit": limits[category]}


@traced_tool
def get_reimbursement_policy(category: str) -> str:
    """查某类报销(差旅/餐饮/办公用品)的材料与流程说明。"""
    return _load_table("reimbursement_policies").get(category, f"Error: 未找到类目 {category} 的报销政策")


@traced_tool
def get_department(name: str) -> Dict[str, Any]:
    """查部门信息(负责人、人数、所在位置)。"""
    return _load_table("departments").get(name, {"error": f"未找到部门 {name}"})


@traced_tool
def get_meeting_room(room_id: str) -> Dict[str, Any]:
    """查会议室信息(位置、容量、配备设备)。"""
    return _load_table("meeting_rooms").get(room_id, {"error": f"未找到会议室 {room_id}"})


@traced_tool
def get_contract(contract_id: str) -> Dict[str, Any]:
    """查合同的审批状态与基本信息(对方、负责人、到期日)。"""
    return _load_table("contracts").get(contract_id, {"error": f"未找到合同 {contract_id}"})


@traced_tool
def get_supplier(name: str) -> Dict[str, Any]:
    """按名称查供应商(类别、评级、联系方式)。"""
    return _load_table("suppliers").get(name, {"error": f"未找到供应商 {name}"})


TOOLS: List[Callable[..., Any]] = [
    list_domains, search_knowledge, get_document,
    find_employee_id, get_employee, get_leave_balance, list_holidays,
    get_it_asset, get_ticket, get_expense_limit, get_reimbursement_policy,
    get_department, get_meeting_room, get_contract, get_supplier,
]
TOOLS_BY_NAME: Dict[str, Callable[..., Any]] = {t.__name__: t for t in TOOLS}
