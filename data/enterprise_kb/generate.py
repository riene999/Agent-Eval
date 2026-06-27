"""enterprise_kb 数据生成器:从 knowledge/ 批量产 grounded 数据。

gold_trajectory 与 reference_outputs 均由知识库数据**直接构造**,因此天生自洽(judge 必过)。
轮转取样保证 6 个域均匀。用法:python data/enterprise_kb/generate.py [N]  (默认 50)。
扩到 1000 条:往 knowledge/ 补数据 + 增大 N 即可。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_DATA = _DIR / "knowledge" / "data"
_NOW = date(2026, 6, 28)
_DOMAIN_CODE = {"HR": "hr", "IT": "it", "Finance": "fin", "Admin": "admin",
                "Legal": "legal", "Procurement": "proc"}


def _load(name: str) -> dict:
    p = _DATA / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _annual_days(hire: str) -> int:
    years = (_NOW - date.fromisoformat(hire)).days / 365.25
    return 0 if years < 1 else (15 if years >= 10 else 10)


def build() -> dict:
    """构造多个类别桶 {bucket: [record,...]}(record 暂不含 id)。"""
    emps, leave, assets = _load("employees"), _load("leave_balances"), _load("it_assets")
    limits, holidays = _load("expense_limits"), _load("holidays").get("2026", [])
    tickets, contracts = _load("it_tickets"), _load("contracts")
    suppliers, rooms, depts = _load("suppliers"), _load("meeting_rooms"), _load("departments")
    b: dict = {}

    def add(bucket: str, rec: dict) -> None:
        b.setdefault(bucket, []).append(rec)

    for eid, info in emps.items():
        nm = info["name"]
        if eid in leave:
            rem = leave[eid]["annual_remaining"]
            add("leave", {"domain": "HR", "question": f"{nm}现在年假还剩几天?", "optimal_steps": 2,
                "gold_trajectory": [{"tool": "find_employee_id", "args": {"name": nm}},
                                    {"tool": "get_leave_balance", "args": {"employee_id": eid}},
                                    {"final": f"{nm}当前年假还剩 {rem} 天。"}],
                "reference_outputs": [str(rem)]})
        days = _annual_days(info["hire_date"])
        if days == 0:
            final, ref = (f"{nm}于 {info['hire_date']} 入职,司龄不满 1 年,当年没有法定年假。", ["不满", "没有"])
        else:
            final, ref = (f"{nm}司龄已满 1 年,按政策每年 {days} 天年假,在 OA「假期申请」提交。", [str(days), "OA"])
        add("eligibility", {"domain": "HR", "question": f"{nm}今年有几天年假?", "optimal_steps": 3,
            "gold_trajectory": [{"tool": "find_employee_id", "args": {"name": nm}},
                                {"tool": "get_employee", "args": {"employee_id": eid}},
                                {"tool": "search_knowledge", "args": {"domain": "HR", "query": "年假 工龄 天数"}},
                                {"final": final}],
            "reference_outputs": ref})
        lv = info["level"]
        meal = limits.get(lv, {}).get("meal")
        if meal is not None:
            add("exp_emp", {"domain": "Finance", "question": f"{nm}出差,餐饮每天报销上限是多少?", "optimal_steps": 3,
                "gold_trajectory": [{"tool": "find_employee_id", "args": {"name": nm}},
                                    {"tool": "get_employee", "args": {"employee_id": eid}},
                                    {"tool": "get_expense_limit", "args": {"level": lv, "category": "meal"}},
                                    {"final": f"{nm}职级为 {lv},出差餐饮每日报销上限为 {meal} 元。"}],
                "reference_outputs": [str(meal)]})

    for eid, a in assets.items():
        nm = emps[eid]["name"]
        add("asset", {"domain": "IT", "question": f"{nm}名下的办公笔记本是什么型号?", "optimal_steps": 2,
            "gold_trajectory": [{"tool": "find_employee_id", "args": {"name": nm}},
                                {"tool": "get_it_asset", "args": {"employee_id": eid}},
                                {"final": f"{nm}名下的办公笔记本是 {a['laptop']}。"}],
            "reference_outputs": [a["laptop"]]})

    for h in holidays:
        add("holiday", {"domain": "HR", "question": f"2026 年{h['name']}假期从哪天开始?", "optimal_steps": 1,
            "gold_trajectory": [{"tool": "list_holidays", "args": {"year": 2026}},
                                {"final": f"2026 年{h['name']}假期自 {h['date']} 开始。"}],
            "reference_outputs": [h["date"]]})

    for lv, cats in limits.items():
        add("exp_level", {"domain": "Finance", "question": f"{lv} 职级出差,酒店住宿每天最多能报多少?", "optimal_steps": 1,
            "gold_trajectory": [{"tool": "get_expense_limit", "args": {"level": lv, "category": "hotel"}},
                                {"final": f"{lv} 职级出差酒店住宿每日报销上限为 {cats['hotel']} 元。"}],
            "reference_outputs": [str(cats["hotel"])]})

    for tid, info in tickets.items():
        add("ticket", {"domain": "IT", "question": f"工单 {tid} 现在是什么状态?", "optimal_steps": 1,
            "gold_trajectory": [{"tool": "get_ticket", "args": {"ticket_id": tid}},
                                {"final": f"工单 {tid}({info['type']})当前状态为「{info['status']}」。"}],
            "reference_outputs": [info["status"]]})

    for cid, info in contracts.items():
        add("contract", {"domain": "Legal", "question": f"合同 {cid} 现在审批到哪一步了?", "optimal_steps": 1,
            "gold_trajectory": [{"tool": "get_contract", "args": {"contract_id": cid}},
                                {"final": f"合同 {cid} 当前状态为「{info['status']}」,负责人为 {info['owner']}。"}],
            "reference_outputs": [info["status"]]})

    for nm, info in suppliers.items():
        add("supplier", {"domain": "Procurement", "question": f"供应商「{nm}」的评级和联系方式是什么?", "optimal_steps": 1,
            "gold_trajectory": [{"tool": "get_supplier", "args": {"name": nm}},
                                {"final": f"供应商「{nm}」类别为{info['category']},评级 {info['rating']},联系电话 {info['contact']}。"}],
            "reference_outputs": [info["rating"], info["contact"]]})

    for rid, info in rooms.items():
        add("room", {"domain": "Admin", "question": f"会议室 {rid} 能坐多少人?", "optimal_steps": 1,
            "gold_trajectory": [{"tool": "get_meeting_room", "args": {"room_id": rid}},
                                {"final": f"会议室 {rid} 位于{info['location']},可容纳 {info['capacity']} 人,配有{'、'.join(info['equipment'])}。"}],
            "reference_outputs": [str(info["capacity"])]})

    for dept, info in depts.items():
        add("dept", {"domain": "Admin", "question": f"{dept}部门现在有多少人?", "optimal_steps": 1,
            "gold_trajectory": [{"tool": "get_department", "args": {"name": dept}},
                                {"final": f"{dept}部门现有 {info['headcount']} 人,位于{info['location']}。"}],
            "reference_outputs": [str(info["headcount"])]})

    policy = [
        ("IT", "VPN 配置 连接", "it/vpn", "请问公司 VPN 怎么配置?", "VPN 客户端为 GlobalConnect,服务器 vpn.corp.com、端口 443;连不上可提 IT 工单。", ["vpn.corp.com"]),
        ("IT", "密码 重置 周期", "it/password_reset", "域账号密码多久需要更换一次?", "域账号密码每 90 天需更换一次,新密码至少 12 位。", ["90"]),
        ("Finance", "发票 抬头 要求", "finance/invoice", "报销发票的抬头有什么要求?", "发票抬头须为公司全称,个人抬头或不合规发票一律不予报销。", ["抬头"]),
        ("Legal", "用印 申请 流程", "legal/seal_usage", "用印需要怎么申请?", "在 OA「法务-用印申请」提交并附文件,经法务与授权人审批后由印章管理员盖章。", ["用印申请"]),
        ("Procurement", "采购 比价 流程", "procurement/purchase_process", "采购至少需要几家比价?", "采购需 3 家及以上询价/比价,单笔 5 万元以上须招标或多方比价。", ["3"]),
        ("Admin", "会议室 预订", "admin/meeting_room_booking", "会议室怎么预订?", "在 OA「行政-会议室预订」按时间段预约,15 分钟未签到自动释放。", ["15"]),
    ]
    for dom, query, doc_id, q, final, ref in policy:
        add("policy", {"domain": dom, "question": q, "optimal_steps": 2,
            "gold_trajectory": [{"tool": "search_knowledge", "args": {"domain": dom, "query": query}},
                                {"tool": "get_document", "args": {"doc_id": doc_id}},
                                {"final": final}],
            "reference_outputs": ref})
    return b


def select(buckets: dict, n: int) -> list:
    order = list(buckets.keys())
    idx = {k: 0 for k in order}
    out: list = []
    while len(out) < n:
        progressed = False
        for k in order:
            if idx[k] < len(buckets[k]):
                out.append(buckets[k][idx[k]])
                idx[k] += 1
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
    return out


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    recs = select(build(), n)
    counters: dict = {}
    final = []
    for r in recs:
        code = _DOMAIN_CODE[r["domain"]]
        counters[code] = counters.get(code, 0) + 1
        final.append({"id": f"ekb_{code}_{counters[code]:03d}", "domain": r["domain"],
                      "question": r["question"], "gold_trajectory": r["gold_trajectory"],
                      "reference_outputs": r["reference_outputs"], "optimal_steps": r["optimal_steps"]})
    out_path = _DIR / "tasks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"已生成 {len(final)} 条 -> {out_path}")
    print("各域分布:", dict(Counter(r["domain"] for r in final)))


if __name__ == "__main__":
    main()
