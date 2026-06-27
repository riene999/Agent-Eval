# enterprise_kb 数据集(企业知识问答)

模拟企业内部知识问答,覆盖 HR / IT / 财务 / 行政 / 法务 / 采购 等域。工具**只读、无状态**(只查不改),
查的是本目录 `knowledge/` 下的文档与结构化表。一条数据 = 问题 + 你给的**正确轨迹**。

## 目录
```
enterprise_kb/
├── tools.py        # 10 个只读工具(自动索引 knowledge/docs/ 下全部 .md)
├── task.py         # EnterpriseKBTask + load_task(task_id)
├── knowledge/
│   ├── docs/{hr,it,finance}/*.md   # 知识文档(新增"丢文件即生效")
│   └── data/*.json                 # 结构化表(员工/假期/节假日/资产/工单/差旅标准/报销政策)
└── tasks.jsonl     # 数据:一行一条
```

## 一条数据的格式(tasks.jsonl,一行一条)
```jsonc
{
  "id": "ekb_hr_001",            // 全局唯一,以 ekb_ 开头(runner 据此分发)
  "domain": "HR",                // HR / IT / Finance
  "question": "……",             // 喂给 agent 的问题
  "gold_trajectory": [           // ★ 正确轨迹(= judge 依据 / SFT 范文 / DPO 的 chosen)
    {"tool": "工具名", "args": {…}},   // 正确该调的工具+参数
    {"say": "……"},                    // 可选:正确该说的话
    {"final": "……"}                   // 最终正确答案
  ],
  "reference_outputs": ["关键串", …],  // 可选:默认 judge 检查 final 是否含这些(也可改用 --llm-judge)
  "optimal_steps": 2                  // 可选:最少工具步,留给 path_length_ratio
}
```

## 可用工具(只读,15 个)
- 通用检索:`list_domains` · `search_knowledge(domain, query)` · `get_document(doc_id)`
- 人员:`find_employee_id(name)`(名字→编号) · `get_employee(id)` · `get_leave_balance(id)` · `get_it_asset(id)` · `get_department(name)`
- HR/IT:`list_holidays(year)` · `get_ticket(ticket_id)`
- 财务:`get_expense_limit(level, category)` · `get_reimbursement_policy(category)`
- 行政/法务/采购:`get_meeting_room(room_id)` · `get_contract(contract_id)` · `get_supplier(name)`

> 用户问题用**姓名**(不给编号);需要编号时 agent 先 `find_employee_id` 再按编号查。

## 跑一条
```bash
uv run python -m runner.run --agent react --task ekb_hr_001
```

## 扩到 1000 条要做的(无需改代码)
1. 往 `knowledge/docs/<域>/` 丢更多 `.md`(自动被 `search_knowledge` 索引);
2. 往 `knowledge/data/*.json` 补更多员工/工单/资产等条目;
3. 往 `tasks.jsonl` 追加更多行(每行一条,带 gold_trajectory)。
工具/加载器/judge 都已就绪,新增纯数据即可。
