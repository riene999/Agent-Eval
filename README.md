# Agent-Eval

Agent-Eval 是面向工具型 Agent 的轨迹评测与失败诊断平台，覆盖任务运行、过程观测、
指标统计、错误归因、版本对比和训练数据导出。项目已接入 τ-bench Retail 与自建企业
知识问答数据集，可用于比较不同模型、Agent 链路和 Skill 配置。

## 能做什么

| 能力 | 说明 |
| --- | --- |
| 批量评测 | 支持指定数据集、题目范围、模型、Agent、并发数、随机种子和同题多次运行 |
| 多模型切换 | 通过 `models.json` 同时配置多个 OpenAI 兼容模型，运行时用 `--model` 切换 |
| 完整轨迹观测 | 查看每次模型请求与响应、工具调用、工具返回、Token、耗时和最终结果 |
| 多维指标 | 统计成功率、Token、成本、p50/p95 延迟、工具调用数、重复调用率、工具选择和参数正确率 |
| 稳定性评估 | 支持同题多次运行，输出 pass@k 和结果波动 |
| Skill 效果评测 | 支持单 Skill 评测和 N+1 对照，识别新增收益、旧能力退化与额外 Token 开销 |
| 辅助评分与归因 | 可选 LLM-as-Judge，并定位失败轨迹首次偏离步骤、错误类型和修改建议 |
| 回归对比 | 对比两次评测的共同题目，检查准确率、Token、延迟、成本和工具调用是否退化 |
| 轨迹回放 | 按时间顺序查看历史轨迹，并对齐两次运行的工具动作和首次差异 |
| 训练数据导出 | 将标准轨迹与模型失败轨迹导出为 SFT 或结构化 DPO 数据 |
| 可视化控制台 | 在网页中管理数据集、模型、Skill、评测任务、报告、轨迹和训练数据 |

适用于模型选型、Agent 链路比较、Prompt/Skill 优化、上线前回归测试，以及从失败案例
中整理后训练数据。

## 界面预览

<img width="2556" height="1403" alt="image" src="https://github.com/user-attachments/assets/08652b35-3a52-4dce-801f-7445473c5f48" />
<img width="2532" height="1388" alt="image" src="https://github.com/user-attachments/assets/00e04f8e-335d-4851-a281-7b1aaabcc924" />
<img width="2547" height="1387" alt="image" src="https://github.com/user-attachments/assets/c61b279e-568d-4641-95b5-6baf71f90d5c" />
<img width="2550" height="1381" alt="image" src="https://github.com/user-attachments/assets/26445672-5f55-47f9-b459-47b408ffb17c" />
<img width="2546" height="1390" alt="image" src="https://github.com/user-attachments/assets/14cd2be3-7486-4492-afb7-aec7f58528ee" />
<img width="2529" height="1383" alt="image" src="https://github.com/user-attachments/assets/1f21198b-f50c-4d1e-b9b0-a6354d5a0daf" />
<img width="2528" height="1392" alt="image" src="https://github.com/user-attachments/assets/280b985a-0469-4fe9-ade0-5adf55165be5" />

## 快速开始

### 1. 安装

项目需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/riene999/Agent-Eval.git
cd Agent-Eval
uv sync
```

### 2. 准备数据集

自建企业知识问答数据集位于 `data/enterprise_kb/`，克隆项目后可以直接使用。

运行 τ-bench 前，将 τ-bench 仓库放到 `data/tau-bench/`，或者在 `.env` 中设置其路径：

```bash
TAU_BENCH_DATA_DIR=./data/tau-bench
```

### 3. 配置模型

在项目根目录创建 `.env`：

```bash
UPSTREAM_BASE_URL=https://api.deepseek.com
UPSTREAM_API_KEY=sk-你的密钥

AGENT_MODEL=deepseek-chat
USER_MODEL=deepseek-chat

OPENAI_BASE_URL=http://localhost:8080/v1
OPENAI_API_KEY=proxy-placeholder
TAU_BENCH_DATA_DIR=./data/tau-bench
```

需要同时使用多个厂商时，在根目录创建 `models.json`：

```json
{
  "deepseek-chat": {
    "base_url": "https://api.deepseek.com",
    "api_key": "sk-..."
  },
  "qwen-plus": {
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "sk-..."
  },
  "glm-5": {
    "base_url": "https://open.bigmodel.cn/api/paas/v4",
    "api_key": "..."
  }
}
```

`models.json` 的键名就是运行时传给 `--model` 的模型名。项目会根据模型名选择对应
地址和密钥；未命中的模型使用 `.env` 中的默认上游。

### 4. 启动服务

终端 A 启动模型代理：

```bash
uv run python -m proxy.server --port 8080
```

终端 B 启动可视化控制台：

```bash
uv run python -m ui.server --port 8090
```

浏览器打开：

```text
http://127.0.0.1:8090
```

控制台可以直接发起评测，并查看任务进度、历史报告、完整轨迹、两次评测差异和训练数据。

## 运行评测

### 单题与批量

```bash
# τ-bench 单题
uv run python -m runner.run \
  --agent react \
  --task tau_retail_000 \
  --model deepseek-chat \
  --run-id retail_001

# τ-bench 批量运行 30 题
uv run python -m runner.run \
  --agent react \
  --split test \
  --start 0 \
  --count 30 \
  --model deepseek-chat \
  --concurrency 5 \
  --run-id retail_30

# 运行全部企业知识问答任务
uv run python -m runner.run \
  --agent react \
  --tasks all-ekb \
  --model deepseek-chat \
  --concurrency 5 \
  --run-id enterprise_all
```

### 切换 Agent 链路

目前提供 ReAct 与 Plan-Solve：

```bash
uv run python -m runner.run --agent react --count 10 --run-id react_10
uv run python -m runner.run --agent plan_solve --count 10 --run-id plan_solve_10
```

### 稳定性评测

```bash
uv run python -m runner.run \
  --agent react \
  --count 30 \
  --trials 3 \
  --temperature 0.5 \
  --seed 1 \
  --concurrency 5 \
  --run-id react_stability
```

报告会给出每道题的多次结果、pass@1、pass@N 和整体波动。

### LLM 辅助评分与失败归因

```bash
uv run python -m runner.run \
  --agent react \
  --count 30 \
  --model deepseek-chat \
  --llm-judge \
  --attribution \
  --attribution-mode failed_only \
  --judge-model deepseek-chat \
  --run-id retail_diagnosis
```

`--llm-judge` 和 `--attribution` 均为可选开关。归因范围支持：

- `failed_only`：只分析失败题
- `all`：分析全部题目
- `sample_N`：抽取 N 道题分析，例如 `sample_10`

## Skill 管理与效果评测

Skill 可以通过控制台导入和管理。每个 Skill 包含名称、适用领域、使用规则和相关工具
说明，同一 Skill 可以搭配 ReAct 或 Plan-Solve 使用。

### 单 Skill 评测

```bash
uv run python -m runner.skill_eval \
  --mode single \
  --agent react \
  --skill hr_leave \
  --model deepseek-chat \
  --count 100 \
  --concurrency 5 \
  --run-id hr_single_100
```

报告会分别展示该 Skill 对应任务和其他任务的成功率、Token、工具调用和延迟。

### N+1 Skill 评测

```bash
uv run python -m runner.skill_eval \
  --mode n_plus_one \
  --agent react \
  --baseline-skills it_support,finance_expense,admin_service,legal_contract,procurement_supplier \
  --skill hr_leave \
  --model deepseek-chat \
  --count 100 \
  --concurrency 5 \
  --seed 1 \
  --run-id hr_n_plus_one
```

N+1 报告会列出：

- 新增 Skill 后做对的题目
- 新增 Skill 后退化的旧题
- 净增益
- 准确率、Token、工具调用、延迟和成本变化

## 评测指标

| 指标 | 含义 |
| --- | --- |
| Accuracy | 任务最终是否通过任务层判分 |
| Total Tokens | 被测 Agent 的输入与输出 Token，不包含用户模拟器、评分器和归因器 |
| Cost | 根据 `prices.json` 中的模型单价估算 |
| Latency p50/p95 | 每条轨迹中模型调用延迟的中位数和高分位数 |
| Tool Call Count | 工具调用总次数 |
| Redundant Call Rate | 同一工具使用相同参数重复调用的比例 |
| Tool Selection | 与标准调用步骤相比，工具选择的正确程度 |
| Argument Correctness | 与标准调用步骤相比，工具参数的正确程度 |
| pass@k | 同一道题运行多次时，至少成功一次的概率 |
| LLM Score | 可选的任务完成度、规范遵守和执行效率评分 |
| Attribution | 首次偏离步骤、错误类别、置信度、根因和修复建议 |

## 报告、轨迹与回归对比

每次运行会生成：

```text
reports/<run_id>.md
reports/<run_id>.json
trajectories/<agent_id>/<task_id>/<run_id>.jsonl
```

Markdown 报告适合直接阅读，JSON 报告适合程序处理。轨迹页面可以查看模型调用、工具
调用、工具返回、最终结果、辅助评分和失败归因。

命令行也可以比较两份报告：

```bash
uv run python -m analysis.compare \
  --baseline reports/baseline.json \
  --candidate reports/candidate.json \
  --out reports/comparisons/baseline__candidate.json
```

对比结果包括共同题目的新增成功、失败退化，以及准确率、Token、p95 延迟、成本和重复
调用比例变化。设置的回归阈值未通过时，命令会返回非零退出码，可用于自动化回归检查。

离线聚合任意轨迹：

```bash
uv run python -m analysis.metrics --glob "trajectories/**/*.jsonl"
```

## 导出训练数据

### SFT

从数据集标准轨迹导出结构化多轮工具调用示范：

```bash
uv run python -m analysis.export sft \
  --out data/train/sft.jsonl
```

### DPO

从标准轨迹和模型失败轨迹中提取“公共上下文后的正确动作与错误动作”：

```bash
uv run python -m analysis.export dpo \
  --run-id enterprise_react \
  --agent react_agent_v1 \
  --mode failed_only \
  --out data/train/dpo.jsonl
```

导出结果保留 OpenAI 原生 `messages`、`tool_calls` 和工具返回格式，同时生成
`tools.json`，可继续适配 TRL、LLaMA-Factory、verl 等训练框架。

## 输出目录

```text
data/enterprise_kb/   自建企业知识问答数据、知识文档、工具和标准轨迹
data/skills/          可导入的 Skill 配置
data/train/           SFT/DPO 导出结果
trajectories/         每次运行的 JSONL 轨迹与大字段内容
reports/              Markdown/JSON 评测报告与对比报告
```

## 使用提示

- 真实评测会产生模型调用费用，建议先用 1～5 题确认模型与代理配置。
- 需要离线检查流程时，可在代理命令后添加 `--mock`。
- `models.json`、`.env`、轨迹、报告和训练数据默认不应提交模型密钥或敏感业务数据。
- 修改代理或控制台代码后，需要重启对应服务。
