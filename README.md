# agent-eval

在 τ-bench 上对 Agent 配置做**多维度**评测的最小框架。与多数只测"对不对"(outcome)的评测不同,本项目额外补两个效率维度,产出**三轴**对比:

- **准确率**:任务是否做对(τ-bench 官方判分)
- **Token 效率**:做对的同时用了多少 token(只计被测 Agent,排除用户模拟器/评测器)
- **轨迹简洁度**:工具调用次数、冗余调用率(走了多少弯路)

可选再叠加两层 LLM 维度的洞察:**LLM-as-judge**(质量打分)与**错误归因**(定位从哪一步开始偏离)。

## 架构总览

```
                      ┌─ tools/wrapper.py  @traced_tool ──┐ 记录 tool_call / tool_return
runner ─ ReAct Agent ─┤                                    ├─→ trajectories/.../{run}.jsonl
                      └─ proxy(OpenAI 兼容)──────────────┘ 记录 llm_call(转发到真实厂商)
                              │ 按 model 名路由(models.json)→ DeepSeek / 智谱 / 通义 …
跑后(可选)            ├─ LLM-judge   → llm_judge 事件
                              └─ 错误归因     → attribution 事件
analysis/metrics.py ── 读 JSONL → 三轴指标 + 红绿灯 markdown 报告
```

四类核心事件 `llm_call / tool_call / tool_return / final_output` 构成 Agent 轨迹;评测开关打开时再追加 `llm_judge / attribution` 两类。手写 ReAct,**不依赖 LangChain/LangGraph 等框架**。

## 快速开始

```bash
# 1. 安装依赖(uv 会自动准备 Python 3.11)
uv sync

# 2. τ-bench 已克隆在 ./data/tau-bench;若在别处,改 .env 的 TAU_BENCH_DATA_DIR

# 3. 配 .env(见下)与 models.json(多厂商时,见下)
```

`.env`(项目根目录,已 gitignore):

```bash
# 上游默认厂商(models.json 里没命中的模型回退到这)
UPSTREAM_BASE_URL=https://api.deepseek.com
UPSTREAM_API_KEY=sk-你的key
# 被测 Agent 与用户模拟器默认模型
AGENT_MODEL=deepseek-chat
USER_MODEL=deepseek-chat
# Agent 如何访问本地代理(代理再注入真实上游 key;此 key 占位即可)
OPENAI_BASE_URL=http://localhost:8080/v1
OPENAI_API_KEY=proxy-placeholder
# tau-bench 仓库根目录
TAU_BENCH_DATA_DIR=./data/tau-bench
```

## 运行

需要两个终端:终端 A 起代理(常驻),终端 B 跑评测。

```bash
# 终端 A:启动 OpenAI 兼容代理(转发到 .env / models.json 配置的上游)
uv run python -m proxy.server --port 8080
#   想无 key 离线联调:加 --mock(返回可计 token 的假响应)

# 终端 B:单题
uv run python -m runner.run --agent react --task tau_retail_000 --run-id r1
uv run python -m runner.run --agent react --task math --run-id m1      # 不依赖 tau 的算术题

# 终端 B:批量(可配 split 与题数)
uv run python -m runner.run --agent react --split test --count 30 --run-id batch1
uv run python -m runner.run --agent react --split train --start 20 --count 5
uv run python -m runner.run --agent react --tasks tau_retail_003,tau_retail_050   # 指定若干题

# 选模型(发给当前上游的模型名;按 models.json 路由)
uv run python -m runner.run --agent react --model glm-4.6 --count 10 --run-id glm

# 叠加 LLM 评测(可自由开关)
uv run python -m runner.run --agent react --count 30 --llm-judge \
    --attribution --attribution-mode failed_only --judge-model deepseek-chat

# 离线聚合指标(跨任意轨迹通配)
uv run python -m analysis.metrics --glob "trajectories/**/*.jsonl"

# 一键端到端冒烟(起代理→跑 5 道 retail→出表)
bash scripts/smoke.sh
```

常用参数:
- `--split` `test|train|dev`(默认 test);`--count N`(从 `--start` 起跑 N 道);`--tasks a,b,c`(指定题,覆盖 count)
- `--model`(被测模型,默认 `.env` 的 `AGENT_MODEL`);`--run-id`(批次标签,不填随机)
- `--llm-judge`(开启质量打分);`--attribution`(开启归因)+ `--attribution-mode` `failed_only`(默认)`| all | sample_N`;`--judge-model`(评测模型,默认复用 `--model`)
- **每次运行都会在 `reports/<run_id>.md` 生成一份报告**

## 多模型 / 多厂商路由(models.json)

并列预置各模型的 url+key,跑时 `--model` 一键切换,无需临时改配置。`models.json`(项目根目录,已 gitignore,含密钥):

```json
{
  "deepseek-chat":  {"base_url": "https://api.deepseek.com",                 "api_key": "sk-..."},
  "glm-4.6":        {"base_url": "https://open.bigmodel.cn/api/paas/v4",     "api_key": "..."},
  "qwen-plus":      {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": "sk-..."}
}
```

- **键名 = 你 `--model` 传的名字 = 发给该厂商的 `model` 字段**,必须填该厂商真实支持的型号名。
- 代理按请求里的 `model` 查表路由;**没命中的模型回退到 `.env` 的 `UPSTREAM_*`**。
- `models.json` 是**热读**的(每次请求读),增删模型无需重启代理;但**改了代理代码**需重启代理。
- 仅支持 OpenAI 协议兼容的厂商即可零代码接入;不兼容的(如 Anthropic 原生)只需改 `proxy/server.py` 的 `_forward_upstream` 做格式转换。

## 目录结构

```
agent-eval/
├── proxy/
│   ├── server.py     # OpenAI 兼容代理:按 model 路由转发、记录 llm_call、--mock
│   └── recorder.py   # Event 模型、路径、上下文、JSONL 追加、内容寻址 blob、追踪客户端、.env
├── tools/
│   ├── wrapper.py    # @traced_tool:旁路记录 tool_call/tool_return
│   ├── dummy_tools.py# add/multiply(MathTask 用)
│   └── tau_tools.py  # 把 τ-bench 工具包装成绑定数据库的可调用对象
├── agents/           # 每种 agent 类型一个文件夹
│   ├── base.py       #   BaseAgent 抽象
│   ├── echo_agent/   #   回显 agent(打通空闭环用)
│   └── react_agent/  #   手写 ReAct 循环(OpenAI function calling)
├── tasks/
│   ├── base.py       # Task 抽象(get_prompt/get_tools/judge + 可选 system_prompt/user_turn/goal_text/reference_summary)
│   ├── echo_task.py / math_task.py / tau_bench.py
├── evaluators/       # 每种评测器一个文件夹
│   ├── base.py / transcript.py
│   ├── llm_judge/    #   LLM-as-judge 质量打分
│   └── attributor/   #   错误归因(偏离点定位)
├── runner/run.py     # 单题/批量入口 + 判分 + 可选评测,产出报告
├── analysis/
│   ├── metrics.py    # 离线指标 + 红绿灯 markdown 报告
│   ├── pareto.py     # 三轴 ASCII 帕累托前沿
│   └── export.py     # 评测产物 → 后训练数据(SFT / DPO)
├── data/enterprise_kb/ # 自建企业知识问答数据集(工具+文档+gold,git 跟踪)
├── models.json       # 多厂商路由(gitignore,含密钥)
├── data/tau-bench/   # 第三方基准克隆(gitignore)
├── trajectories/     # 轨迹 {agent}/{task}/{run}.jsonl + blobs/(gitignore)
└── reports/          # 每次运行的报告 <run_id>.md(gitignore)
```

## 轨迹格式与 blob 寻址

一行一个事件的 JSONL,字段见 `proxy/recorder.py` 的 `Event`:

| event_type | data 主要字段 |
| --- | --- |
| `llm_call` | role, model, request(完整请求体), response(完整响应), prompt_tokens, completion_tokens, latency_ms |
| `tool_call` | tool_name, args |
| `tool_return` | tool_name, result, error, latency_ms |
| `final_output` | output, success, error |
| `llm_judge`(可选) | overall, dimensions, reason |
| `attribution`(可选) | deviation_seq, error_category, confidence, recoverable, summary, root_cause_hypothesis, fix_suggestion |

- **内容寻址 blob**:超过阈值(默认 1KB,`BLOB_THRESHOLD_BYTES`)的大字段(system prompt、tools schema、长工具返回等)按 sha256 外置到 `trajectories/blobs/`,事件里只留 `{"$blob": "<hash>"}` 引用。相同内容只存一份——重复的 wiki / 历史消息全局去重;`recorder.rehydrate()` 可递归还原完整内容。
- **seq**:单调递增整数(按写入行号分配),即使代理与 Agent 跨进程写同一文件也保序;`final_output.success` 由 task 层判分填入,不是 Agent 自报。
- **向后兼容**:没有 `llm_judge/attribution` 事件、或没有 blob 引用的旧轨迹,分析端都能正常读(自动跳过/透传)。

## 报告

每次运行写 `reports/<run_id>.md`:顶部红绿灯总览(🟢🟡🔴)、每题 ✅/❌ + token 条 + 冗余灯、规则失败原因;开了评测则追加"LLM 评分"与"失败归因(第 N 步:……)"两段。目标是让不了解评测的人也能一眼看出好坏。

## 导出后训练数据(SFT / DPO)

`analysis/export.py` 把评测产物转成后训练数据,喂给训练框架(LLaMA-Factory / TRL / verl 等)。输出默认落 `data/train/`(已 gitignore)。

| | SFT | DPO |
| --- | --- | --- |
| 教什么 | 照着「正确示范」学工具调用 | 同一题:好答案 vs 坏答案,学会偏好前者 |
| 原料 | **只要数据集 gold**(无需跑模型) | **gold(chosen) + 模型实际失败轨迹(rejected)**,须先跑评测 |
| 产物 | `sft.jsonl`(每行 `{messages:[...]}`)+ `tools.json`(工具定义) | `dpo.jsonl`(每行 `{prompt, chosen, rejected, task_id, success, run_id}`) |

```bash
# SFT:从 gold 渲染多轮 function-calling,不依赖任何评测run
uv run python -m analysis.export sft --out data/train/sft.jsonl

# DPO:从某次评测的失败题导出偏好对(chosen=gold,rejected=模型实跑)
uv run python -m analysis.export dpo --run-id ekb_qw8b_react --out data/train/dpo.jsonl

# 多实验一并导入(空格分隔)
uv run python -m analysis.export dpo --run-id exp1 exp2 --out data/train/dpo.jsonl
# 后续新实验增量并入(自动按 (题目,rejected) 去重)
uv run python -m analysis.export dpo --run-id exp3 --append --out data/train/dpo.jsonl
```

DPO 选项:`--agent`(默认 `react_agent_v1`)、`--mode failed_only`(默认,只挑失败题)`| all`(连"做对但绕路"的也收,教简洁)、`--append`(追加去重)。

> 注意:DPO 按 `--run-id` **精确匹配** `<run_id>.jsonl` 与多试验的 `<run_id>_t*.jsonl`,不会前缀误吸其它实验(如 `ekb_ds_100_react` 不会把 `ekb_ds_100_react_qw` 卷进来)。
>
> 工具不写进 system prompt 文字、而是靠 `tools` 字段传——这是 function-calling 规范。训练时需用框架的 chat template 把 `tools.json` 与每条样本绑定(LLaMA-Factory 的 `tools` 字段 / TRL 的 `apply_chat_template(..., tools=...)`)。

## 扩展点(新增文件,不改主干)

- **加一种 Agent**:`agents/` 下建文件夹实现 `BaseAgent.run`,在 `runner/run.py` 的 `make_agent` 注册一行。
- **加一个数据集/任务**:写 `Task` 子类实现 `get_prompt/get_tools/judge`(格式与评分完全自定义);合成任务可填 `reference_path_length`(已知最优步数)用于后续 path_length_ratio 指标。
- **加一种评测器**:`evaluators/` 下建文件夹实现 `Evaluator.evaluate`。
- **加一个模型/厂商**:往 `models.json` 加一条;OpenAI 兼容则零代码。

## 备注

- `tau-bench` 顶层 `__init__` 会 `import litellm`(仅其内置用户模拟器用)。本项目用自写的、走代理的用户模拟器,从不触发该路径,故 `tools/tau_tools.py` 注入 litellm 桩,无需安装这一重依赖。
- 验证方式是**对真实 LLM 跑**(`runner.run` 跑真实题后看轨迹/报告),不使用 mock/fixture 单测。
- 国内环境若开了全局/TUN 代理,可能导致代理→某些国内厂商(如 dashscope)的 TLS 被打断;此时给对应域名加直连规则,或换用可达的厂商端点。
