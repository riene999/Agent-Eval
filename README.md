# agent-eval

在 τ-bench 上对不同 Agent 配置做评测的最小框架。与多数只测"对不对"(outcome)的
评测不同,本项目额外补两个效率维度,产出 **三轴**对比:

- **准确率轴**:任务是否做对(τ-bench 官方判分);
- **Token 效率轴**:做对的同时用了多少 token;
- **轨迹简洁度轴**:做对的同时走了多少弯路(工具调用次数、冗余调用率)。

闭环由四部分组成:OpenAI 兼容的**本地代理**(记录每次 LLM 调用)、工具 **wrapper**
(记录每次工具调用)、自写 **ReAct Agent**、以及把上述事件落成 JSONL 后的**离线分析**。

## 快速开始

```bash
# 1. 安装依赖(uv 会自动准备 Python 3.11)
uv sync

# 2. τ-bench 已克隆在 ./data/tau-bench;如在别处,改 .env 的 TAU_BENCH_DATA_DIR

# 3. 配环境变量:编辑项目根目录下的 .env(按文件内中文注释填写),
#    至少把 UPSTREAM_API_KEY 改成真实的 DeepSeek key

# 4. 端到端冒烟:起代理 -> 跑 5 道 retail 题 -> 出指标表
bash scripts/smoke.sh
```

单独跑一道题:

```bash
# 终端 A:起代理(转发到 .env 配置的上游)
uv run python -m proxy.server --port 8080
#   想无 key 离线联调,加 --mock(返回可计 token 的假响应)

# 终端 B:跑一组 (agent, task)
uv run python -m runner.run --agent react --task tau_retail_000 --run-id r1
uv run python -m runner.run --agent react --task math --run-id m1   # 不依赖 tau 的算术题

# 分析
uv run python -m analysis.metrics --glob "trajectories/**/*.jsonl"
```

## 目录结构

```
agent-eval/
├── proxy/            # OpenAI 兼容反向代理 + 轨迹录制基础设施
│   ├── server.py     #   /v1/chat/completions 转发(支持 --mock),记录 llm_call
│   └── recorder.py   #   Event 模型、路径、上下文、JSONL 追加、追踪客户端、.env 加载
├── tools/            # 工具层
│   ├── wrapper.py    #   @traced_tool:旁路记录 tool_call/tool_return
│   ├── dummy_tools.py#   add/multiply,供 MathTask 与测试用
│   └── tau_tools.py  #   把 τ-bench 工具包装成绑定数据库的可调用对象
├── agents/           # 每种 agent 类型一个文件夹
│   ├── base.py       #   BaseAgent 抽象
│   ├── echo_agent/   #   回显 agent(打通空闭环用)
│   └── react_agent/  #   手写 ReAct 循环(OpenAI function calling)
├── tasks/            # 任务层
│   ├── base.py       #   Task 抽象(get_prompt/get_tools/judge + 可选 system_prompt)
│   ├── echo_task.py  #   恒成功任务
│   ├── math_task.py  #   (3+5)*2 算术题
│   └── tau_bench.py  #   τ-bench 任务加载、用户模拟器、judge
├── runner/run.py     # 主入口:解析 (agent, task),跑一次并判分、补 final_output
├── analysis/metrics.py # 离线读 JSONL 算四项指标,聚合成 markdown 表
└── trajectories/     # 运行产物:{agent_id}/{task_id}/{run_id}.jsonl(gitignore)
```

## 核心契约

轨迹是一行一个事件的 JSONL,字段见 `proxy/recorder.py` 的 `Event`:

| event_type | data 主要字段 |
| --- | --- |
| `llm_call` | model, messages, response, prompt_tokens, completion_tokens, latency_ms, role |
| `tool_call` | tool_name, args |
| `tool_return` | tool_name, result, error, latency_ms |
| `final_output` | output, success, error |

`success` 由 task 层判分填入,不是 Agent 自报。代理(LLM 调用)与 wrapper(工具调用)
是两个写入方,靠"读取当前行数 → seq = 行数 → 追加"维持时序递增的 `seq`。

## 扩展点(新增文件,不改主干)

**加一种新 Agent**:在 `agents/` 下建新文件夹(如 `agents/plan_solve/`),实现
`BaseAgent.run(task, run_id)`,在 `agents/<name>/__init__.py` re-export 类,再到
`runner/run.py` 的 `make_agent` 注册一行。轨迹会被代理/wrapper 自动记录。

**加一个新任务/数据集**:写一个 `Task` 子类,实现 `get_prompt / get_tools / judge`
(格式与评分**完全由你定义**,主干不感知),在 `runner/run.py` 的 `make_task` 注册。
合成任务可填 `reference_path_length`(已知最优步数),供后续计算 path_length_ratio。

**注**:`tau-bench` 顶层 `__init__` 会 `import litellm`(仅其内置用户模拟器用)。本项目
用自写的、走代理的用户模拟器,从不触发该路径,故 `tools/tau_tools.py` 注入了 litellm
桩,无需安装这一重依赖。
