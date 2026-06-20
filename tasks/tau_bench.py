"""τ-bench(retail)任务接入与判分。

每个 TauBenchTask 封装一道 retail 任务:用 tau 的政策 wiki+规则作 system 提示,用自写
的用户模拟器(走代理、记为 role=user_sim)扮演客户,DB 工具与 respond_to_user 一并暴露
给 Agent。judge 完全复刻 tau-bench 官方逻辑:在新数据库上重放 gold actions 得目标哈希,
与 Agent 实际操作后的数据库哈希比对,并检查必需输出子串是否出现在发给客户的话里。
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from proxy.recorder import Event, make_client
from tasks.base import Task
from tools.tau_tools import (
    RESPOND_ACTION_NAME,
    RULES,
    TOOLS_MAP,
    WIKI,
    consistent_hash,
    load_data,
    load_tasks,
    make_db_tools,
    to_hashable,
)

def tau_task_id(split: str, index: int) -> str:
    """task_id 命名:test 集用短名,其它 split 把名字带上,避免轨迹目录冲突。"""
    return f"tau_retail_{index:03d}" if split == "test" else f"tau_retail_{split}_{index:03d}"


# 供 smoke 脚本等使用的默认 5 道题(取 test 集前 5 个,按 index 确定、可复现、不挑答案)
DEFAULT_TASK_IDS = [tau_task_id("test", i) for i in range(5)]

# 对话方式说明,追加在政策 wiki 之后
_DIALOG_NOTE = (
    "你直接用自然语言与客户对话:输出一段文字就会发送给客户并拿到客户的回复。"
    "需要查询或修改后台数据时调用相应工具(每次只调用一个工具,且调用工具时不要同时回复客户)。"
    "当客户的诉求已全部解决,用一句话向客户确认即可,对话会自然结束。"
)

# 用户模拟器 system 提示,照搬 tau-bench LLMUserSimulationEnv 的措辞以保持行为一致
_USER_SYSTEM_TMPL = """You are a user interacting with an agent.

Instruction: {instruction}

Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""


class UserSimulator:
    """LLM 用户模拟器:走代理调用,客户的话即模型的 assistant 输出。

    消息列表是"从模拟器视角"组织的:role=user 是 Agent 说的话,role=assistant 是
    客户(模拟器)说的话——与 tau-bench 的约定一致。
    """

    def __init__(self, instruction: str, model: Optional[str] = None) -> None:
        self.client = make_client("user_sim")
        self.model = model or os.getenv("USER_MODEL", "deepseek-chat")
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _USER_SYSTEM_TMPL.format(instruction=instruction)},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]

    def _generate(self) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=self.messages, temperature=0.0
        )
        content = resp.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": content})
        return content

    def start(self) -> str:
        return self._generate()

    def step(self, agent_message: str) -> str:
        self.messages.append({"role": "user", "content": agent_message})
        return self._generate()


class TauBenchTask(Task):
    def __init__(self, task_index: int, split: str = "test") -> None:
        self.task_index = task_index
        self.split = split
        self.task_id = tau_task_id(split, task_index)
        self.tau_task = load_tasks(split)[task_index]
        # Agent 与 judge 共享同一份数据库:工具调用会就地修改它
        self.data: Dict[str, Any] = load_data()
        self._user_sim: Optional[UserSimulator] = None
        self._tools: Optional[List[Callable[..., Any]]] = None
        # Agent 发给客户的每段话,judge 据此检查必需输出子串
        self.agent_messages: List[str] = []

    def _ensure_ready(self) -> None:
        # 延迟初始化:用户模拟器需要在 run_context 内构造客户端
        if self._user_sim is None:
            self._user_sim = UserSimulator(self.tau_task.instruction)
            self._tools = make_db_tools(self.data)

    def system_prompt(self) -> str:
        rules = "\n".join(f"- {r}" for r in RULES)
        return f"{WIKI}\n\n# 规则\n{rules}\n\n# 对话方式\n{_DIALOG_NOTE}"

    def get_prompt(self) -> str:
        self._ensure_ready()
        assert self._user_sim is not None
        return self._user_sim.start()

    def get_tools(self) -> List[Callable[..., Any]]:
        self._ensure_ready()
        assert self._tools is not None
        return self._tools

    def user_turn(self, message: str) -> Optional[str]:
        # Agent 的自然语言即对客户说的话:记录后交给用户模拟器,返回客户回复
        self._ensure_ready()
        assert self._user_sim is not None
        self.agent_messages.append(message)
        return self._user_sim.step(message)

    def judge(self, final_output: str, trajectory: List[Event]) -> Dict[str, Any]:
        # 1) 数据库正确性:Agent 操作后的哈希 == 新库上重放 gold actions 后的哈希
        agent_hash = consistent_hash(to_hashable(self.data))
        fresh = load_data()
        for action in self.tau_task.actions:
            if action.name == RESPOND_ACTION_NAME:
                continue
            cls = TOOLS_MAP.get(action.name)
            if cls is None:
                continue
            try:
                cls.invoke(data=fresh, **action.kwargs)
            except Exception:
                pass
        gt_hash = consistent_hash(to_hashable(fresh))
        r_actions = agent_hash == gt_hash

        # 2) 必需输出:逐个子串需出现在发给客户的话(或最终输出)里
        said = self.agent_messages + [final_output]
        haystack = " ".join(said).lower().replace(",", "")
        missing = [o for o in self.tau_task.outputs if o.lower() not in haystack]
        r_outputs = len(missing) == 0

        success = bool(r_actions and r_outputs)
        reason = f"r_actions={r_actions} r_outputs={r_outputs}"
        if missing:
            reason += f" missing_outputs={missing}"
        return {"success": success, "score": 1.0 if success else 0.0, "reason": reason}


def load_tau_task(task_id: str) -> TauBenchTask:
    # 形如 tau_retail_020(test 集)或 tau_retail_train_020(指定 split)
    body = task_id[len("tau_retail_"):] if task_id.startswith("tau_retail_") else task_id
    if "_" in body:
        split, raw_idx = body.rsplit("_", 1)
    else:
        split, raw_idx = "test", body
    try:
        index = int(raw_idx)
    except ValueError as e:
        raise SystemExit(f"无法解析 tau 任务编号: {task_id!r}") from e
    tasks = load_tasks(split)
    if not (0 <= index < len(tasks)):
        raise SystemExit(f"tau 任务编号越界: {index}(split={split},共 {len(tasks)} 道)")
    return TauBenchTask(index, split)
