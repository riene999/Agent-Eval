"""τ-bench(retail)工具与判分素材的接入层。

把 tau-bench 的 Tool 类(静态 invoke(data, **kwargs))包装成绑定到某份数据库 dict
的独立函数,并加上 @traced_tool 旁路记录;同时再导出 tau 的数据加载、政策 wiki、
规则、任务集与官方哈希判分算法,供 TauBenchTask 复用。

tau-bench 顶层 __init__ 会 import litellm(仅其内置用户模拟器使用),而本项目用自写
的、走代理的用户模拟器,从不触发该路径,故注入 litellm 桩以避免引入这一重依赖。
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, List

from proxy.recorder import PROJECT_ROOT
from tools.wrapper import traced_tool


def _ensure_tau_importable() -> None:
    tau_dir = os.environ.get("TAU_BENCH_DATA_DIR") or str(PROJECT_ROOT / "data" / "tau-bench")
    tau_dir = str(Path(tau_dir).resolve())
    if not (Path(tau_dir) / "tau_bench").is_dir():
        raise SystemExit(
            f"未找到 tau_bench 包,请设置 TAU_BENCH_DATA_DIR 指向 tau-bench 仓库根目录(当前: {tau_dir})"
        )
    if tau_dir not in sys.path:
        sys.path.insert(0, tau_dir)
    if "litellm" not in sys.modules:
        try:
            importlib.import_module("litellm")
        except ModuleNotFoundError:
            stub = types.ModuleType("litellm")

            def _stub_completion(*_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError("litellm 桩被调用:本项目不应触发 tau-bench 的 LLM 路径")

            stub.completion = _stub_completion  # type: ignore[attr-defined]
            sys.modules["litellm"] = stub


_ensure_tau_importable()

from tau_bench.envs.base import consistent_hash, to_hashable  # noqa: E402
from tau_bench.envs.retail.data import load_data  # noqa: E402
from tau_bench.envs.retail.rules import RULES  # noqa: E402
from tau_bench.envs.retail.tools import ALL_TOOLS  # noqa: E402
from tau_bench.envs.retail.wiki import WIKI  # noqa: E402
from tau_bench.types import RESPOND_ACTION_NAME  # noqa: E402

# 工具名 -> Tool 类,judge 重放 gold actions 时用
TOOLS_MAP: Dict[str, Any] = {
    cls.get_info()["function"]["name"]: cls for cls in ALL_TOOLS
}


def load_tasks(split: str = "test") -> Any:
    """按 split 返回 retail 任务列表(test/train/dev)。"""
    if split == "test":
        from tau_bench.envs.retail.tasks_test import TASKS_TEST as tasks
    elif split == "train":
        from tau_bench.envs.retail.tasks_train import TASKS_TRAIN as tasks
    elif split == "dev":
        from tau_bench.envs.retail.tasks_dev import TASKS_DEV as tasks
    else:
        raise SystemExit(f"未知 split: {split!r}(可选 test/train/dev)")
    return tasks


def _wrap_tool(tool_cls: Any, data: Dict[str, Any]) -> Callable[..., Any]:
    schema = tool_cls.get_info()
    name = schema["function"]["name"]

    def _impl(**kwargs: Any) -> Any:
        return tool_cls.invoke(data=data, **kwargs)

    _impl.__name__ = name
    _impl.__doc__ = schema["function"].get("description", "")
    wrapped = traced_tool(_impl)
    wrapped._openai_tool_schema = schema  # type: ignore[attr-defined]
    return wrapped


def make_db_tools(data: Dict[str, Any]) -> List[Callable[..., Any]]:
    """返回绑定到给定数据库 dict、且已被 @traced_tool 包装的全部 retail 工具。"""
    return [_wrap_tool(cls, data) for cls in ALL_TOOLS]


__all__ = [
    "make_db_tools",
    "TOOLS_MAP",
    "load_data",
    "WIKI",
    "RULES",
    "load_tasks",
    "RESPOND_ACTION_NAME",
    "to_hashable",
    "consistent_hash",
]
