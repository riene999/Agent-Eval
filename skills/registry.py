"""Skill 的持久化、查询和安全工具解析。

导入的 Skill 保存在 data/skills。配置只能引用企业知识库已注册工具，既支持用户
自行组合能力包，也避免网页上传的 JSON 获得任意代码执行权限。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from proxy.recorder import PROJECT_ROOT

from .models import SkillSpec


def skill_dir() -> Path:
    raw = os.getenv("AGENT_EVAL_SKILL_DIR")
    return Path(raw).resolve() if raw else PROJECT_ROOT / "data" / "skills"


def _tool_registry() -> dict[str, Callable[..., Any]]:
    data_root = str(PROJECT_ROOT / "data")
    if data_root not in sys.path:
        sys.path.insert(0, data_root)
    from enterprise_kb.tools import TOOLS_BY_NAME

    return TOOLS_BY_NAME


def available_tool_names() -> list[str]:
    return sorted(_tool_registry())


def _path(skill_id: str) -> Path:
    return skill_dir() / f"{skill_id}.json"


def validate_tools(spec: SkillSpec) -> None:
    unknown = sorted(set(spec.tools) - set(_tool_registry()))
    if unknown:
        raise ValueError(f"Skill 引用了未注册工具: {', '.join(unknown)}")


def import_skill(payload: dict[str, Any], *, overwrite: bool = False) -> SkillSpec:
    spec = SkillSpec.model_validate(payload)
    validate_tools(spec)
    path = _path(spec.skill_id)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Skill {spec.skill_id!r} 已存在")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(spec.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return spec


def get_skill(skill_id: str, *, include_disabled: bool = False) -> SkillSpec:
    path = _path(skill_id)
    if not path.exists():
        raise KeyError(f"未找到 Skill {skill_id!r}")
    spec = SkillSpec.model_validate_json(path.read_text(encoding="utf-8"))
    validate_tools(spec)
    if not include_disabled and not spec.enabled:
        raise KeyError(f"Skill {skill_id!r} 已停用")
    return spec


def list_skills(*, include_disabled: bool = True) -> list[SkillSpec]:
    root = skill_dir()
    if not root.exists():
        return []
    result: list[SkillSpec] = []
    for path in sorted(root.glob("*.json")):
        try:
            spec = SkillSpec.model_validate_json(path.read_text(encoding="utf-8"))
            validate_tools(spec)
        except (OSError, ValueError):
            continue
        if include_disabled or spec.enabled:
            result.append(spec)
    return result


def resolve_tools(spec: SkillSpec) -> list[Callable[..., Any]]:
    registry = _tool_registry()
    return [registry[name] for name in spec.tools]
