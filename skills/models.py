"""Skill 配置的数据模型与输入校验。

配置保持为普通 JSON，方便在界面导入、版本管理和跨环境复制。工具只记录注册名，
实际可调用函数由 registry 在运行时安全解析。
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator


class SkillSpec(BaseModel):
    """一个可安装的 Agent 能力包。"""

    skill_id: str
    name: str
    description: str
    instructions: str
    tools: list[str] = Field(min_length=1)
    domains: list[str] = Field(default_factory=list)
    version: str = "1.0.0"
    enabled: bool = True

    @field_validator("skill_id")
    @classmethod
    def validate_skill_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_.-]{1,63}", value):
            raise ValueError("skill_id 需以小写字母开头，只能包含小写字母、数字、点、横线和下划线")
        return value

    @field_validator("name", "description", "instructions")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("名称、说明和使用规则不能为空")
        return value

    @field_validator("tools")
    @classmethod
    def unique_tools(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @field_validator("domains")
    @classmethod
    def normalize_domains(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip().lower() for item in value if item.strip()))
