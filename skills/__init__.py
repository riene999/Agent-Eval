"""Agent-Eval 的可导入 Skill 能力包。

Skill 由说明、规则、适用业务域和已注册工具组成。运行时只解析项目中已有工具，
不会执行导入文件里的任意代码。
"""

from .models import SkillSpec
from .registry import get_skill, import_skill, list_skills, resolve_tools

__all__ = ["SkillSpec", "get_skill", "import_skill", "list_skills", "resolve_tools"]
