"""领域工序层（无状态；规则单一来源）。

由原 planning/（schema/prompts/extract/render）与 agent/（critique/prompts）
合并而来：所有领域提示词唯一住在 domain/prompts.py，领域工序不含会话状态、
不做用户交互。

再导出旧符号，作为领域包的稳定公共入口（工具层/离线数据构建从这里取）。
注意：llm 传输层不在本包，规划/评审/渲染经由 ChatClient 由调用方注入
（见 llm/ 包，本包函数收 client 参数、不负责构造）。
"""

from domain.critique import CritiqueVerdict, critique_plan
from domain.extraction import PlanFormatError, create_generation_plan, parse_generation_plan
from domain.prompts import (
    CRITIQUE_SYSTEM_PROMPT,
    RENDER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_correction_message,
    build_revise_message,
    build_system_prompt,
)
from domain.rendering import build_final_prompt
from domain.schema import Element, GenerationPlan

__all__ = [
    "Element",
    "GenerationPlan",
    "PlanFormatError",
    "create_generation_plan",
    "parse_generation_plan",
    "critique_plan",
    "CritiqueVerdict",
    "build_final_prompt",
    "SYSTEM_PROMPT",
    "RENDER_SYSTEM_PROMPT",
    "CRITIQUE_SYSTEM_PROMPT",
    "build_system_prompt",
    "build_correction_message",
    "build_revise_message",
]
