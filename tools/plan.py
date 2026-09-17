"""plan 工具：调 PlanAgent 把需求拆解成结构化 GenerationPlan，返回其 JSON 文本。

旧版（HEAD:tools/plan.py）那套会话状态——修订 / 续版 base=prev / 新主题清空——
由 PlanAgent 全量保留的 history 自然承接，ReviewState 与 RuntimeCtx 都不再需要。
旧版随时可取回：git show HEAD:tools/plan.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from core.tool import Args, Tool


class PlanArgs(BaseModel):
    """plan 的参数。"""

    requirement: str = Field(
        description=(
            "本次要规划的画面需求，也可以传入对之前计划的修改需求"
        )
    )


@dataclass
class PlanTool(Tool):
    name: str = "plan"
    description: str = (
        "一个计划助手，把一段画面需求拆解成详细的生成计划，可以提高生成的效果。"
        "它负责将需求拆解为生成计划，计划可能存在问题，因此可以接收修改意见二次优化生成计划。"
    )
    args_schema: Args = PlanArgs

    # 装配入口：由 build_tools 构造好传进来。离线测试注假件，不拉起真 LLM。
    # 工具不持有 llm——LLM 是 PlanAgent 的实现细节，工具从头到尾不该碰它。
    plan_agent: Any = None

    def __post_init__(self) -> None:
        if self.plan_agent is None:
            raise ValueError(
                "PlanTool 需要 plan_agent（PlanAgent）。装配走 tools.build_tools(config)；"
                "离线测试传 plan_agent=FakePlanAgent()。"
            )

    def run(self, parameters: PlanArgs) -> str:
        return self.plan_agent.run(parameters.requirement)
