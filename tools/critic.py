"""critic 工具：把需求 + 一份计划交给人看一遍，返回问题清单。

与 PlanTool 同构：工具只管参数与回执，评审这件事整个在 CriticAgent 里。

**要不要调用由主 agent 判断**——这里不设任何自动触发。需求简单、一次就拆清楚时
不必审；需求复杂、含多个主体与互动，或计划已经改过几轮时，值得审一次。
审出问题之后怎么办（让 plan 重出一版 / 先讲给用户听）同样由主 agent 定。

因此本工具是**无状态**的：不需要 ctx.state、不需要确认门，也不持有 LLM 句柄
——CriticAgent 自带。旧版 tools/review.py 那套人机评审会才需要那些，它没有迁过来。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from core.tool import Args, Tool


class CriticArgs(BaseModel):
    """critic 的参数。"""

    requirement: str = Field(
        description="要评审的那条需求原文（或对计划的修改要求），用来判断计划是否忠实于它"
    )

    plan: str = Field(
        description="待评审的 GenerationPlan JSON 文本，通常是 plan 工具刚返回的那份"
    )


@dataclass
class CriticTool(Tool):
    name: str = "critic"
    description: str = (
        "一个专业的评审员，把一份 GenerationPlan 对照需求看一遍，给出评分和修改意见，"
        "返回的是JSON格式的内容，包括评分，修改意见和有几个问题。"
    )
    args_schema: Args = CriticArgs

    # 装配入口：由 build_tools 构造好传进来。离线测试注假件，不拉起真 LLM。
    # 与 PlanTool 同理：工具不持有 llm——LLM 是 CriticAgent 的实现细节，工具不该碰它。
    critic_agent: Any = None

    def __post_init__(self) -> None:
        if self.critic_agent is None:
            raise ValueError(
                "CriticTool 需要 critic_agent（CriticAgent）。装配走 tools.build_tools(config)；"
                "离线测试传 critic_agent=FakeCriticAgent()。"
            )

    def run(self, parameters: CriticArgs) -> str:
        return self.critic_agent.run(parameters.requirement, parameters.plan)
