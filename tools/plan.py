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
    # 本轮的编排话术暂时挂在这里：MyAgent 仍是「生图助手」，等它改成通用 agent 时，
    # 下面第 2 段「拿到计划后怎么渲染」应当搬回系统提示词，这里瘦身回纯工具说明。
    description: str = (
        "把用户的图片需求拆解成一份结构化计划（GenerationPlan JSON），返回其 JSON 文本。"
        "拿到计划后，你必须自己把它渲染成一段自然连贯的中文提示词，再交给 generate_image"
        "——不要把 JSON 原样传下去。渲染时：全局属性（风格/构图/光线/画幅/多宫格框架）"
        "写在开头；每个元素按 位置→称呼→外表→动作 写成自然句；正文里不要出现字段名，"
        "也不要把 identity 里的字母编号（如「牛仔A」）照搬进去，仅作为plan结构层面的区分。"
        "本工具记得上一版计划：如果传入「在刚才基础上加/改 XX」时，工具仍旧可以正常运行"
        "不用把旧 JSON 传回来；这个工具通过子agent实现，能记住上次对话的内容。"
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
