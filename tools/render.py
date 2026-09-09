"""render 工具：把 review=confirmed 的定稿计划渲染成最终提示词。

确认门 gate：必须已有定稿（review=confirmed）才放行；渲染出的提示词记入
会话状态（rendered_prompt），generate 只能用这一份（提示词必须出自定稿）。
"""

from __future__ import annotations

from pydantic import BaseModel

from agentkit.runtime import RuntimeCtx
from agentkit.tool import Tool
from domain.rendering import build_final_prompt
from domain.schema import GenerationPlan
from tools.review import ReviewState, review_state


class RenderArgs(BaseModel):
    plan: str | None = None  # 定稿计划 JSON 文本；留空 = 用已确认的定稿


def make_render_tool() -> Tool:
    def _gate(ctx: RuntimeCtx, args: RenderArgs) -> str | None:
        rs: ReviewState = review_state(ctx)
        if rs.confirmed_plan is None:
            return (
                "[runtime] render 被门控拦截：还没有 review=confirmed 的定稿计划。"
                "请先调用 review，在评审会中让用户确认定稿。"
            )
        return None

    def _run(ctx: RuntimeCtx, args: RenderArgs) -> str:
        rs: ReviewState = review_state(ctx)
        plan = rs.confirmed_plan
        assert plan is not None  # gate 已保证

        if args.plan:
            candidate = GenerationPlan.model_validate_json(args.plan)
            if candidate != plan:
                return (
                    "[runtime] render 只能渲染已确认的定稿计划；"
                    "你传入的计划与定稿不一致，请用定稿（或留空让会话自动用定稿）。"
                )

        prompt = build_final_prompt(plan, llm=ctx.worker, user_input=ctx.user_input)
        rs.rendered_prompt = prompt
        return prompt

    return Tool(
        name="render",
        description=(
            "把已确认的定稿计划渲染成一段可直接出图的最终提示词（自然语言）。"
            "只能在 review 返回 confirmed 之后调用；返回的提示词文本就是给 generate 用的。"
        ),
        args_schema=RenderArgs,
        gate=_gate,
        run=_run,
    )
