"""generate 工具：真实出图（MVP 出定稿 render 的「结构化提示词」一张）。

确认门 gate（两道）：
  1. 必须已有 review=confirmed 的定稿计划；
  2. prompt 必须出自该定稿的 render 结果（rendered_prompt）——提示词不得来路不明。

注册条件（环境能力分层）：tools/__init__.build_tools 仅在 config.image_model
非空（配了生图后端）时注册本工具；未配置时大脑根本看不到 generate。
"""

from __future__ import annotations

from pydantic import BaseModel

from agentkit.runtime import RuntimeCtx
from agentkit.tool import Tool
from tools.review import ReviewState, review_state


class GenerateArgs(BaseModel):
    prompt: str | None = None  # 最终提示词（应来自 render 的结果）；留空 = 用会话里的
    tag: str = "plan"          # 产物标签（默认 plan，出图保存路径据此区分）


def make_generate_tool() -> Tool:
    def _gate(ctx: RuntimeCtx, args: GenerateArgs) -> str | None:
        rs: ReviewState = review_state(ctx)
        if rs.confirmed_plan is None:
            return (
                "[runtime] generate 被门控拦截：还没有 review=confirmed 的定稿计划，"
                "不能出图。请先调用 review 让用户确认，再 render。"
            )
        if rs.rendered_prompt is None:
            return (
                "[runtime] generate 被门控拦截：定稿已确认但还没 render 出最终提示词，"
                "请先调用 render。"
            )
        if args.prompt is not None and args.prompt != rs.rendered_prompt:
            return (
                "[runtime] generate 被门控拦截：提示词必须来自对定稿 render 的结果"
                "（与你传入的不一致）。请用 render 返回的提示词（或留空）。"
            )
        return None

    def _run(ctx: RuntimeCtx, args: GenerateArgs) -> str:
        rs: ReviewState = review_state(ctx)
        prompt = args.prompt or rs.rendered_prompt

        if ctx.generator is None:
            return "[runtime] 当前环境未配置生图后端，无法真实出图。"

        tag = args.tag or "plan"
        path = ctx.generator.generate(prompt, tag=tag)
        return f"[image] 图片已生成（tag={tag}）：{path}"

    return Tool(
        name="generate",
        description=(
            "用最终提示词真实生成一张图片并保存，返回保存路径。"
            "只能在 review=confirmed 且 render 之后调用（提示词出自定稿）。"
        ),
        args_schema=GenerateArgs,
        gate=_gate,
        run=_run,
    )
