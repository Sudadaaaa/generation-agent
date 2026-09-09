"""make_plan 工具：生成 / 修订 / 续写一份 GenerationPlan。

- 工人 = ctx.worker（config 决定 deepseek 或 qwen-lora），复用
  domain/extraction.create_generation_plan（内含格式自检与自修）。
- 三种入口（按此判定）：
  1. **修订**：带了 problems/notes（显式或会话里待修订的 pending）→ 基于上一版计划，
     用评审问题 + 你的意见修订。
  2. **续版**（`base="prev"`，连续会话「在刚才基础上加 XX 再生成」）：无问题/意见，
     但基于上一版已确认计划 + 新需求续写（跨消息的上一版仍在会话状态里）。
  3. **新线程**（默认 `base="auto"` 且无待修订，或显式 `base="new"`）：全新需求 → 清空
     上一主题的评审状态（防定稿/意见串场），从第 1 版生成。
- 生成的新计划记入会话状态（last_plan + version），review 直接评审它。
- 返回：计划 JSON 文本（observation；大脑无需复述，状态已自动追踪）。
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from agentkit.runtime import RuntimeCtx
from agentkit.tool import Tool
from domain.extraction import create_generation_plan
from domain.prompts import build_continue_message, build_revise_message
from domain.schema import GenerationPlan
from tools.review import ReviewState, review_state


class MakePlanArgs(BaseModel):
    requirement: str | None = None  # 用户需求；留空 = 用当前会话需求
    base: str = "auto"    # auto | new | prev：见模块 docstring 三种入口
    prev_plan: str | None = None  # 上一版计划 JSON 文本；留空 = 用会话里最新计划
    problems: list[str] | None = None  # 需修正的评审问题（通常自动带上，可留空）
    notes: list[str] | None = None     # 用户的修改意见（通常自动带上，可留空）


def _previous_plan(args: MakePlanArgs, rs: ReviewState) -> GenerationPlan | None:
    """修订/续版要基于的上一版：显式传参优先，否则会话里最新计划。"""
    if args.prev_plan:
        return GenerationPlan.model_validate_json(args.prev_plan)
    return rs.last_plan


def _begin_fresh_thread(ctx: RuntimeCtx, user_input: str) -> str:
    """新线程：清掉上一主题的评审状态，从第 1 版生成一份全新计划。"""
    ctx.state["review"] = ReviewState()  # 防上一主题的定稿/意见/版本串场
    rs = review_state(ctx)
    rs.version = 1
    plan = create_generation_plan(user_input, llm=ctx.worker)
    rs.last_plan = plan
    return plan.model_dump_json(indent=2)


def make_plan_tool() -> Tool:
    def _run(ctx: RuntimeCtx, args: MakePlanArgs) -> str:
        user_input = args.requirement or ctx.user_input
        rs = review_state(ctx)

        # 修订来源：显式传参优先，否则取会话里待修订的问题/意见；随后清空
        problems = list(args.problems) if args.problems else []
        notes = list(args.notes) if args.notes else []
        if not problems and not notes:
            problems = list(rs.pending_problems)
            notes = list(rs.pending_notes)
        rs.pending_problems = []
        rs.pending_notes = []

        # ---- 修订（问题/意见非空）----
        if problems or notes:
            base = _previous_plan(args, rs)
            if base is None:
                return (
                    "[runtime] make_plan 收到修订要求但没有上一版计划可改。"
                    "请先不带 problems/notes 参数生成一份计划，再走 review。"
                )
            rs.version += 1  # 修订 = 新一版
            context_messages = [
                {"role": "assistant", "content": base.model_dump_json(indent=2)},
                {"role": "user", "content": build_revise_message(problems, notes)},
            ]
            plan = create_generation_plan(
                user_input, llm=ctx.worker, context_messages=context_messages
            )
            rs.last_plan = plan
            return plan.model_dump_json(indent=2)

        # ---- 续版（base=prev：在上一版基础上加新要求）----
        if args.base == "prev":
            base = _previous_plan(args, rs)
            if base is None:
                return (
                    "[runtime] make_plan 收到续版要求（base=prev）但没有上一版计划可续。"
                    "请先不带 base 参数生成一份计划，或在下一句明确新主题。"
                )
            rs.version += 1  # 续版 = 新一版
            context_messages = [
                {"role": "assistant", "content": base.model_dump_json(indent=2)},
                {"role": "user", "content": build_continue_message(user_input)},
            ]
            plan = create_generation_plan(
                user_input, llm=ctx.worker, context_messages=context_messages
            )
            rs.last_plan = plan
            return plan.model_dump_json(indent=2)

        # ---- 新线程（默认 auto / 显式 new：全新需求）----
        return _begin_fresh_thread(ctx, user_input)

    return Tool(
        name="make_plan",
        description=(
            "生成（或修订、续写）一份结构化 GenerationPlan，返回其 JSON 文本。"
            "三种用法：带 problems/notes = 按问题与你的意见修订上一版（会话自动带最近 "
            "review 的问题/意见）；base='prev' = 在上一版计划基础上按当前需求续写 "
            "（用户说『在刚才基础上加 XX』时用这个）；都不带 = 全新需求，从第 1 版生成。"
            "参数大多可留空，会话状态会自动追踪当前需求与上一版计划。"
        ),
        args_schema=MakePlanArgs,
        run=_run,
    )
