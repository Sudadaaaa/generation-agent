"""review 工具：人机评审会（机器评审 + 你把关 = 一道「确认门」）。

流程语义（从 agent_plan_review 迁入并冻结，为新主流程唯一实现）：
  1. 机器评审先跑（domain/critique.critique_plan，fail-open；历轮你的补充意见经
     human_notes 累计传入 user_additions，避免把你要求新增的内容误判成编造）；
  2. 展示计划与机器问题，然后沿用 agent_plan_review 的问答：
       - 机器发现问题时：① 回车=按问题修正 / p=强制通过 / q=放弃；①处直接写文字
         当作一条你的意见（跳过②）；
       - 机器评审通过时：回车=确认通过 / 直接输入文字=你的意见（再修一轮）/ q=放弃。
  3. 结果三种：confirmed（你认可，成为定稿）/ revise（问题+意见交大脑再修订）/
     aborted（你放弃）。

提示语（三个问答行）与 R0 前的 agent_plan_review 逐字一致（tests 冻结断言比对，勿改）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import BaseModel

from agentkit.io import IO
from agentkit.runtime import RuntimeCtx
from agentkit.tool import Tool
from domain.critique import critique_plan
from domain.schema import GenerationPlan


# ---- 评审/确认门的会话状态（挂在 ctx.state["review"]）--------------------
@dataclass
class ReviewState:
    """一条需求内、计划从生成到定稿的会话状态（由各工具读写，不靠大脑记忆）。"""

    version: int = 0                       # 当前计划版本（展示用，从 1 起）
    last_plan: GenerationPlan | None = None
    confirmed_plan: GenerationPlan | None = None  # review=confirmed 定稿（确认门）
    human_notes: list[str] = field(default_factory=list)   # 累计意见（评审 user_additions）
    pending_problems: list[str] = field(default_factory=list)  # 待修订问题
    pending_notes: list[str] = field(default_factory=list)    # 待修订用户意见
    rendered_prompt: str | None = None     # 定稿 render 出的最终提示词


def review_state(ctx: RuntimeCtx) -> ReviewState:
    """取（必要时初始化）评审状态。"""
    st = ctx.state.get("review")
    if st is None:
        st = ReviewState()
        ctx.state["review"] = st
    return st


# ---- 冻结的问答提示语（勿改；改则破坏 legacy 回归对比）-------------------
_PROMPT_REVISE = "① 回车 = 按评审问题修正一轮 | p = 强制通过 | q = 放弃本次\n> "
_PROMPT_OPINION = "② 你的修改意见？（回车=无；会与评审问题一起回传）\n> "
_PROMPT_CONFIRM = (
    "  回车 = 确认通过；或直接输入你的修改意见（按意见再修正一轮）；q = 放弃本次\n> "
)

_QUIT_WORDS = {"q", "quit", "exit", "放弃"}
_PASS_WORDS = {"p", "pass", "通过", "确认"}


def _is_quit(text: str) -> bool:
    return text.strip().lower() in _QUIT_WORDS


def _is_pass(text: str) -> bool:
    return text.strip().lower() in _PASS_WORDS


def review_meeting(
    io: IO,
    *,
    user_input: str,
    plan: GenerationPlan,
    critic,
    human_notes: list[str],
    version: int,
) -> dict:
    """开一场人机评审会（单轮）。

    直接消费/修改 human_notes（历轮你的补充意见累计：随修订回传、也给评审当
    user_additions）。返回 {status: confirmed|aborted|revise, problems, notes}。
    """
    io.out(f"\n[review] 当前 GenerationPlan（第 {version} 版）：")
    io.out(plan.model_dump_json(indent=2))

    verdict = critique_plan(user_input, plan, critic, user_additions=human_notes)

    if verdict.ok:
        # 机器评审通过 → 仍需你最终确认（或补意见再修一轮）
        io.out("[review] 机器评审通过（未发现问题）。请最终确认：")
        reply = io.ask(_PROMPT_CONFIRM).strip()

        if _is_quit(reply):
            io.out("[review] 本次已放弃。")
            return {"status": "aborted", "problems": [], "notes": []}

        if not reply or _is_pass(reply):
            io.out("[review] 计划已由你确认通过（定稿）。")
            return {"status": "confirmed", "problems": [], "notes": []}

        problems: list[str] = []
        notes = [reply]

    else:
        # 机器发现问题 → 两段式问答
        io.out(f"[review] 机器评审发现 {len(verdict.problems)} 个问题：")
        for problem in verdict.problems:
            io.out(f"  - {problem}")

        act = io.ask(_PROMPT_REVISE).strip()

        if _is_quit(act):
            io.out("[review] 本次已放弃。")
            return {"status": "aborted", "problems": [], "notes": []}

        if _is_pass(act):
            io.out("[review] 你强制通过，计划已确认（定稿）。")
            return {"status": "confirmed", "problems": [], "notes": []}

        problems = verdict.problems
        if act:
            notes = [act]  # ①处直接写文字 → 当作唯一一条意见，跳过②
        else:
            note = io.ask(_PROMPT_OPINION).strip()
            notes = [note] if note else []

    human_notes.extend(notes)  # 计入累计意见，供下一轮评审/回传使用

    parts = []
    if problems:
        parts.append(f"{len(problems)} 条评审问题")
    if notes:
        parts.append(f"{len(notes)} 条用户意见")
    io.out(f"[review] 本次『{' + '.join(parts)}』将交回给大脑，由它再次修订。")

    return {"status": "revise", "problems": problems, "notes": notes}


# ---- 工具定义 ------------------------------------------------------------
class ReviewArgs(BaseModel):
    plan: str | None = None  # 待评审计划的 JSON 文本；留空 = 用上一版已生成计划


def make_review_tool() -> Tool:
    def _run(ctx: RuntimeCtx, args: ReviewArgs) -> str:
        if ctx.io is None:
            raise RuntimeError("review 需要 io（人机评审会）。")
        rs = review_state(ctx)

        if args.plan:
            plan = GenerationPlan.model_validate_json(args.plan)
        elif rs.last_plan is not None:
            plan = rs.last_plan
        else:
            return json.dumps(
                {"status": "error", "message": "还没有可评审的计划：请先调用 make_plan。"},
                ensure_ascii=False,
            )

        rs.last_plan = plan
        result = review_meeting(
            ctx.io,
            user_input=ctx.user_input,
            plan=plan,
            critic=ctx.critic,
            human_notes=rs.human_notes,
            version=rs.version or 1,
        )

        if result["status"] == "confirmed":
            rs.confirmed_plan = plan
        elif result["status"] == "revise":
            rs.pending_problems = list(result["problems"])
            rs.pending_notes = list(result["notes"])

        return json.dumps(result, ensure_ascii=False)

    return Tool(
        name="review",
        description=(
            "人机评审会：对当前/指定计划开一场评审会——机器先按你的需求挑问题，"
            "再在终端请你确认（回车=按问题修正 / p=强制通过 / q=放弃 / 直接输入文字=一条意见）。"
            "返回 {status: confirmed|revise|aborted, problems, notes}。"
            "只有 status=confirmed 的计划才是定稿，render/generate 前必须先过这一关。"
        ),
        args_schema=ReviewArgs,
        run=_run,
    )
