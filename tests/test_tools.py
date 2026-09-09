"""tools 层离线回归：make_plan（生成/修订/自动带问题意见）、review 人机评审会
（回车/p/q/意见/累计，即 S1–S6 语义）、render / generate 的确认门。"""

import json

from agentkit.runtime import RuntimeCtx
from tools.image import make_generate_tool
from tools.plan import make_plan_tool
from tools.render import make_render_tool
from tools.review import make_review_tool, review_state

from tests.helpers import FakeChat, FakeGenerator, FakeIO, sample_plan, sample_plan_json

_OK = json.dumps({"ok": True, "problems": []})
_BAD = json.dumps({"ok": False, "problems": ["把猫移到草地上"]})


def _ctx(*, io=None, worker=None, critic=None, generator=None, user_input="一只橘猫蹲坐着看镜头"):
    from agentkit.tool import ToolRegistry

    return RuntimeCtx(
        user_input=user_input,
        io=io or FakeIO([]),
        registry=ToolRegistry(),
        clients={
            "brain": FakeChat(),
            "worker": worker or FakeChat(json_fn=lambda m: sample_plan_json()),
            "critic": critic or FakeChat(json_replies=[_OK]),
        },
        generator=generator,
    )


def _run_tool(tool, ctx, args):
    parsed = tool.args_schema(**args)
    return tool.gate_or_run(ctx, parsed)


# ---- make_plan -----------------------------------------------------------
def test_make_plan_fresh_stores_last_plan():
    ctx = _ctx()
    obs = _run_tool(make_plan_tool(), ctx, {})
    plan = json.loads(obs)
    assert plan["elements"][0]["identity"] == "一只橘猫"
    rs = review_state(ctx)
    assert rs.last_plan is not None and rs.version == 1


def test_make_plan_revision_auto_carries_pending():
    """review=revise 后直接再调 make_plan（不带参）→ 自动带上问题/意见修订上一版。"""
    worker = FakeChat(json_fn=lambda m: sample_plan_json())
    ctx = _ctx(worker=worker, critic=FakeChat(json_replies=[_BAD, _OK]))

    _run_tool(make_plan_tool(), ctx, {})
    review_tool = make_review_tool()
    # ① 回车（按机器问题修正）→ ② 无补充意见
    ctx.io = FakeIO([""] * 2)
    obs = json.loads(_run_tool(review_tool, ctx, {}))
    assert obs["status"] == "revise"

    rs = review_state(ctx)
    assert rs.pending_problems == ["把猫移到草地上"]

    _run_tool(make_plan_tool(), ctx, {})  # 不带参 → 应自动带 pending 修订
    assert rs.version == 2
    last_user = worker.seen[-1][-1]["content"]
    assert "把猫移到草地上" in last_user  # 修订消息含机器问题


def test_make_plan_continue_prev_uses_last_plan():
    """base=prev 续版：无 problems/notes 也基于上一版续写（连续会话「加 XX」核心）。"""
    worker = FakeChat(json_fn=lambda m: sample_plan_json(overall="戴上墨镜"))
    ctx = _ctx(worker=worker, user_input="在刚才基础上给猫加一副墨镜")
    _run_tool(make_plan_tool(), ctx, {})  # v1 全新
    obs = _run_tool(make_plan_tool(), ctx, {"base": "prev"})  # 续版 → v2
    assert json.loads(obs)["elements"][0]["identity"] == "一只橘猫"
    rs = review_state(ctx)
    assert rs.version == 2
    # 续版时 worker 收到：system + 当前需求 + assistant(上一版 JSON) + build_continue_message
    msgs = worker.seen[-1]
    assert msgs[-2]["role"] == "assistant" and "一只橘猫" in msgs[-2]["content"]
    assert "续写" in msgs[-1]["content"] and "墨镜" in msgs[-1]["content"]


def test_make_plan_continue_without_prior_returns_error():
    ctx = _ctx()
    obs = _run_tool(make_plan_tool(), ctx, {"base": "prev"})
    assert "没有上一版计划可续" in obs
    assert review_state(ctx).last_plan is None


def test_make_plan_new_thread_clears_previous_state():
    """新主题开新线程：旧主题的定稿/意见/版本被清空，不串场。"""
    ctx = _ctx()
    rs = review_state(ctx)
    rs.confirmed_plan = sample_plan()
    rs.human_notes = ["旧主题的意见"]
    rs.version = 7
    _run_tool(make_plan_tool(), ctx, {})  # auto 且无 pending → 开新线程
    rs2 = review_state(ctx)
    assert rs2.version == 1
    assert rs2.human_notes == [] and rs2.confirmed_plan is None


# ---- review 人机评审会 ---------------------------------------------------
def test_review_machine_ok_enter_confirms():
    ctx = _ctx(critic=FakeChat(json_replies=[_OK]), io=FakeIO([""]))
    _run_tool(make_plan_tool(), ctx, {})
    obs = json.loads(_run_tool(make_review_tool(), ctx, {}))
    assert obs["status"] == "confirmed"
    assert review_state(ctx).confirmed_plan is not None


def test_review_machine_ok_typing_opinion_revises():
    ctx = _ctx(critic=FakeChat(json_replies=[_OK]), io=FakeIO(["给猫加一顶牛仔帽"]))
    _run_tool(make_plan_tool(), ctx, {})
    obs = json.loads(_run_tool(make_review_tool(), ctx, {}))
    assert obs["status"] == "revise"
    assert obs["notes"] == ["给猫加一顶牛仔帽"]
    assert review_state(ctx).human_notes == ["给猫加一顶牛仔帽"]


def test_review_problems_p_force_confirms():
    ctx = _ctx(critic=FakeChat(json_replies=[_BAD]), io=FakeIO(["p"]))
    _run_tool(make_plan_tool(), ctx, {})
    obs = json.loads(_run_tool(make_review_tool(), ctx, {}))
    assert obs["status"] == "confirmed"  # 强制通过


def test_review_q_aborts_no_confirm():
    ctx = _ctx(critic=FakeChat(json_replies=[_BAD]), io=FakeIO(["q"]))
    _run_tool(make_plan_tool(), ctx, {})
    obs = json.loads(_run_tool(make_review_tool(), ctx, {}))
    assert obs["status"] == "aborted"
    assert review_state(ctx).confirmed_plan is None


def test_review_typed_text_at_step1_skips_step2():
    ctx = _ctx(critic=FakeChat(json_replies=[_BAD]), io=FakeIO(["①处直接写意见"]))
    _run_tool(make_plan_tool(), ctx, {})
    obs = json.loads(_run_tool(make_review_tool(), ctx, {}))
    assert obs["status"] == "revise"
    assert obs["notes"] == ["①处直接写意见"]
    assert obs["problems"] == ["把猫移到草地上"]


def test_review_accumulates_human_notes_across_rounds():
    """历轮意见累计进 human_notes（供评审 user_additions 用，避免误判）。"""
    io = FakeIO(["想让它戴帽子", ""])  # 第一轮 ① 打字=一条意见；第二轮 ② 无意见
    critic = FakeChat(json_replies=[_BAD, _BAD])
    ctx = _ctx(critic=critic, io=io)
    _run_tool(make_plan_tool(), ctx, {})

    obs1 = json.loads(_run_tool(make_review_tool(), ctx, {}))
    assert obs1["status"] == "revise"

    _run_tool(make_plan_tool(), ctx, {})  # 修订
    obs2 = json.loads(_run_tool(make_review_tool(), ctx, {}))
    assert obs2["status"] == "revise"

    assert review_state(ctx).human_notes == ["想让它戴帽子"]  # 两轮未新增意见则只累计一条


def test_review_no_plan_yet_returns_error():
    ctx = _ctx()
    obs = json.loads(_run_tool(make_review_tool(), ctx, {}))
    assert obs["status"] == "error"


# ---- render / generate 确认门 -------------------------------------------
def _confirmed_ctx(worker=None, generator=None):
    ctx = _ctx(worker=worker, generator=generator)
    _run_tool(make_plan_tool(), ctx, {})
    rs = review_state(ctx)
    rs.confirmed_plan = rs.last_plan  # 模拟已 review=confirmed
    return ctx


def test_render_blocked_before_confirm():
    ctx = _ctx()
    _run_tool(make_plan_tool(), ctx, {})
    obs = _run_tool(make_render_tool(), ctx, {})
    assert "门控拦截" in obs and "review" in obs


def test_render_confirmed_renders_and_records_prompt():
    worker = FakeChat(
        json_fn=lambda m: sample_plan_json(),
        free_replies=["一幅温暖插画：一只橘猫蹲坐着。"],
    )
    ctx = _confirmed_ctx(worker=worker)
    obs = _run_tool(make_render_tool(), ctx, {})
    assert obs.startswith("一幅温暖插画")
    assert review_state(ctx).rendered_prompt == obs


def test_render_rejects_foreign_plan():
    worker = FakeChat(
        json_fn=lambda m: sample_plan_json(),
        free_replies=["渲染成功"],
    )
    ctx = _confirmed_ctx(worker=worker)
    obs = _run_tool(make_render_tool(), ctx, {"plan": sample_plan_json(overall="另一张图")})
    assert "不一致" in obs and review_state(ctx).rendered_prompt is None


def test_generate_gates():
    gen = FakeGenerator()
    worker = FakeChat(
        json_fn=lambda m: sample_plan_json(),
        free_replies=["提示词P"],
    )
    ctx = _confirmed_ctx(generator=gen, worker=worker)
    # 未 render → 拦
    obs = _run_tool(make_generate_tool(), ctx, {})
    assert "门控拦截" in obs and gen.calls == []

    # render 后成功出图
    _run_tool(make_render_tool(), ctx, {})
    obs = _run_tool(make_generate_tool(), ctx, {})
    assert gen.calls == [("提示词P", "plan")] and "图片已生成" in obs


def test_generate_rejects_foreign_prompt_even_after_render():
    gen = FakeGenerator()
    worker = FakeChat(
        json_fn=lambda m: sample_plan_json(),
        free_replies=["提示词P"],
    )
    ctx = _confirmed_ctx(generator=gen, worker=worker)
    _run_tool(make_render_tool(), ctx, {})
    obs = _run_tool(make_generate_tool(), ctx, {"prompt": "不是我 render 的提示词"})
    assert "门控拦截" in obs and gen.calls == []
