"""domain 工序离线回归：schema 兜底渲染、提取格式自修、评审 fail-open 与
user_additions 不误判、渲染兜底、revise 消息、评审问答提示语冻结。"""

import json

from domain.critique import CritiqueVerdict, critique_plan
from domain.extraction import PlanFormatError, create_generation_plan, parse_generation_plan
from domain.prompts import build_continue_message, build_revise_message
from domain.rendering import build_final_prompt
from domain.schema import GenerationPlan
from tools.review import (
    _PROMPT_CONFIRM,
    _PROMPT_OPINION,
    _PROMPT_REVISE,
)

from tests.helpers import FakeChat, RaisingChat, sample_plan, sample_plan_json

_BAD_JSON = "这不是 JSON"
_BAD_SHAPE = json.dumps([1, 2, 3])
_BAD_SCHEMA = json.dumps({"elements": [], "overall": "x"})


# ---- schema / 兜底 ------------------------------------------------------
def test_to_prompt_text_is_deterministic():
    plan = sample_plan()
    text = plan.to_prompt_text()
    assert "画面中央" in text and "一只橘猫" in text and "蹲坐着看向镜头" in text
    assert "温暖日系插画风" in text
    # 不含字段名/结构标记
    assert "elements" not in text and "identity" not in text


def test_parse_valid_plan():
    plan = parse_generation_plan(sample_plan_json())
    assert isinstance(plan, GenerationPlan) and plan.elements[0].identity == "一只橘猫"


def test_parse_rejects_non_json():
    for bad in (_BAD_JSON, _BAD_SHAPE):
        try:
            parse_generation_plan(bad)
            raise AssertionError(f"应抛 PlanFormatError：{bad[:20]!r}")
        except PlanFormatError:
            pass


def test_parse_rejects_bad_schema():
    try:
        parse_generation_plan(_BAD_SCHEMA)
        raise AssertionError("应抛 PlanFormatError（elements 为空）")
    except PlanFormatError:
        pass


# ---- 提取：格式自修 ------------------------------------------------------
def test_extraction_self_corrects_until_valid():
    worker = FakeChat(json_replies=[_BAD_JSON, sample_plan_json()])
    plan = create_generation_plan("一只橘猫蹲坐着", llm=worker)
    assert plan.elements[0].identity == "一只橘猫"
    assert len(worker.seen) == 2  # 第一发不合法 → 带修正消息重试


def test_extraction_gives_up_after_retries():
    worker = FakeChat(json_replies=[_BAD_JSON, _BAD_SCHEMA, _BAD_SHAPE])
    try:
        create_generation_plan("一只橘猫", llm=worker)
        raise AssertionError("应抛 RuntimeError（重试耗尽）")
    except RuntimeError:
        pass


def test_extraction_raises_on_transport_failure():
    try:
        create_generation_plan("一只橘猫", llm=RaisingChat())
        raise AssertionError("应抛 RuntimeError（传输失败）")
    except RuntimeError:
        pass


# ---- 评审：fail-open 与 user_additions ----------------------------------
def test_critique_fail_open_on_transport_error():
    verdict = critique_plan("一只橘猫", sample_plan(), RaisingChat())
    assert isinstance(verdict, CritiqueVerdict)
    assert verdict.ok is True  # 故障按通过（由 review 的人把关兜底）


def test_critique_parses_ok_and_problems():
    good = FakeChat(json_replies=[json.dumps({"ok": True, "problems": []})])
    assert critique_plan("x", sample_plan(), good).ok is True

    bad = FakeChat(
        json_replies=[json.dumps({"ok": False, "problems": ["整体只有一个元素？"]})]
    )
    verdict = critique_plan("x", sample_plan(), bad)
    assert verdict.ok is False and verdict.problems == ["整体只有一个元素？"]


def test_critique_carries_user_additions_into_prompt():
    """用户的补充意见必须作为需求的一部分喂给评审，不判成编造（S1–S6 语义）。"""
    seen: list[dict] = []

    def judge(messages):
        seen.append(messages)
        return json.dumps({"ok": True, "problems": []})

    critic = FakeChat(json_fn=judge)
    critique_plan(
        "一只橘猫",
        sample_plan(overall="额外加了牛仔帽"),
        critic,
        user_additions=["给猫加一顶牛仔帽（用户要求）"],
    )
    user_content = seen[0][-1]["content"]
    assert "给猫加一顶牛仔帽（用户要求）" in user_content
    assert "不要把它们当成凭空编造或与需求冲突去挑错" in user_content


# ---- 渲染 ---------------------------------------------------------------
def test_render_falls_back_on_failure():
    plan = sample_plan()
    assert build_final_prompt(plan, RaisingChat(), user_input="一只橘猫") == plan.to_prompt_text()


def test_render_uses_free_text_output():
    worker = FakeChat(free_replies=["一幅温暖日系插画：一只橘猫蹲坐在画面中央。"])
    prompt = build_final_prompt(sample_plan(), worker, user_input="一只橘猫")
    assert prompt.startswith("一幅温暖日系插画")


# ---- revise 消息 --------------------------------------------------------
def test_revise_message_without_user_notes():
    msg = build_revise_message(["问题A"], None)
    assert "1 个问题" in msg and "问题A" in msg
    assert "用户本人" not in msg


def test_revise_message_prefers_user_notes():
    msg = build_revise_message(["问题A"], ["给猫加帽子"])
    assert msg.index("用户本人提出了以下修改要求") < msg.index("问题A")
    assert "给猫加帽子" in msg


# ---- 续写消息 ----------------------------------------------------------
def test_build_continue_message_carries_requirement_and_rules():
    msg = build_continue_message("在刚才基础上给猫加一副墨镜")
    assert "给猫加一副墨镜" in msg            # 新要求带进消息
    assert "续写" in msg and "identity 必须逐字沿用" in msg  # 续写/保留规则
    assert "只输出 JSON object" in msg        # 只输出 JSON，不要解释


# ---- 冻结：评审问答提示语与 agent_plan_review 逐字一致 -------------------
def test_review_prompts_are_frozen():
    # 三个问答行来自 agent_plan_review（R0 前原文），改动即破坏逐字冻结断言
    assert _PROMPT_REVISE == "① 回车 = 按评审问题修正一轮 | p = 强制通过 | q = 放弃本次\n> "
    assert _PROMPT_OPINION == "② 你的修改意见？（回车=无；会与评审问题一起回传）\n> "
    assert _PROMPT_CONFIRM == (
        "  回车 = 确认通过；或直接输入你的修改意见（按意见再修正一轮）；q = 放弃本次\n> "
    )
