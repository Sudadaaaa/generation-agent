"""核心闭环离线回归：用假大脑整场驱动 main 的单 while 主循环 + 真实工具集，
断言门控 / 连续会话 / 修订环 / 语义分层的行为，全程不触网不占显存。

与旧版（run() 单任务）不同：new_session 持有一份整场 transcript，
用户输入、工具观察、模型返回跨消息持续——⑥ 即复现「先出图 → 加 XX 再生成」缺陷。"""

import json

from agentkit.tool import ToolRegistry
from config import AgentConfig
from main import new_session
from tools import build_tools

from tests.helpers import FakeBrain, FakeChat, FakeGenerator, FakeIO, brain_tool, sample_plan_json

_OK = json.dumps({"ok": True, "problems": []})
_BAD = json.dumps({"ok": False, "problems": ["把猫移到草地上"]})


class _RaisingBrain(FakeBrain):
    """complete_tools 每次都抛给定 RuntimeError（测传输失败 / 上下文超限的兜底）。"""

    def __init__(self, message: str) -> None:
        super().__init__(script=[])
        self._message = message

    def complete_tools(self, messages, tools, *, temperature=None):
        self.seen.append(list(messages))
        self._usage.add_values(prompt_tokens=10, completion_tokens=5)
        raise RuntimeError(self._message)


def _services(*, brain, image_model="zimage", worker=None, critic=None):
    """按 cfg 装配一套完整服务：真实工具集 + 注入的假大脑/工人/评审/生图后端。"""
    cfg = AgentConfig.from_env({"AGENT_IMAGE_MODEL": image_model})
    registry = ToolRegistry(build_tools(cfg))
    clients = {
        "brain": brain,
        "worker": worker or FakeChat(json_fn=lambda m: sample_plan_json()),
        "critic": critic or FakeChat(json_replies=[_OK]),
    }
    generator = FakeGenerator() if image_model else None
    return cfg, clients, generator, registry


def _run_session(cfg, clients, generator, registry, answers):
    """用 FakeIO 把 main 的整个 while 主循环驱动到退出。"""
    io = FakeIO(answers)
    new_session(cfg, io=io, services=(clients, generator, registry))
    return io


def _tool_obs(brain, marker: str) -> bool:
    """任一大脑调用所见历史里出现带 marker 的工具观察。"""
    for step in brain.seen:
        for msg in step:
            if msg.get("role") == "tool" and marker in msg["content"]:
                return True
    return False


# ---- ① 无 confirmed 定稿时 generate 被拦（真后端不被调用）----------------
def test_generate_blocked_before_review_confirm_session_continues():
    brain = FakeBrain(
        script=[
            (None, [brain_tool("generate")]),
            ("没有 review=confirmed 的定稿，先不硬出图。", []),
            ("好，你随时可以重来。", []),   # 第二句：大脑直接收尾，会话仍在继续
        ]
    )
    cfg, clients, generator, registry = _services(brain=brain)
    io = _run_session(
        cfg, clients, generator, registry,
        answers=["出一张橘猫图", "换个话题", "exit"],
    )
    assert generator.calls == []                     # 生图后端从未被触碰
    assert _tool_obs(brain, "门控拦截")               # 大脑第二轮看到了门控观察
    outs = io.join_outs()
    assert outs.count("[结果]") == 2 and "Bye！" in outs


# ---- ② review=aborted → 无出图 -------------------------------------------
def test_review_aborted_no_generate():
    brain = FakeBrain(
        script=[
            (None, [brain_tool("make_plan")]),
            (None, [brain_tool("review")]),
            ("用户放弃了，收尾。", []),
            (None, [brain_tool("generate")]),   # 第二句想绕过确认直接出图
            ("没确认过，不能出图。", []),
        ]
    )
    cfg, clients, generator, registry = _services(brain=brain)
    io = _run_session(
        cfg, clients, generator, registry,
        answers=["出一张橘猫图", "q", "再画一张宇航员", "exit"],  # review 机器通过 → q=放弃
    )
    assert generator.calls == []
    assert _tool_obs(brain, "门控拦截")
    assert io.join_outs().count("[结果]") == 2


# ---- ③ 大脑连续传输失败 → 交回用户，不无限重试 ---------------------------
def test_brain_continuous_failures_hand_back_to_user():
    brain = _RaisingBrain("DeepSeek 调用失败：连接超时")
    cfg, clients, generator, registry = _services(brain=brain)
    io = _run_session(cfg, clients, generator, registry, answers=["随便画只猫", "exit"])
    assert len(brain.seen) == 3                     # 试 3 次即交回，没无限空转
    outs = io.join_outs()
    assert "大脑连续调用失败" in outs and "Bye！" in outs


# ---- ④ 修订环 → confirmed → render → generate 只出一次图 ------------------
def test_revise_loop_until_confirmed_then_generate_once():
    brain = FakeBrain(
        script=[
            (None, [brain_tool("make_plan")]),      # v1
            (None, [brain_tool("review")]),         # 机器发现问题
            (None, [brain_tool("make_plan")]),      # 自动带问题修订 → v2
            (None, [brain_tool("review")]),         # 机器通过
            (None, [brain_tool("render")]),
            (None, [brain_tool("generate")]),
            ("完成。", []),
        ]
    )
    worker = FakeChat(
        json_fn=lambda m: sample_plan_json(overall="把猫放到草地上"),
        free_replies=["一幅插画：草地上的橘猫。"],
    )
    cfg, clients, generator, registry = _services(
        brain=brain, worker=worker,
        critic=FakeChat(json_replies=[_BAD, _OK]),
    )
    io = _run_session(
        cfg, clients, generator, registry,
        answers=["出一张草地橘猫图", "", "", "", "exit"],  # ①回车 ②无意见 确认回车
    )
    # 只出图一次，prompt 严格等于 render 产出的最终提示词
    assert generator.calls == [("一幅插画：草地上的橘猫。", "plan")]
    assert io.join_outs().count("[结果]") == 1


# ---- ⑤ 只要提示词：未配生图后端 → 注册表无 generate（结构性拦越权）------
def test_no_image_model_no_generate_registered():
    brain = FakeBrain(
        script=[
            (None, [brain_tool("make_plan")]),
            (None, [brain_tool("review")]),
            (None, [brain_tool("render")]),
            (None, [brain_tool("generate")]),   # 工具不存在
            ("好了，只有提示词。", []),
        ]
    )
    worker = FakeChat(
        json_fn=lambda m: sample_plan_json(),
        free_replies=["提示词：一只橘猫。"],
    )
    cfg, clients, generator, registry = _services(
        brain=brain, worker=worker, image_model="",   # 环境不配生图 → 无 generate
    )
    io = _run_session(
        cfg, clients, generator, registry,
        answers=["只要提示词，先不出图", "", "exit"],
    )
    assert registry.names() == ["make_plan", "review", "render"]
    assert generator is None
    assert _tool_obs(brain, "未知工具")
    assert io.join_outs().count("[结果]") == 1


# ---- ⑥ 跨消息续作：上一版定稿在「加 XX 再生成」里被续写（复现缺陷）---------
def test_cross_message_continue_uses_previous_plan():
    def plan_fn(messages):
        # 含「墨镜」的续写请求 → 新版计划；否则默认（第一版）
        if any(m.get("role") == "user" and "墨镜" in m.get("content", "") for m in messages):
            return sample_plan_json(overall="戴上墨镜，蹲坐看着镜头")
        return sample_plan_json()

    brain = FakeBrain(
        script=[
            (None, [brain_tool("make_plan")]),
            (None, [brain_tool("review")]),
            (None, [brain_tool("render")]),
            (None, [brain_tool("generate")]),
            ("第一张完成。", []),
            # —— 第二句：在刚才基础上加 XX ——
            (None, [brain_tool("make_plan", base="prev")]),
            (None, [brain_tool("review")]),
            (None, [brain_tool("render")]),
            (None, [brain_tool("generate")]),
            ("第二张完成。", []),
        ]
    )
    worker = FakeChat(
        json_fn=plan_fn,
        free_replies=["提示词P1", "提示词P2"],
    )
    cfg, clients, generator, registry = _services(
        brain=brain, worker=worker,
        critic=FakeChat(json_replies=[_OK, _OK]),
    )
    io = _run_session(
        cfg, clients, generator, registry,
        answers=[
            "画一只橘猫蹲坐", "",
            "在刚才基础上给猫加一副墨镜再生成一张", "",
            "exit",
        ],
    )
    # 出了两张图，第二次 prompt ≠ 第一次（用新版续出的计划渲染）
    assert len(generator.calls) == 2
    assert generator.calls[0][0] == "提示词P1"
    assert generator.calls[1][0] == "提示词P2"
    assert generator.calls[0][0] != generator.calls[1][0]
    # 续版那次 make_plan 真的带上了上一版计划（assistant 消息）与续写要求
    cont = [
        ms for ms in worker.seen
        if ms and "续写" in (ms[-1].get("content") or "")
    ]
    assert cont, "第二次 make_plan 应带 build_continue_message"
    assert cont[0][-2]["role"] == "assistant"
    assert "温暖日系插画风" in cont[0][-2]["content"]  # 上一版计划 JSON（默认 overall）
    assert "墨镜" in cont[0][-1]["content"]
    assert io.join_outs().count("[结果]") == 2


# ---- ⑦ 上下文超限 → 提示重跑 main，优雅终止 ------------------------------
def test_context_limit_ends_session_gracefully():
    brain = _RaisingBrain(
        "This model's maximum context length is 8192 tokens. "
        "Please reduce the length of the messages."
    )
    cfg, clients, generator, registry = _services(brain=brain)
    io = _run_session(cfg, clients, generator, registry, answers=["画一只猫", "exit"])
    assert len(brain.seen) == 1                      # 试一次即识别为上下文超限
    outs = io.join_outs()
    assert "已达上下文上限" in outs and "Bye！" in outs
