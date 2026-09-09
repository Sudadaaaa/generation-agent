"""会话级运行上下文 + 单步 ReAct（agent_act）。

与「每消息一个任务」的旧设计不同：main 直接持有一个 while 主循环，把每次用户输入、
工具调用结果、模型返回都按序追加进**同一份整场 transcript**（history 由 main 持有），
每轮用 agent_act 推进**一次**大脑往返，直到模型给出纯文本收尾 / 上下文超限 /
大脑连续传输失败交回用户。

- 大脑 = ctx.brain（须支持 complete_tools）；单步按原生 function calling 的
  assistant/tool 消息协议把往返追加进 history。
- 工具门控：每个工具可自报 gate（如 generate 需 review=confirmed）；gate 返回字符串
  即拦截成观察，不执行工具。
- 无预算兜底（不加步数/调用次数上限）；异常回喂让大脑换策略，连续失败上限（3 次）
  只是防基础设施故障无限空转，交回给用户决定，不是流程限制。
- 计费记账：每步把各客户端用量增量同步进 ctx.usage（整场累计，跨消息不清零）。

RuntimeCtx 是**整场会话**的运行上下文：clients / io / generator / registry 由主流程
注入一次，state 是「会话内累计状态」的通用挂载点（如 review 门控状态，跨消息持续），
user_input 由主流程在每个用户消息到来时更新为当前需求。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentkit.io import IO
from agentkit.policy import POLICY
from agentkit.tool import Tool, ToolRegistry
from agentkit.usage import Usage
from llm.base import ChatClient, ToolCall

#: 连续传输失败上限：达到即交回用户（防 API 故障时无限空转）。非流程限制。
_MAX_CONSECUTIVE_FAILURES = 3


class ContextLimit(RuntimeError):
    """整场对话超出模型上下文上限，继续只会反复失败 → 终止会话，提示重跑 main。"""


def _is_context_limit(exc: RuntimeError) -> bool:
    """判断一次传输失败是不是「上下文超限」类（这类错误重试也无意义）。"""
    text = str(exc).lower()
    markers = (
        "context",
        "maximum context",
        "too long",
        "reduce the length",
        "token limit",
        "input is too long",
        "max context",
        "exceeded the max",
    )
    return any(m in text for m in markers)


@dataclass
class RuntimeCtx:
    """整场会话的运行上下文。主流程（main）注入一次，跨消息持续（不重置）。"""

    user_input: str = ""
    io: IO | None = None
    registry: ToolRegistry | None = None
    clients: dict[str, ChatClient] = field(default_factory=dict)
    generator: Any | None = None  # 生图后端（agentkit 零图像知识，这里仅透传）
    state: dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        self._seen: dict[str, tuple[int, int, int]] = {}
        self.failures = 0  # 连续传输失败计数（成功即清零）

    # -- 客户端访问 ------------------------------------------------------
    @property
    def brain(self) -> ChatClient:
        return self.clients["brain"]

    @property
    def worker(self) -> ChatClient:
        """plan/渲染工人（deepseek 或 qwen，由 config 决定）。"""
        return self.clients["worker"]

    @property
    def critic(self) -> ChatClient:
        """机器评审工人（固定 deepseek，与规划解耦）。"""
        return self.clients["critic"]

    # -- 计费同步 --------------------------------------------------------
    def sync_usage(self) -> None:
        """把各客户端自上次同步以来的用量增量并入 ctx.usage。"""
        for key, client in self.clients.items():
            st = client.usage
            now = (st.prompt_tokens, st.completion_tokens, st.calls)
            prev = self._seen.get(key)
            if prev is None:
                delta = now
            else:
                delta = (
                    now[0] - prev[0],
                    now[1] - prev[1],
                    now[2] - prev[2],
                )
            self._seen[key] = now
            self.usage.add_tokens(*delta)


def agent_act(ctx: RuntimeCtx, history: list[dict[str, Any]]) -> str | None:
    """推进**一次**大脑往返，返回收尾文本或 None。

    返回 None = 大脑这一轮调了工具，观察已追加进 history，主循环应继续让大脑行动
    （不读用户输入）；返回字符串 = 大脑给出纯文本收尾 / 连续失败交回，主循环展示后
    回到用户输入。上下文超限抛 ContextLimit（整场终止）。
    """
    if ctx.registry is None:
        raise ValueError("RuntimeCtx.registry 未注入（工具集为空无法运行）")
    if "brain" not in ctx.clients:
        raise ValueError("RuntimeCtx.clients 缺少 brain（大脑客户端未注入）")

    ctx.sync_usage()

    messages = (
        [{"role": "system", "content": POLICY}]
        + history
    )

    try:
        text, calls = ctx.brain.complete_tools(
            messages, [t.openai_schema() for t in ctx.registry.tools]
        )
    except RuntimeError as exc:
        ctx.sync_usage()

        if _is_context_limit(exc):
            raise ContextLimit(
                f"[runtime] 整场对话已超出模型上下文上限，继续只会反复失败。\n"
                f"原因：{exc}"
            ) from exc

        ctx.failures += 1
        if ctx.failures >= _MAX_CONSECUTIVE_FAILURES:
            ctx.failures = 0
            return (
                f"[runtime] 大脑连续调用失败（{_MAX_CONSECUTIVE_FAILURES} 次）。"
                f"最后错误：{exc}\n请检查网络/API 配置后重试，或输入 exit 结束会话。"
            )

        # 单次失败：异常回喂，让大脑重试或换策略（不算一条用户需求）
        history.append(
            {
                "role": "user",
                "content": f"[runtime] 大脑调用失败：{exc}。请重试，"
                "或改用现有工具继续推进。",
            }
        )
        return None

    ctx.failures = 0
    ctx.sync_usage()

    # 无论是否调工具，模型这次返回都进整场 transcript（用户要求「模型返回也在整场对话里」）
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": text or ""}
    if calls:
        assistant_msg["tool_calls"] = [_wire_tool_call(c) for c in calls]
    history.append(assistant_msg)

    if not calls:
        final = (text or "").strip() or "（大脑未给出内容，本轮已结束。）"
        return final

    for call in calls:
        observation = _dispatch(ctx, call)
        ctx.sync_usage()
        history.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": observation,
            }
        )

    return None


def _wire_tool_call(call: ToolCall) -> dict[str, Any]:
    """把 ToolCall 编码成 OpenAI 兼容的 assistant.tool_calls 元素。"""
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.arguments, ensure_ascii=False),
        },
    }


def _dispatch(ctx: RuntimeCtx, call: ToolCall) -> str:
    """执行一次工具调用，返回观察文本。门控/校验/执行错误都转成观察。"""
    registry = ctx.registry
    assert registry is not None
    tool = registry.get(call.name)

    if tool is None:
        return (
            f"[runtime] 未知工具 {call.name!r}；当前可用工具："
            f"{registry.names()}。请从清单中选择。"
        )

    try:
        args = tool.args_schema(**call.arguments)  # 校验；额外字段忽略
        return tool.gate_or_run(ctx, args)
    except Exception as exc:
        return (
            f"[runtime] 工具 {call.name} 执行出错（参数 {call.arguments!r}）："
            f"{type(exc).__name__}: {exc}"
        )
