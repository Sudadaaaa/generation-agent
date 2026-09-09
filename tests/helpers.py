"""离线测试共用构件：假客户端 / 假 IO / 假生图后端 / 样本计划。"""

from __future__ import annotations

from typing import Any, Callable

from llm.base import ChatClient, ToolCall, UsageStat
from domain.schema import Element, GenerationPlan


# ---- 样本计划 ----------------------------------------------------------
def sample_plan(*, overall: str = "温暖日系插画风") -> GenerationPlan:
    return GenerationPlan(
        elements=[
            Element(
                identity="一只橘猫",
                appearance="毛色橘黄，眼睛圆亮",
                layout="画面中央",
                action="蹲坐着看向镜头",
            )
        ],
        overall=overall,
    )


def sample_plan_json(*, overall: str = "温暖日系插画风") -> str:
    return sample_plan(overall=overall).model_dump_json(indent=2)


# ---- 假 IO -------------------------------------------------------------
class FakeIO:
    """脚本化 IO：按序消耗 answers；记录所有输出。"""

    def __init__(self, answers: list[str] | None = None) -> None:
        self._answers = list(answers or [])
        self.outs: list[str] = []

    def ask(self, prompt: str) -> str:
        if not self._answers:
            return ""
        return self._answers.pop(0)

    def out(self, text: str = "") -> None:
        self.outs.append(text)

    def join_outs(self) -> str:
        return "\n".join(self.outs)


# ---- 假生图后端 --------------------------------------------------------
class FakeGenerator:
    """记录调用、假返回路径，供门控断言「到底有没有真出图」。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []  # (prompt, tag)

    def generate(self, prompt: str, tag: str) -> str:
        self.calls.append((prompt, tag))
        return f"/tmp/out/{tag}_{len(self.calls)}.png"


# ---- 假传输客户端 ------------------------------------------------------
class FakeChat(ChatClient):
    """脚本化 chat()：json_mode=True（规划/评审）与 False（渲染）分路取回复。

    json 侧与 free 侧均可给「固定列表」或「按消息列表计算回复的函数」。
    """

    def __init__(
        self,
        *,
        json_replies: list[str] | None = None,
        free_replies: list[str] | None = None,
        json_fn: Callable[[list[dict[str, str]]], str] | None = None,
        free_fn: Callable[[list[dict[str, str]]], str] | None = None,
    ) -> None:
        super().__init__(model="fake")
        self._jr = list(json_replies or [])
        self._fr = list(free_replies or [])
        self._jfn = json_fn
        self._ffn = free_fn
        self.seen: list[list[dict[str, str]]] = []  # 每次调用的完整 messages

    def _next(self, *, json_mode: bool, messages: list[dict[str, str]]) -> str:
        self.seen.append(list(messages))
        if json_mode:
            if self._jr:
                return self._jr.pop(0)
            if self._jfn is not None:
                return self._jfn(messages)
        else:
            if self._fr:
                return self._fr.pop(0)
            if self._ffn is not None:
                return self._ffn(messages)
        raise RuntimeError(f"FakeChat 无更多 {'json' if json_mode else 'free'} 回复")

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = True) -> str:
        self._usage.add_values(prompt_tokens=10, completion_tokens=5)
        return self._next(json_mode=json_mode, messages=messages)


class RaisingChat(ChatClient):
    """chat 永远抛 RuntimeError（测 fail-open / 兜底）。"""

    def __init__(self) -> None:
        super().__init__(model="fake")
        self.calls = 0

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = True) -> str:
        self.calls += 1
        raise RuntimeError("模拟传输失败")


class FakeBrain(ChatClient):
    """脚本化大脑：按序吐 (text, tool_calls)；耗尽后抛错（防测错方向）。"""

    def __init__(
        self,
        script: list[tuple[str | None, list[dict[str, Any]]]],
    ) -> None:
        super().__init__(model="fake-brain")
        self._script = list(script)
        self.seen: list[list[dict[str, str]]] = []
        self._counter = 0

    def _next_call_id(self) -> str:
        self._counter += 1
        return f"call_{self._counter}"

    def complete_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        *,
        temperature: float | None = None,
    ) -> tuple[str | None, list[ToolCall]]:
        self.seen.append(list(messages))
        self._usage.add_values(prompt_tokens=10, completion_tokens=5)
        if not self._script:
            raise RuntimeError("FakeBrain 脚本耗尽")
        text, raw_calls = self._script.pop(0)
        calls = [
            ToolCall(
                id=self._next_call_id(),
                name=rc["name"],
                arguments=dict(rc.get("arguments") or {}),
            )
            for rc in raw_calls
        ]
        return text, calls

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = True) -> str:
        # 大脑只经 complete_tools 驱动；chat 抽象方法补个桩防误用。
        raise RuntimeError("FakeBrain 是大脑假客户端：请走 complete_tools，不要调 chat。")


def brain_tool(name: str, **kwargs: Any) -> dict[str, Any]:
    """构造一条大脑要发的工具调用。"""
    return {"name": name, "arguments": kwargs}
