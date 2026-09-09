"""llm 传输适配协议：只管「把消息发给模型、取回文本」，不做角色（大脑/工人/评审）。

角色（brain / plan worker / critic）由调用方用提示词组合而成，不落在这里。
任一子类把「调用失败 / 返回空内容」统一映射为 RuntimeError，供上层（domain 工序）
按单异常类型重试或兜底，从而与具体 SDK（openai / transformers）解耦。

计费记账：每个子类在每次真实 API 调用后把 usage 记入 self._usage（UsageStat），
会话/任务级汇总由 agentkit/usage 或主流程读取 usage 属性完成。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """一次 function calling 调用：已解析的参数 dict（OpenAI 兼容 tools 协议的产物）。"""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class UsageStat:
    """单个传输客户端的累计用量（按次叠加）。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def describe(self) -> str:
        """一行展示（会话退出汇总用），文案与 agentkit/usage.Usage 一致。"""
        return (
            f"{self.calls} 次调用 / {self.total_tokens} tokens"
            f"（入 {self.prompt_tokens} + 出 {self.completion_tokens}）"
        )

    def add_raw(self, raw: Any | None) -> None:
        """把一次响应的 usage 记进来（openai 的 usage 对象或 dict 均可，缺字段容错）。"""
        self.calls += 1
        if raw is None:
            return

        def _get(obj: Any, name: str) -> int:
            if isinstance(obj, dict):
                return int(obj.get(name) or 0)
            return int(getattr(obj, name, None) or 0)

        self.prompt_tokens += _get(raw, "prompt_tokens")
        self.completion_tokens += _get(raw, "completion_tokens")

    def add_values(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.calls += 1
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)


class ChatClient(ABC):
    """传输适配协议。

    - chat(messages, *, json_mode) —— 返回文本回复；供所有「工人」（规划/评审/渲染）。
    - complete_tools(messages, tools) -> (text, tool_calls) —— 原生 function calling；
      仅具备该能力的客户端实现（DeepSeekChat）。本地 qwen 不实现（它不当大脑）。
    - usage —— 累计用量（会话级）。
    失败契约：调用失败或返回内容不可用 → 抛 RuntimeError（不要抛 SDK 专有异常）。
    """

    #: 展示用：提供方 + 模型，便于 usage 打印与排错
    provider: str = "?"

    def __init__(self, *, model: str) -> None:
        self.model = model
        self._usage = UsageStat()

    @property
    def usage(self) -> UsageStat:
        return self._usage

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = True,
    ) -> str:
        """调用模型，返回其文本回复。json_mode=True 请求结构化（JSON）输出。

        空内容 / 网络等传输级失败必须抛 RuntimeError。
        """

    def complete_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        *,
        temperature: float | None = None,
    ) -> tuple[str | None, list[ToolCall]]:
        """原生 function calling 一步往返。

        返回 (content, tool_calls)；content 与 tool_calls 至少一个有值。
        不支持 function calling 的客户端默认抛 NotImplementedError
        （实现方如 DeepSeekChat 覆写此方法）。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 不支持原生 function calling（{self.model}）。"
        )

    def __repr__(self) -> str:  # pragma: no cover - 排错辅助
        return f"<{type(self).__name__} model={self.model!r} usage={self._usage}>"
