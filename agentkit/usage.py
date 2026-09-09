"""用量记账。

Usage 聚合一次会话里各传输客户端（llm/base 的 UsageStat）的增量。
会话结束条件由用户决定（exit / 上下文超限 / 大脑连续传输失败交回），
不设步数 / API 调用次数上限，故无 Budget。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    """会话级累计用量（由 runtime 从各客户端逐次同步增量而来）。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add_tokens(self, prompt: int, completion: int, calls: int) -> None:
        self.prompt_tokens += int(prompt)
        self.completion_tokens += int(completion)
        self.calls += int(calls)

    def merge(self, other: "Usage") -> None:
        self.add_tokens(other.prompt_tokens, other.completion_tokens, other.calls)

    def describe(self) -> str:
        return (
            f"{self.calls} 次调用 / {self.total_tokens} tokens"
            f"（入 {self.prompt_tokens} + 出 {self.completion_tokens}）"
        )
