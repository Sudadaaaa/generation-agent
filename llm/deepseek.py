"""DeepSeek 云端（OpenAI 兼容接口）传输实现。

实现 chat（工人用）与 complete_tools（大脑用，原生 function calling）。
所有 OpenAI SDK 异常映射为 RuntimeError（统一上层契约）；
deepseek-reasoner 与 tools 组合有 API 限制 → 大脑默认走 deepseek-chat。
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI, OpenAIError

from llm.base import ChatClient, ToolCall

_DEFAULT_BASE_URL = "https://api.deepseek.com"


def _create_client(api_key: str, base_url: str | None) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=base_url or _DEFAULT_BASE_URL,
    )


class DeepSeekChat(ChatClient):
    """DeepSeek 云端 API：大脑（complete_tools）+ 工人（chat）都用它。"""

    provider = "deepseek"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 50000,
        temperature: float = 0.1,
    ) -> None:
        model = model or os.getenv("LLM_MODEL") or "deepseek-chat"
        super().__init__(model=model)
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client: OpenAI | None = None

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "DEEPSEEK_API_KEY is not configured. "
                    "Please add it to your .env file."
                )
            self._client = _create_client(self.api_key, self.base_url)
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = True,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._ensure_client().chat.completions.create(**kwargs)
        except OpenAIError as exc:
            raise RuntimeError(f"DeepSeek 调用失败：{exc}") from exc

        self._usage.add_raw(getattr(response, "usage", None))

        choice = response.choices[0]
        message = choice.message
        content = message.content

        if not content:
            # 推理类模型：reasoning_content 与 content 都计入 max_tokens，
            # 推理过长或瞬时故障时 content 可能为空。带上现场信息方便排查。
            reasoning = getattr(message, "reasoning_content", None) or ""

            raise RuntimeError(
                "DeepSeek 返回了空内容（message.content 为空）。"
                f" finish_reason={choice.finish_reason!r}, "
                f"usage={response.usage!r}, "
                f"reasoning_content 长度={len(reasoning)}。"
                "可能是 max_tokens 被推理内容占满或瞬时故障。"
            )

        return content

    def complete_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        *,
        temperature: float | None = None,
    ) -> tuple[str | None, list[ToolCall]]:
        """原生 function calling 一步往返：返回 (content, tool_calls)。

        content 为 None 时必有 tool_calls；二者皆无（空响应/只推理）抛 RuntimeError。
        tool_calls 的 arguments 是 JSON 字符串，这里解析成 dict 再返回。
        """
        try:
            response = self._ensure_client().chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )
        except OpenAIError as exc:
            raise RuntimeError(f"DeepSeek 调用失败：{exc}") from exc

        self._usage.add_raw(getattr(response, "usage", None))

        message = response.choices[0].message
        content = message.content or None
        calls: list[ToolCall] = []

        for raw in getattr(message, "tool_calls", None) or []:
            fn = getattr(raw, "function", None)
            if fn is None:
                continue
            name = getattr(fn, "name", None) or "?"
            try:
                arguments: dict[str, Any] = json.loads(
                    getattr(fn, "arguments", None) or "{}"
                )
            except json.JSONDecodeError:
                arguments = {}
            calls.append(ToolCall(id=getattr(raw, "id", None) or "?", name=name, arguments=arguments))

        if not content and not calls:
            raise RuntimeError(
                "DeepSeek function calling 返回了空内容且无 tool_calls"
                f"（model={self.model}）。"
                "注意 deepseek-reasoner 与 tools 组合可能有 API 限制；大脑建议 deepseek-chat。"
            )

        return content, calls
