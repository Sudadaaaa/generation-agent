"""按 provider 构造传输客户端（llm 包唯一出口）。

大脑本轮固定 deepseek（需 complete_tools / 原生 function calling）；
plan 工人 deepseek | qwen（仅 chat，qwen 训练完成后一行切换）。

构造只创建传输对象，不加载模型/不占显存——本地 qwen 首次 chat() 才占显存。
"""

from __future__ import annotations

from llm.base import ChatClient
from llm.deepseek import DeepSeekChat

_PROVIDERS = ("deepseek", "qwen")


def make_chat_client(
    provider: str = "deepseek",
    *,
    model: str | None = None,
    lora_path: str | None = None,
    device: str | None = None,
) -> ChatClient:
    """创建传输客户端。provider：deepseek（默认）| qwen。

    deepseek 默认模型由 DeepSeekChat 自身解析（LLM_MODEL → deepseek-chat）；
    qwen 为本地 Qwen3-8B（+ 可选 LoRA），仅实现 chat，不做大脑。
    """
    provider = (provider or "deepseek").strip().lower()

    if provider == "deepseek":
        return DeepSeekChat(model=model)

    if provider == "qwen":
        from llm.local_qwen import LocalQwenChat  # 惰性：避免测试/纯 deepseek 场景拉重型依赖

        return LocalQwenChat(model=model, lora_path=lora_path, device=device)

    raise ValueError(
        f"未知 LLM 提供方：{provider!r}（可选 {' / '.join(_PROVIDERS)}）"
    )
