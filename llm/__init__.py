"""llm 传输适配层：只做「发消息、取回复」，不做角色。

导出：
  ChatClient           —— 传输协议（chat + usage + 可选 complete_tools）
  ToolCall             —— function calling 一次调用的解析结果
  UsageStat            —— 单客户端累计用量
  DeepSeekChat         —— 云端实现（chat + 原生 function calling）
  LocalQwenChat        —— 本地 Qwen3-8B（仅 chat，plan 工人用）
  make_chat_client     —— 按 provider 构造
"""

from llm.base import ChatClient, ToolCall, UsageStat
from llm.deepseek import DeepSeekChat
from llm.factory import make_chat_client
from llm.local_qwen import LocalQwenChat

__all__ = [
    "ChatClient",
    "ToolCall",
    "UsageStat",
    "DeepSeekChat",
    "LocalQwenChat",
    "make_chat_client",
]
