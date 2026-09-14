"""工具注册表：存工具、按名派发、导出 OpenAI schema。

机制，不是策略。它不知道这次部署有哪些工具、plan 用哪个 LLM、有没有生图——
那些是装配根（main.py / tools.build_tools）的决定。所以这里不 import agent，
也不 import tools。
"""

import logging

from .tool import Tool
from typing import Optional
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    工具注册表
    """

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool, overwrite: bool = False):
        """注册Tool"""
        if tool.name in self.tools and not overwrite:
            raise ValueError(
                f"工具名 {tool.name!r} 已注册：{self.tools[tool.name]!r}。"
                "如确定要替换，请传 overwrite=True。"
            )
        self.tools[tool.name] = tool

    def unregister(self, name: str):
        """注销Tool"""
        if name in self.tools:
            del self.tools[name]
            return True
        return False

    def get_tool(self, name: str) -> Optional[Tool]:
        """获取Tool对象"""
        return self.tools.get(name)

    def execute_tool(self, name: str, arguments: str | dict | None) -> str:
        """执行工具"""
        tool = self.get_tool(name)
        if tool is None:
            return f"错误：没有名为{name!r}的工具，当前工具列表{self.list_tools()}"
        try:
            args = tool.parse_args(arguments)
            return str(tool.run(args))
        except ValidationError as e:
            brief = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
            return f"错误：工具 {name!r} 参数不合法 —— {brief}"
        except Exception as e:
            logger.exception("工具 %s 执行失败", name)     # ← 完整堆栈进日志
            return f"错误：{e}"

    def get_openai_schemas(self) -> list[dict]:
        """
        获取所有可用工具的openai_schema描述
        """
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(tool.to_openai_schema())

        return descriptions

    def list_tools(self) -> list[str]:
        """查看所有工具名称"""
        return list(self.tools.keys())

    def clear(self):
        """清空所有工具"""
        self.tools.clear()

    def __len__(self):
        return len(self.tools)
