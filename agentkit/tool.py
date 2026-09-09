"""Tool 契约 + 注册表。

Tool 与 MCP 同构：name / description / args_schema(JSON Schema 来源)——
args_schema.model_json_schema() 即 MCP inputSchema；OpenAI 兼容 function calling
的 tools 描述由 runtime 据此生成（见 runtime.to_openai_tools），不在本类固化。

gate —— 工具自报的「准入检查」：返回 None 放行；返回字符串表示拦截（作为
observation 回喂给大脑）。确认门（render/generate 需 review=confirmed）就实现为
工具的 gate，由工具自己声明领域状态前提，runtime 保持通用。

run —— (ctx, args) -> observation(str)。args 是 args_schema 校验后的实例
（额外字段被忽略；缺必填字段会抛 ValidationError，由 runtime 兜成错误观察）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel

if TYPE_CHECKING:  # 仅类型标注，避免与 runtime 循环导入
    from agentkit.runtime import RuntimeCtx

ArgsT = type[BaseModel]
GateFn = Callable[["RuntimeCtx", BaseModel], str | None]
RunFn = Callable[["RuntimeCtx", BaseModel], str]


@dataclass
class Tool:
    name: str
    description: str
    args_schema: ArgsT
    run: RunFn
    gate: GateFn | None = None

    def gate_or_run(self, ctx: "RuntimeCtx", args: BaseModel) -> str:
        """先过门（若声明），再过执行；返回观察文本。"""
        if self.gate is not None:
            blocked = self.gate(ctx, args)
            if blocked is not None:
                return blocked
        return self.run(ctx, args)

    def openai_schema(self) -> dict[str, Any]:
        """生成 OpenAI 兼容 function calling 的工具描述（原生 function calling 用）。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }


class ToolRegistry:
    """名字 → Tool 的注册表。"""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重名：{tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def tools(self) -> list[Tool]:
        """按注册顺序返回（确定性：schema 顺序与名字清单稳定）。"""
        return list(self._tools.values())

    def names(self) -> list[str]:
        return [t.name for t in self.tools]
