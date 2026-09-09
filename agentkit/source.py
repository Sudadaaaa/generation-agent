"""ToolSource 协议：工具来源的发现接口。

内置工具（tools/__init__.build_tools）是第一个 ToolSource 实现；
将来接 MCP 服务器时新增一个 ToolSource（server → 逐个包装成 Tool）即可，
主循环不改。runtime 只消费 discover() 得到的 list[Tool]。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from agentkit.tool import Tool


class ToolSource(Protocol):
    """一个能按配置/上下文产出工具列表的来源。"""

    def discover(self, config: Any, ctx: Any) -> list["Tool"]:
        """发现并构造工具。config 为环境配置，ctx 为运行上下文（均可为空/仅作依据）。"""
        ...
