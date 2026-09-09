"""本 agent 内置工具来源：build_tools(config) 决定注册集。

环境能力分层：生图工具 generate 只在 config.image_model 非空（配了生图后端）
时注册——「能不能出图」由环境配置决定，不由用户请求内容决定（结构性防越权）。
干跑/只要提示词场景：.env 里不配 AGENT_IMAGE_MODEL → 注册集只有
make_plan / review / render。
"""

from __future__ import annotations

from agentkit.tool import Tool
from tools.image import make_generate_tool
from tools.plan import make_plan_tool
from tools.render import make_render_tool
from tools.review import make_review_tool


def build_tools(config) -> list[Tool]:
    """按 config 装配本 agent 的工具集。config: AgentConfig。"""
    tools = [make_plan_tool(), make_review_tool(), make_render_tool()]

    if config.image_enabled:
        tools.append(make_generate_tool())

    return tools
