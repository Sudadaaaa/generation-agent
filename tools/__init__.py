"""本 agent 内置工具来源：build_tools(config) 决定注册集。

环境能力分层：generate_image 只在 config.image_enabled（配了 T2I_MODEL_ID）时注册
——「能不能出图」由环境配置决定，不由用户请求内容决定（结构性防越权：没配生图时
主大脑根本看不到这个工具）。干跑 / 只出提示词 = 不配 T2I_MODEL_ID。

plan 只依赖 LLM，不依赖生图后端，所以**总是注册**。

tools.t2i 顶层是轻的（torch / diffusers 都在方法里惰性 import），所以这里可以
无条件 import 它，不需要旧版那种 `if image_enabled: from generation...` 的条件 import。
"""

from __future__ import annotations

from agent.planagent import PlanAgent
from core.llm import AgentLLM
from core.tool import Tool
from tools.math import AddTool
from tools.plan import PlanTool
from tools.t2i import GenerateImageTool


def build_tools(config, plan_agent=None) -> list[Tool]:
    """按 config 装配工具集。config: AgentConfig。

    plan_agent 是给离线测试用的注入口：不给的话，任何跑 build_tools 的测试都会被
    逼着造一个真的 OpenAI client（AgentLLM 在空 key 上直接抛 ValueError）。
    """
    if plan_agent is None:
        plan_agent = PlanAgent(name="计划", llm=AgentLLM(**config.plan_llm_kwargs))

    tools: list[Tool] = [AddTool(), PlanTool(plan_agent=plan_agent)]

    if config.image_enabled:
        tools.append(GenerateImageTool(
            model=config.t2i_model_id,
            output_dir=config.t2i_outputs_dir,
            device=config.t2i_model_device,
        ))

    return tools
