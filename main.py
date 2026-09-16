"""入口：装配工具集 → 进对话主循环。

整场对话一份 history（在 MyAgent 里），直到 exit。每次用户输入、工具调用结果、
模型返回都按序追加进同一份 history，模型基于上下文反复执行。
重置对话 = 重跑 main（重跑即全新会话）。
"""

from agent.main_agent import MainAgent
from core.config import AgentConfig
from core.io import ConsoleIO
from core.llm import AgentLLM
from core.exceptions import HelloAgentsException
from tools import build_tools
from core.registry import ToolRegistry

_EXIT_WORDS = {"exit", "quit", "退出"}


def main() -> None:
    """入口"""
    io = ConsoleIO()
    config = AgentConfig()          # 自己读仓库根的 .env（见 core/config.py）

    registry = ToolRegistry()
    for tool in build_tools(config):
        registry.register(tool)

    agent = MainAgent(
        name="助手",
        llm=AgentLLM(
            model=config.llm_model_id,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
        ),
        tool_registry=registry,
        io=io,
    )

    io.out("=" * 60)
    io.out("Agent 服务已启动")
    io.out(f"  生图：{'已启用（' + config.t2i_model_id + '）' if config.image_enabled else '未配置——不注册 generate_image'}")
    io.out(f"  工具：{', '.join(registry.list_tools())}")
    io.out("=" * 60)

    while True:
        request = io.ask("> ").strip()

        if not request:
            io.out("开始聊天吧！")
            continue

        if request.lower() in _EXIT_WORDS:
            break

        # 一次网络抖动不该结束整场会话：错误念给用户听，循环继续。
        try:
            agent.run(request, stream=True)
        except HelloAgentsException as e:
            io.out(f"\n⚠️ {e}")


if __name__ == "__main__":
    main()
