import os

# 物理卡选择：本地 LLM 固定逻辑 cuda:0、生图固定逻辑 cuda:1。
# 这里决定这两张逻辑卡映射到哪两张物理卡（换卡只改这一行）。
# 必须在任何 torch 导入之前设置，否则掩码不生效。
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

from dotenv import load_dotenv

from agentkit.io import ConsoleIO
from agentkit.runtime import ContextLimit, RuntimeCtx, agent_act
from agentkit.tool import ToolRegistry

_EXIT_WORDS = {"exit", "quit", "退出"}


def build_session(config):
    """构建整场会话共享的服务：clients / generator / registry。

    - 客户端与生图后端**只建一次、整场复用**：生图模型不随每条需求重载；
      usage 在客户端对象里整场累计，退出打印的才是整场真实用量。
    - 会话状态（ctx.state / transcript）跨消息持续，不随消息重置。
    """
    from llm.factory import make_chat_client

    clients = {
        "brain": make_chat_client("deepseek", model=config.brain_model),
        "worker": make_chat_client(
            config.worker_provider,
            model=config.worker_model,
            lora_path=config.worker_lora,
        ),
        "critic": make_chat_client("deepseek", model=config.critic_model),
    }

    generator = None
    if config.image_enabled:
        from generation.factory import create_image_generator

        generator = create_image_generator(
            model=config.image_model,
            output_dir=config.output_dir,
        )

    from tools import build_tools

    registry = ToolRegistry(build_tools(config))
    return clients, generator, registry


def new_session(config, *, services=None, io=None) -> None:
    """主循环：main 直接就是一个 while——整场对话一份 transcript，直到 exit。

    每次用户输入、工具调用结果、模型返回都按序追加进同一份 transcript（history），
    模型基于上下文反复执行，直到你 exit、整场对话超出上下文（提示重跑 main），
    或大脑连续传输失败交回给你。重置对话 = 重跑 main（重跑即全新会话）。

    services/io 可注入（离线测试用）：services=(clients, generator, registry)，
    缺省由 build_session 构建；io 缺省 ConsoleIO。
    """
    io = io or ConsoleIO()
    clients, generator, registry = services or build_session(config)

    io.out("=" * 60)
    io.out("Generative AI Agent（大脑 = DeepSeek ReAct，整场连续会话）")
    io.out("=" * 60)
    io.out(
        f"  大脑 {config.brain_model} | plan 工人 {config.worker_provider} "
        f"{config.worker_model if config.worker_provider == 'deepseek' else '+lora'}"
        f"{' | 生图已启用(' + config.image_model + ')' if config.image_enabled else ' | 干跑（未配生图后端，只出计划/提示词）'}"
    )
    io.out("=" * 60)
    io.out("\n请输入你的图片生成需求（exit 退出）。")
    io.out("提示：说「只要提示词」就只出提示词；想真出图需在 .env 配 AGENT_IMAGE_MODEL。")
    io.out("连续会话：整场历史都算数——「在刚才基础上加 XX 再生成」可直接续作；重跑 main 即新会话。\n")

    # 整场唯一 ctx 与 transcript：所有 user/assistant/tool 消息都在这里，不清零
    ctx = RuntimeCtx(
        io=io,
        registry=registry,
        clients=clients,
        generator=generator,
    )
    history: list[dict] = []
    human_turn = True

    while True:
        if human_turn:
            request = io.ask("> ").strip()

            if request.lower() in _EXIT_WORDS:
                break

            if not request:
                io.out("请输入一些内容。")
                continue

            ctx.user_input = request
            history.append({"role": "user", "content": request})
            human_turn = False

        else:
            try:
                final = agent_act(ctx, history)
            except ContextLimit as exc:
                io.out("\n" + str(exc))
                io.out("已达上下文上限：请退出并重新运行 main，开始一段新会话。")
                break

            if final is None:
                continue  # 大脑这一轮调了工具：继续让它行动，不读用户输入

            io.out("\n" + "-" * 60)
            io.out("[结果] " + final)
            io.out("=" * 60 + "\n")
            human_turn = True

    io.out("\nBye！本次会话用量汇总：")
    for key, client in clients.items():
        io.out(f"  {key}（{client.provider} {client.model}）：{client.usage.describe()}")


def main() -> None:
    """入口：不做命令行参数解析，直接进主循环读用户输入。"""
    load_dotenv()

    from config import AgentConfig

    new_session(AgentConfig.from_env())


if __name__ == "__main__":
    main()
