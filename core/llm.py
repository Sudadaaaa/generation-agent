"""LLM 传输层：只收参数，不读环境。

配置的来源是 core/config.py 的 AgentConfig（全项目唯一读环境的地方）；
本模块不做 os.getenv、也不 load_dotenv。
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from core.exceptions import HelloAgentsException


class AgentLLM:
    """OpenAI 兼容的 chat 客户端。"""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        provider: str = "deepseek",
        timeout: int = 60,
    ) -> None:
        missing = [
            name
            for name, value in (
                ("LLM_MODEL_ID", model),
                ("LLM_API_KEY", api_key),
                ("LLM_BASE_URL", base_url),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"缺少配置：{'、'.join(missing)}"
                "（检查仓库根的 .env，键表见 core/config.py）"
            )

        self.model = model
        self.provider = provider
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def invoke(self, messages, stream=False, **kwargs):
        """大模型调用，通过 stream 来控制是否流式。"""
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                **kwargs,
            )
        except Exception as e:
            raise HelloAgentsException(f"LLM调用失败: {str(e)}") from e


if __name__ == "__main__":
    """本层自测入口：不经过 MainAgent，直接验传输层 + 原生 function calling。

    用 `python -m core.llm`（在仓库根跑）或 `python core/llm.py`。
    """
    from core.config import AgentConfig
    from core.io import ConsoleIO
    from tools.math import AddTool
    from core.registry import ToolRegistry

    io = ConsoleIO()
    io.out("=" * 60)
    io.out("开始测试（核心层：LLM 传输 + 工具调用，不经过 MainAgent）")
    io.out("=" * 60)

    cfg = AgentConfig()
    llm = AgentLLM(
        model=cfg.llm_model_id,
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
    )

    history = [{"role": "system",
                "content": "你是一个人工智能，你需要认真分析用户的问题，然后给出准确的回答和解决方案"}]

    tool_registry = ToolRegistry()
    tool_registry.register(AddTool())
    tools = tool_registry.get_openai_schemas()

    # 基础对话测试 + 工具调用测试
    while True:
        request = io.ask("> ").strip()
        if request.lower() == "exit":
            break

        try:
            messages = history + [{"role": "user", "content": request}]
            target = False
            for step in range(5):                       # ← 循环 + 上限，防模型反复调同一个工具
                msg = llm.invoke(messages, tools=tools).choices[0].message

                if msg.content:                         # ← 中间那句“我来算一下”也念出来
                    io.out("🧠 " + msg.content)

                if not msg.tool_calls:                  # ← 收工条件：只看 tool_calls
                    target = True
                    messages.append({"role": "assistant", "content": msg.content or ""})
                    break

                messages.append(msg.model_dump(exclude_none=True))
                for call in msg.tool_calls:
                    result = tool_registry.execute_tool(
                        call.function.name, call.function.arguments)
                    io.out(f"  🔧 {call.function.name}({call.function.arguments}) → {result}")
                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": result})
            if not target:
                io.out("⚠️ 达到最大工具轮数，模型仍未给出最终回答。")

        except (HelloAgentsException, ValueError) as e:
            io.out(str(e))
        else:
            history = messages
