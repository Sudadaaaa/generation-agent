if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.agent import BaseAgent
from core.llm import AgentLLM
from core.config import AgentConfig
from core.message import Message
from core.io import IO, ConsoleIO
from core.exceptions import HelloAgentsException
from core.registry import ToolRegistry
from core.prompt import MAIN_SYSTEM_PROMPT
from typing import Optional

class MainAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        llm: AgentLLM,
        system_prompt: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        io: Optional[IO] = None
    ):
        super().__init__(name, llm, system_prompt or MAIN_SYSTEM_PROMPT, config)
        self.tool_registry = tool_registry or ToolRegistry()
        self.io = io or ConsoleIO()

    def run(self, input_text, max_tool_iterations: int = 10, stream=False, **kwargs):
        """
        Agent运行函数，通过stream控制是否流式输出，支持工具调用 
        """
        self.add_message(Message("user", input_text))          # ← 先落库
        tools = self.tool_registry.get_openai_schemas() or None

        for _ in range(max_tool_iterations):
            response = self.llm.invoke(self.get_messages(), tools=tools, stream=stream, **kwargs)
            msg = self.process_response(response, stream)
            self.add_message(msg)

            if not msg.tool_calls:
                # 收尾回复。模型拒答/空回复时 content 是 None：提示用户一句，
                if not msg.content:
                    refusal = getattr(msg, "refusal", None)
                    self.io.out(f"⚠️ 模型拒绝回答：{refusal}" if refusal
                                else "⚠️ 模型没有返回内容")
                return msg.content or ""

            for call in msg.tool_calls:
                name = call["function"]["name"]
                arguments = call["function"]["arguments"]
                result = self.tool_registry.execute_tool(name, arguments)
                self.io.out(f"🔧 {name}({arguments}) → {result}")
                self.add_message(Message(role="tool", content=result, tool_call_id=call["id"]))
        raise HelloAgentsException(f"达到最大工具轮数 {max_tool_iterations}")

    def process_response(self, response, stream: bool = False) -> Message:
        """
        把一次 LLM 往返的返回，统一成一条 Message。
        不同的provider 的 SDK 类型到此为止都变成统一的Message，

        - 非流式：response.choices[0].message 是 SDK 对象，转成 Message
        - 流式：边显示边把 delta 碎片重组成一条完整消息
          ⚠️ 不能只累加 content —— 模型调工具时 content 全程是 None，
             真正的信息在 delta.tool_calls 里，要按 index 归位、arguments 累加
        """
        if not stream:
            msg = response.choices[0].message
            if msg.content:
                self.io.out("🧠：" + msg.content)
            return Message.model_validate(msg.model_dump(exclude_none=True))

        content = ""
        role = "assistant"
        fragments: dict[int, dict] = {}      # index -> {"id", "name", "arguments"}
        started = False                      # 还没吐过字就先不打印前缀

        for chunk in response:
            delta = chunk.choices[0].delta

            if delta.role:
                role = delta.role

            if delta.content:
                if not started:
                    self.io.out("🧠：", end="")
                    started = True
                content += delta.content
                self.io.out(delta.content, end="")

            # 工具调用是【碎片】：第一片带 id 和 name，后续片只追加 arguments
            for tc in delta.tool_calls or []:
                frag = fragments.setdefault(
                    tc.index, {"id": None, "name": None, "arguments": ""})
                if tc.id:
                    frag["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        frag["name"] = tc.function.name
                    if tc.function.arguments:
                        frag["arguments"] += tc.function.arguments

        if started:
            self.io.out()

        # id 由协议保证会出现，但不是所有 provider 都守信：缺了就补一个，
        # 否则 tool 消息会带着 tool_call_id=None 撞上 Message 的不变量，
        # 在工具循环中途炸掉。
        tool_calls = [
            {"id": frag["id"] or f"call_{idx}", "type": "function",
             "function": {"name": frag["name"], "arguments": frag["arguments"]}}
            for idx, (_, frag) in enumerate(sorted(fragments.items()))
        ] or None

        return Message(role=role, content=content or None, tool_calls=tool_calls)


if __name__ == "__main__":
    """本层自测入口：不经过 main.py，直接验 MainAgent 的对话 + 流式重组。

    用 `python -m agent.main_agent`（在仓库根跑）或 `python agent/main_agent.py`。
    不传 tool_registry，跑的是一个空注册表——这里只验对话与流式重组，不掺工具。
    """
    from core.config import AgentConfig

    cfg = AgentConfig()
    agent = MainAgent(
        name="基础助手",
        llm=AgentLLM(
            model=cfg.llm_model_id,
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_base_url,
        ),
        system_prompt="你是一个友好的AI助手，请用简洁明了的方式回答问题。",
    )
    print("── 非流式 ──")
    agent.run("你好，请用一句话介绍你自己")
    print("\n── 流式 ──")
    agent.run("你好，请用一句话介绍你自己", stream=True)