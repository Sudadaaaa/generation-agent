from abc import ABC, abstractmethod
from typing import Optional
from .message import Message
from .llm import AgentLLM
from .config import AgentConfig

# 基类的最小默认提示词。子类通常在自己的模块里定义更具体的提示词并传进来；
# 这里给一句真正能用的，而不是占位符——否则不传 system_prompt 时会把一条
# 空白 system 消息静默发到 API。
SYSTEM_PROMPT = "你是一个乐于助人的AI助手，请用简洁准确的方式回答用户的问题。"

class BaseAgent(ABC):
    """Agent 基类"""
    
    def __init__(
        self,
        name: str,
        llm: AgentLLM,
        system_prompt: Optional[str] = None,
        config: Optional[AgentConfig] = None
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        # 只收下调用方给的东西，不在这里 new 一个：AgentConfig() 会读 .env，
        # 而这个字段目前没有任何读取方——也就是每建一个 agent 就悄悄读一次环境、
        # 然后丢掉。更糟的是它藏在签名背后：签名写着 config 可省，实际省略时
        # 发生了 IO，调用方看不见。全项目读环境的地方应当只有 main.py 一处。
        self.config = config
        self.history: list[Message] = []
        self.history.append(Message("system", self.system_prompt))
        
    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent"""
        pass
    
    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self.history.append(message)

    def clear_history(self):
        """清空历史记录"""
        self.history.clear()
        self.history.append(Message("system", self.system_prompt))

    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self.history.copy()

    def get_messages(self) -> list[dict]:
        messages = []
        for msg in self.history:
            messages.append(msg.to_dict())
        return messages
    
    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"