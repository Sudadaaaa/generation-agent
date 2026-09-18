from abc import ABC, abstractmethod
from typing import Optional
from .message import Message
from .llm import AgentLLM
from .config import AgentConfig
from .prompt import BASE_SYSTEM_PROMPT

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
        self.system_prompt = system_prompt or BASE_SYSTEM_PROMPT
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