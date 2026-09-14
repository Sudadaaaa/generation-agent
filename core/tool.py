"""工具基类。

住在 core/ 而不是 tools/：这个抽象跟图像、math、plan 都无关——`name/description/
args_schema` + `parse_args` + `to_openai_schema` 是纯机制，任何 agent 都该能用。
留在 tools/ 会让「子 agent 复用工具机制」变成 `agent/ → tools/ → tools/__init__.py`
（Python 里 `from tools.xxx import Y` 必然先完整执行 `tools/__init__.py`），而后者
要 import agent 来装配——死锁。
"""

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel
from dataclasses import dataclass

Args = type[BaseModel]

@dataclass
class Tool(ABC):
    name: str
    description: str
    args_schema: Args

    @abstractmethod
    def run(self, parameters: BaseModel) -> str:
        """执行工具"""
        pass

    def parse_args(self, arguments: str | dict | None) -> BaseModel:
        """把模型给的 arguments（JSON 字符串或 dict）变成校验过的Args实例。"""
        if isinstance(arguments, str):
            return self.args_schema.model_validate_json(arguments or "{}")
        return self.args_schema.model_validate(arguments or {})

    def to_openai_schema(self) -> dict[str, Any]:
        """生成 OpenAI 兼容 function calling 的工具描述（原生 function calling 用）。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }
