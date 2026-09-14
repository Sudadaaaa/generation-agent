from dataclasses import dataclass
from pydantic import BaseModel, Field
from core.tool import Tool, Args

class AddArgs(BaseModel):
    """add 的参数"""
    a: int = Field(description=("第一个数字"))
    b: int = Field(description=("第二个数字"))


@dataclass
class AddTool(Tool):
    name: str = "add"
    description: str = "求两个整数之和，返回计算结果"
    args_schema: Args = AddArgs

    def run(self, parameters: AddArgs) -> str:
        return str(parameters.a + parameters.b)