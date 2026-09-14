"""消息系统"""
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, ConfigDict, model_validator

# 定义消息角色的类型，限制其取值
MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """
    一条对话消息。它同时是两个东西：
      - 你这边 history 里存的对象
      - 发给 API 的 wire 格式（to_dict() 就是为这个存在的）

    设计要点：
      1. wire 字段显式声明 —— 声明的是「契约」，不是「列全」
      2. extra="allow" 兜底 —— API 新增的字段原样收下、原样吐出，不丢
      3. 不变量构造时就报错 —— 不用等 API 返回 400
    """

    model_config = ConfigDict(extra="allow")      # 兜底：不认识的字段一律原样收下

    # ---- wire 字段（API 认识的） ----
    role: MessageRole
    content: Optional[str | List[Dict[str, Any]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    @model_validator(mode="after")
    def _check_invariants(self):
        """把 API 的隐式规则，变成构造时的硬错误。"""
        if self.role in ("system", "user") and not self.content:
            raise ValueError(f"role={self.role!r} 的消息必须有 content")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("role='tool' 的消息必须带 tool_call_id")
        return self

    def __init__(self, role: MessageRole, content=None, **kwargs):
        # 全量转发：不挑拣字段，一律交给 pydantic
        super().__init__(role=role, content=content, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """转成发给 API 的 wire 格式：默认全量保留，只剔空值。"""
        return self.model_dump(exclude_none=True)

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
