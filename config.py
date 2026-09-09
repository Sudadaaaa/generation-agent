"""环境配置：.env → AgentConfig。

新键（均可省略，省略走默认值；旧的 DEEPSEEK_API_KEY / LLM_MODEL 继续生效）：

  AGENT_BRAIN_MODEL      大脑（DeepSeek）模型；缺省取 LLM_MODEL，再缺省 deepseek-chat
  AGENT_WORKER_PROVIDER  plan 工人提供方：deepseek（默认）| qwen（本地 LoRA，训练完成后切换）
  AGENT_WORKER_MODEL     plan 工人模型（deepseek 时）；缺省同上
  AGENT_WORKER_LORA      本地 qwen 的 LoRA adapter 目录（qwen 时）
  AGENT_CRITIC_MODEL     机器评审模型（固定 deepseek，与规划解耦）；缺省同上
  AGENT_IMAGE_MODEL      生图后端（zimage / flux / 完整模型 id）；**留空 = 无生图能力**，
                         注册的工具集里就没有 generate（干跑/只出提示词的验证方式）
  AGENT_OUTPUT_DIR       图片输出目录；缺省 outputs

环境能力分层：是否注册生图工具由 AGENT_IMAGE_MODEL 是否配置决定（结构上防越权），
不由用户请求内容决定。

会话结束无预算上限：main 就是主循环，结束于用户 exit / 上下文超限 / 大脑连续传输失败
交回用户（见 agentkit/runtime）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _default_model(env: dict[str, str], name: str) -> str:
    """解析某角色的模型：优先 AGENT_<NAME>_MODEL，其次旧 LLM_MODEL，最后 deepseek-chat。"""
    key = f"AGENT_{name}_MODEL"
    return env.get(key) or env.get("LLM_MODEL") or "deepseek-chat"


@dataclass
class AgentConfig:
    # 大脑：本轮固定 DeepSeek 云端（tool calling 由 llm/deepseek 提供）。
    brain_provider: str = "deepseek"
    brain_model: str = "deepseek-chat"

    # plan 工人：默认 deepseek；qwen = 本地 Qwen3-8B(+LoRA)，协议不变（仅 chat）。
    worker_provider: str = "deepseek"      # deepseek | qwen
    worker_model: str = "deepseek-chat"    # worker_provider=deepseek 时生效
    worker_lora: str = ""                  # worker_provider=qwen 时的 LoRA adapter 目录

    # 机器评审：固定 deepseek，与规划解耦（防「盖章式通过」）。
    critic_model: str = "deepseek-chat"

    # 生图可用性（环境能力）：留空 = 无生图后端 → 注册集不含 generate。
    image_model: str = ""
    output_dir: str = "outputs"

    @property
    def image_enabled(self) -> bool:
        """生图是否可用 = 是否配置了生图后端（环境能力，非用户请求决定）。"""
        return bool(self.image_model.strip())

    @classmethod
    def from_env(
        cls,
        env: dict[str, str] | None = None,
    ) -> "AgentConfig":
        env = os.environ if env is None else dict(env)

        return cls(
            brain_provider="deepseek",
            brain_model=_default_model(env, "BRAIN"),
            worker_provider=env.get("AGENT_WORKER_PROVIDER", "deepseek").strip().lower(),
            worker_model=_default_model(env, "WORKER"),
            worker_lora=env.get("AGENT_WORKER_LORA", "").strip(),
            critic_model=_default_model(env, "CRITIC"),
            image_model=env.get("AGENT_IMAGE_MODEL", "").strip(),
            output_dir=env.get("AGENT_OUTPUT_DIR", "outputs").strip() or "outputs",
        )

    def __post_init__(self) -> None:
        if self.worker_provider not in ("deepseek", "qwen"):
            raise ValueError(
                f"AGENT_WORKER_PROVIDER={self.worker_provider!r} 不支持"
                "（可选 deepseek / qwen）"
            )
