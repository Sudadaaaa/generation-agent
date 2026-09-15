"""环境配置：.env / 环境变量 → AgentConfig。

全项目唯一读环境的地方。业务代码里不该再出现 os.getenv——core/llm.py 只收参数，
.env 由这里通过 pydantic-settings 的 env_file 统一加载。

  LLM_MODEL_ID        大脑模型 id
  LLM_API_KEY         API 密钥
  LLM_BASE_URL        服务地址；缺省 https://api.deepseek.com
  PLAN_MODEL_ID       plan 子 agent 的模型 id；**留空 = 沿用 LLM_MODEL_ID**
  PLAN_API_KEY        plan 子 agent 的密钥；**留空 = 沿用 LLM_API_KEY**
  PLAN_BASE_URL       plan 子 agent 的服务地址；**留空 = 沿用 LLM_BASE_URL**
  PLAN_PROVIDER       plan 子 agent 的服务商标识；**留空 = deepseek**。
                      纯显示用（core/agent.py 的 __str__），不参与任何分支判断——
                      「按 schema 约束输出」用的是同一个 response_format，
                      在 DeepSeek 与 vLLM 上写法一致，不需要按 provider 翻译
  T2I_MODEL_ID        生图后端，取 tools/t2i.py 里 MODELS 的键
                      （Z-Image-Turbo / FLUX.2-klein-9B）；**留空 = 无生图能力**，
                      注册的工具集里就没有 generate_image（干跑 / 只出提示词的验证方式）
  T2I_MODEL_DEVICE    生图卡；缺省 cuda:0
  T2I_OUTPUTS_DIR     图片输出目录；缺省 outputs

优先级（pydantic-settings 处理）：构造参数 > 环境变量 > .env > 字段缺省。

不进环境变量的：步数 / guidance_scale / 分辨率 / 种子 / 最大重试次数——那些是
「模型怎么跑」的配方，属于代码，在 tools/t2i.py 的 _ModelSpec 与
agent/planagent.py 的 PlanAgent / RESPONSE_FORMAT 里。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE_DIR = Path(__file__).resolve().parent.parent      # 仓库根：从任何目录跑都对


class AgentConfig(BaseSettings):
    """本 agent 的装配配置。"""

    model_config = SettingsConfigDict(
        env_file=_BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",        # .env 里多出来的键不报错
        case_sensitive=False,  # 字段 llm_model_id ← 环境变量 LLM_MODEL_ID
        # 现在字段名与键名规则一致（字段名大写 = 键名），没有任何 validation_alias，
        # 所以这一行在当前代码里是空转的。留着是防以后：一旦给某个字段加了 alias，
        # 忘了同时开这一行的话，按字段名传参会被 extra="ignore" 静默吃掉——很难查。
        populate_by_name=True,
    )

    # ── LLM（传输层）：字段名大小写不敏感地匹配 LLM_* 三个键，无需 alias ──
    llm_model_id: str = "deepseek-v4-flash"
    llm_api_key: str = Field(default="", repr=False)      # ← 防密钥进日志/回溯
    llm_base_url: str = "https://api.deepseek.com"

    # ── plan 子 agent 的 LLM：与 LLM_* 同一条规则，字段名大写即键名 ──
    # 三项各自留空 = 沿用同名 LLM_*（同一家服务商，只是换模型）。
    # 这是对「凭证按服务商分、不按 agent 分」的一处【有意偏离】：这三项是模型覆盖，
    # 不是新的一组凭证；真要换服务商，仍应另开一组（形状见 .env.example 末尾）。
    plan_model_id: str = ""
    plan_api_key: str = Field(default="", repr=False)     # ← 与 llm_api_key 同理：防密钥进日志
    plan_base_url: str = ""
    plan_provider: str = ""                               # ← 纯显示用，见模块 docstring

    # ── 生图（工具集装配）：与 LLM_* 同一条规则，字段名大写即键名 ──
    t2i_model_id: str = ""
    t2i_model_device: str = "cuda:0"
    t2i_outputs_dir: str = "outputs"

    @property
    def plan_llm_kwargs(self) -> dict[str, str]:
        """PlanAgent 用的 AgentLLM 构造参数（键名与 AgentLLM.__init__ 一一对应）。

        回退是必须的：.env 里没有 PLAN_* 时 plan 也要能跑起来（plan 工具总是注册）。
        想给 plan 换模型才去填，否则零配置即可用。
        """
        return {
            "model": self.plan_model_id.strip() or self.llm_model_id,
            "api_key": self.plan_api_key.strip() or self.llm_api_key,
            "base_url": self.plan_base_url.strip() or self.llm_base_url,
            "provider": self.plan_provider.strip() or "deepseek",
        }

    @property
    def image_enabled(self) -> bool:
        """生图是否可用 = 是否配置了生图后端（环境能力，非用户请求决定）。"""
        return bool(self.t2i_model_id.strip())
