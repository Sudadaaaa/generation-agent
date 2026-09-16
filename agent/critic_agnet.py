from core.agent import BaseAgent
from core.config import AgentConfig
from core.exceptions import HelloAgentsException
from core.llm import AgentLLM
from core.message import Message
from core.prompt import build_plan_system_prompt, build_correction_message

class PlanAgent(BaseAgent):
    """跑起来整体静默：不流式、不打印，只把最终的 GenerationPlan JSON 交回调用方。

    history 全量保留（就是 BaseAgent 那份），所以「在刚才基础上加 XX」天然成立——
    上一次的需求与计划都在上下文里，模型看得见。
    """

    def __init__(
        self,
        name: str,
        llm: AgentLLM,
        system_prompt: str | None = None,
        config: AgentConfig | None = None,
        max_repairs: int = 2,
        response_format: dict | None = RESPONSE_FORMAT,
    ) -> None:
        super().__init__(name, llm, system_prompt or build_plan_system_prompt(), config)
        self.max_repairs = max_repairs
        # 本次 run() 实际走了几轮修正（0 = 一次过）。只在末尾读，不参与控制流。
        self.repair_count = 0
        # 默认沿用 RESPONSE_FORMAT（vLLM 下会变成 xgrammar 逐 token 掩码）。
        # 显式传 None 可关掉——评估侧靠它量「约束解码值多少」，生产侧不传即维持原行为。
        self.response_format = response_format

    def run(self, requirement: str) -> str:
        # 每次 run 重新计数：同一个 agent 反复调时，累计值没有意义。
        self.repair_count = 0
        self.add_message(Message("user", requirement))

        last: Exception | None = None
        for _ in range(self.max_repairs + 1):
            content = self._complete()
            self.add_message(Message("assistant", content))
            try:
                plan = parse_generation_plan(content)
            except PlanFormatError as e:
                last = e
                self.repair_count += 1
                self.add_message(Message("user", build_correction_message(e)))
                continue

            return plan.model_dump_json(indent=2)

        raise HelloAgentsException(
            f"计划生成失败（{self.max_repairs + 1} 次尝试）：{last}\n"
            "可以直接自己写提示词，或换个措辞再调 plan。"
        )