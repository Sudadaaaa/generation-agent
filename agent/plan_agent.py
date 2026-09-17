"""plan 子 agent：自然语言需求 → 结构化 GenerationPlan。

组合进 tools/plan.py 的 PlanTool，与 GenerateImageTool 持有 ImageGenerator 同构：
工具只管参数与回执，拆解这件事整个在这里。

system 提示词不在这里定义——唯一真源是 core/prompt.py 的 build_plan_system_prompt()。
本模块只调用它、不存副本；训练侧 mydatasets/plan_sft.py 调用的是同一个函数，
两边靠 import 绑定，而不是靠人工同步。

这么改是因为踩过一次真实的坑：从前推理侧现拼、训练侧冻结成一个字符串字面量，
只靠注释提醒同步。后来 core/schema.py 里 GenerationPlan 的类 docstring 改了措辞，
而 pydantic 会把它嵌成 schema 顶层的 description——训练语料（sha1 510f9d91）
与推理提示词（3b8c905a）就此不一致，且没有任何机制会报错，是比对哈希才发现的。
现在副本在物理上不存在了。
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import re

from pydantic import ValidationError

from core.agent import BaseAgent
from core.config import AgentConfig
from core.exceptions import HelloAgentsException
from core.llm import AgentLLM
from core.message import Message
from core.prompt import build_plan_system_prompt, build_correction_message
from core.schema import GenerationPlan


class PlanFormatError(RuntimeError):
    """模型这次输出的不合格式。带原因，用来生成给它的修正指令。"""


# 围栏（```json ... ```）与收尾空白。模型不总听话，剥掉再解析。
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


def parse_generation_plan(text: str) -> GenerationPlan:
    """模型正文 → GenerationPlan。三类失败给三种不同措辞的理由。"""
    body = _FENCE.sub("", text.strip()).strip()

    if not body.startswith("{"):
        # 兜底：模型常在 JSON 前加一句「好的，以下是计划：」，就取首 { 到末 }
        start, end = body.find("{"), body.rfind("}")
        if start != -1 and end > start:
            body = body[start:end + 1]

    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise PlanFormatError(
            f"你输出的不是合法 JSON（{e}）。只输出一个完整的 JSON object，"
            "不要 Markdown 围栏、不要任何解释文字。"
        ) from e

    if not isinstance(data, dict):
        raise PlanFormatError('你输出的 JSON 不是 object（应该是 {"elements": [...]}）。')

    try:
        return GenerationPlan.model_validate(data)
    except ValidationError as e:
        # 带上精确字段路径（elements.1.identity 这种），模型才改得准
        problems = "\n".join(
            f"- {'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()
        )
        raise PlanFormatError(f"不符合 GenerationPlan Schema：\n{problems}") from e

RESPONSE_FORMAT = {"type": "json_object"}
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

    def _complete(self) -> str:
        """一次调用，返回正文。

        两种「重试也没用」的失败在这里直接抛，不占修正预算：
        - 空 content：推理模型的思考与正文共用一个输出预算，思考吃满时正文会是空的，
          同一份请求再发一次结果一样。
        - finish_reason == 'length'：正文被 max_tokens 截断。截断的 JSON 必然非法，
          而修正指令要求「重新输出【完整】的 JSON object」——只会再截断一次。
        第二个判断必须在解析【之前】：截断是 json_object 模式最常见的解析失败原因，
        不先判的话，模型会收到一条「你输出的不是合法 JSON」的误导性修正指令。
        """
        # 关约束时不能传 response_format=None——SDK 会把它序列化成 null 发出去，
        # 只能整个省掉这个键。
        fmt = {"response_format": self.response_format} if self.response_format else {}
        response = self.llm.invoke(self.get_messages(), **fmt)
        choice = response.choices[0]
        content = (choice.message.content or "").strip()

        if not content:
            raise HelloAgentsException(
                f"plan 模型返回了空内容（finish_reason={choice.finish_reason}，"
                f"usage={response.usage}）。"
                "推理模型的思考与正文共用一个输出预算，重发同一份请求没有意义。"
            )
        if choice.finish_reason == "length":
            raise HelloAgentsException(
                f"plan 输出被 max_tokens 截断（usage={response.usage}）。"
                "截断的 JSON 必然解析失败，重发也只会再截断一次；"
                "请调高 max_tokens，或把需求拆小一点。"
            )
        return content

if __name__ == "__main__":
    """本层自测入口：不经过 main.py / PlanTool，直接验 plan agent 的产出与修正循环。

    用 `python -m agent.plan_agent`（在仓库根跑）或 `python agent/plan_agent.py`。
    连说两句「画一只橘猫蹲坐在窗台上，黄昏暖光」→「在刚才基础上加一副墨镜」，
    可验 history 全量保留。
    """
    from core.config import AgentConfig
    from core.io import ConsoleIO

    _EXIT_WORDS = {"exit", "quit", "退出"}

    io = ConsoleIO()
    cfg = AgentConfig()
    agent = PlanAgent(name="计划", llm=AgentLLM(**cfg.plan_llm_kwargs))

    io.out("=" * 60)
    io.out(f"plan agent 模型：{agent.llm.model}")
    io.out("=" * 60)

    while True:
        request = io.ask("> ").strip()
        if not request:
            continue
        if request.lower() in _EXIT_WORDS:
            break

        try:
            io.out(agent.run(request))
        except HelloAgentsException as e:
            io.out(f"\n⚠️ {e}")
