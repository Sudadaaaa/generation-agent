"""计划评审：对照用户需求检查 GenerationPlan 的质量。"""

import json
from dataclasses import dataclass, field

from openai import OpenAIError

from agent.prompts import CRITIQUE_SYSTEM_PROMPT
from planning.llm import PlanLLM
from planning.schema import GenerationPlan


@dataclass
class CritiqueVerdict:
    """评审结论。ok=False 时 problems 给出具体修正方向。"""

    ok: bool
    problems: list[str] = field(default_factory=list)


def critique_plan(
    user_input: str,
    plan: GenerationPlan,
    llm: PlanLLM,
) -> CritiqueVerdict:
    """让评审模型对照需求评审计划。

    llm 是评审模型，建议与规划模型解耦（独立后端，如 DeepSeek），
    由调用方传入；同模型自评容易「盖章式通过」。
    评审是软质量门：调用失败或返回内容无法解析时按「通过」处理
    （fail-open），打印 [agent] 警告，不硬阻塞主流程。
    """

    messages = [
        {
            "role": "system", 
            "content": CRITIQUE_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": (
                "用户的图片生成需求：\n"
                f"{user_input}\n\n"
                "当前 GenerationPlan：\n"
                f"{plan.model_dump_json(indent=2)}"
            ),
        },
    ]

    try:
        content = llm.complete(messages)
    except (RuntimeError, OpenAIError) as exc:
        print(f"[agent] 评审调用失败，按通过处理：{exc}")
        return CritiqueVerdict(ok=True)

    try:
        data = json.loads(content)

        problems = data.get("problems", [])
        if not isinstance(problems, list):
            problems = []

        return CritiqueVerdict(
            ok=bool(data["ok"]),
            problems=[str(p) for p in problems],
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"[agent] 评审输出无法解析，按通过处理：{exc}")
        print(content)
        return CritiqueVerdict(ok=True)
