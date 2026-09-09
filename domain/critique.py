"""计划评审：对照用户需求检查 GenerationPlan 的质量。"""

import json
from dataclasses import dataclass, field

from domain.prompts import CRITIQUE_SYSTEM_PROMPT
from domain.schema import GenerationPlan
from llm.base import ChatClient


@dataclass
class CritiqueVerdict:
    """评审结论。ok=False 时 problems 给出具体修正方向。"""

    ok: bool
    problems: list[str] = field(default_factory=list)


def critique_plan(
    user_input: str,
    plan: GenerationPlan,
    llm: ChatClient,
    user_additions: list[str] | None = None,
) -> CritiqueVerdict:
    """让评审模型对照需求评审计划。

    llm 是评审模型，建议与规划模型解耦（独立后端，如 DeepSeek），
    由调用方传入；同模型自评容易「盖章式通过」。
    评审是软质量门：调用失败或返回内容无法解析时按「通过」处理
    （fail-open），打印 [agent] 警告，不硬阻塞主流程。

    user_additions —— 用户在评审过程中逐步补充的修改意见（累计，含已生效的）。
    评审必须把「原始需求 + 这些补充意见」一起当作完整的用户需求来评判：
    计划里为实现这些补充要求而新增/改动的内容是合法的，不得判成
    「凭空编造/与需求冲突」。缺省 None = 只看原始需求（自动循环 agent_plan 用）。
    """

    parts = [f"用户的图片生成需求：\n{user_input}"]

    if user_additions:
        additions = "\n".join(f"- {a}" for a in user_additions)
        parts.append(
            "用户在上述需求提出后，又逐步补充了以下修改要求"
            "（同样属于用户需求的一部分，优先级与原始需求相同："
            "计划里为落实这些要求而新增或调整的内容是合法的，"
            "不要把它们当成凭空编造或与需求冲突去挑错）：\n"
            f"{additions}"
        )

    parts.append(f"当前 GenerationPlan：\n{plan.model_dump_json(indent=2)}")

    messages = [
        {
            "role": "system",
            "content": CRITIQUE_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": "\n\n".join(parts),
        },
    ]

    try:
        content = llm.chat(messages)
    except RuntimeError as exc:
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
