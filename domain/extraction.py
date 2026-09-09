import json
from typing import Any

from pydantic import ValidationError

from domain.prompts import build_correction_message, build_system_prompt
from domain.schema import GenerationPlan
from llm.base import ChatClient


class PlanFormatError(RuntimeError):
    """模型返回了格式不对的内容（非法 JSON 或不符合 Schema）。可重试修正。"""


def parse_generation_plan(content: str) -> GenerationPlan:
    """
    解析并校验模型返回的内容。

    合法：返回 GenerationPlan。
    不合法：抛出 PlanFormatError，错误信息说明具体原因——
    是没输出合法 JSON、输出的是数组而非 object、还是缺少必填字段 / 字段类型不对等。
    """

    try:
        data: Any = json.loads(content)

    except json.JSONDecodeError as exc:
        raise PlanFormatError(
            "你输出的不是合法 JSON（解析器错误："
            f"{exc}）。请只输出一个完整的 JSON object，"
            "不要带 Markdown 代码块、解释文字或其他内容。"
        ) from exc

    if not isinstance(data, dict):
        raise PlanFormatError(
            "你输出的 JSON 不是 object（应为 {...}）。"
            '请输出形如 {"elements": [...], "overall": "..."} 的 JSON object。'
        )

    try:
        return GenerationPlan.model_validate(data)

    except ValidationError as exc:
        problems = [
            f"- {'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        ]

        raise PlanFormatError(
            "你输出的 JSON 不符合 GenerationPlan Schema，"
            "具体问题如下：\n"
            + "\n".join(problems)
        ) from exc


def create_generation_plan(
    user_input: str,
    *,
    llm: ChatClient,
    max_retries: int = 2,
    context_messages: list[dict[str, str]] | None = None,
) -> GenerationPlan:
    """
    Convert natural language into a validated GenerationPlan.

    每次拿到模型返回后，先打印返回内容，再调用 parse_generation_plan 校验；
    不合法就带着具体错误原因请模型修正，然后进入下一次调用。
    llm 是规划工人客户端，由调用方注入（worker 具体 deepseek 或 qwen）。
    context_messages：可选，追加在初始 user 消息之后（如上一版计划 + 评审修正要求），
    供评审→修正循环复用本函数的格式校验与重试逻辑。
    """

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": build_system_prompt(),
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    if context_messages:
        messages.extend(context_messages)

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):

        try:
            content = llm.chat(messages)

        except RuntimeError as exc:
            # 空内容 / 网络等传输级失败：瞬时故障，直接重试同一次请求，不追加修正
            last_error = exc

            print(f"[planner] 模型调用失败（第 {attempt + 1} 次）：{exc}")

            if attempt >= max_retries:
                break

            continue

        # 先输出模型返回的内容，方便排查
        print(f"[planner] 第 {attempt + 1} 次返回：")
        print(content)
        print()

        try:
            return parse_generation_plan(content)

        except PlanFormatError as exc:

            last_error = exc

            print(f"[planner] 第 {attempt + 1} 次输出不合法：")
            print(exc)
            print()

            if attempt >= max_retries:
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": build_correction_message(exc),
                }
            )

    raise RuntimeError(
        "Model failed to produce a valid GenerationPlan "
        f"after {max_retries + 1} attempts.\n\n"
        f"Last error:\n{last_error}"
    )
