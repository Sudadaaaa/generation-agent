import json
import os
from typing import Any

from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from schemas.generation_plan import GenerationPlan


class PlanFormatError(RuntimeError):
    """DeepSeek 返回了格式不对的内容（非法 JSON 或不符合 Schema）。可重试修正。"""


SYSTEM_PROMPT = """
你是一个 Generative AI 图像提示词规划器。

你的任务是把用户的自然语言图片生成需求转换为严格的
GenerationPlan JSON。

你是一个结构化提取器，不是 Prompt Enhancer。

不要生成最终图片 Prompt。
不要输出解释。
只能输出 JSON。

# 元素拆解（elements）

1. 只把画面中需要单独控制的主要主体拆为元素（elements）：
   主要人物、核心物体、前景关键对象、需要独立描述的自然主体（太阳、山、建筑等）。
   不要为每个名词都拆一个元素；背景、天气、氛围、环境特效写进 overall。
   元素个数由画面内容决定，不设上限。
2. 独立 vs 并入：对画面中每个对象先判断「它是主体，还是附属？」
   - 主体或并列主体（哪怕画面里只有一把伞、一个杯子这样的孤立对象）→ 独立成元素。
   - 依附于另一主体的随身/互动道具（拿在手里、背在身上、作为配饰的伞、剑、书、
     包、乐器、杯子、帽子）→ 并入持有主体的 action 描述（含道具外观），不单独成元素。
3. 同一主体在画面中多次出现（如多宫格同一只猫、同一人物出现在画面多处）：
   只拆为一个元素。固定外观写一次；layout 与 action 是它的出现列表，逐项对齐
   （第 i 项位置对应第 i 项行为）；每次出现特有的事物写进对应的 action 项。

# 字段

4. 每个元素统一用同一套字段：name、count、appearance、layout、action。
   各字段的语义、示例与填写边界以文末 JSON Schema 的字段描述为准，不要自创字段。

# 补全与约束

5. 可选字段（count/appearance/layout/action）不要留 null：按画面场景与元素类型，
   用最合理、最常见的默认值自动补齐，且不得与用户需求冲突。
   例如：单一出现的元素默认 layout=["画面中央，占据主体"]、action=["静止，正视前方"]。
6. overall 只写画面级（全局）属性（风格/构图/光线/天气/氛围/画幅/特效/画质/
   整体排版布局框架），只在需要时写一次；多宫格/分镜的整体框架
   （等大画框、留白边框、几行几列）只写进 overall；
   元素的细节（外观/行为/位置/服饰/表情等）只写进对应 element，overall 不重复。
7. 元素上出现的文字（招牌、书名、标语）保持用户原文逐字保留。
8. 用户明确提供的信息必须尽可能完整保留；自动补齐不得与用户需求冲突；
   不要编造画面外的新主体。

最终只能输出符合 JSON Schema 的 JSON object。
"""

def build_system_prompt() -> str:
    schema = GenerationPlan.model_json_schema()

    return (
        SYSTEM_PROMPT
        + "\n\n以下是必须严格遵循的 JSON Schema：\n\n"
        + json.dumps(
            schema,
            ensure_ascii=False,
            indent=2,
        )
    )

def get_generation_plan_schema() -> dict:
    """
    Get the JSON Schema generated directly from the Pydantic model.
    """

    return GenerationPlan.model_json_schema()

def create_client() -> OpenAI:
    """Create a DeepSeek API client."""

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not configured. "
            "Please add it to your .env file."
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


def call_deepseek(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
) -> str:
    """Call DeepSeek and return its textual response."""

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            "type": "json_object"
        },
        temperature=0.1,
        max_tokens=50000,
        stream=False,
    )

    choice = response.choices[0]
    message = choice.message
    content = message.content

    if not content:
        # deepseek-v4-flash 是推理模型：reasoning_content 与 content 都计入 max_tokens，
        # 推理过长或瞬时故障时 content 可能为空。带上现场信息方便排查。
        reasoning = getattr(message, "reasoning_content", None) or ""

        raise RuntimeError(
            "DeepSeek 返回了空内容（message.content 为空）。"
            f" finish_reason={choice.finish_reason!r}, "
            f"usage={response.usage!r}, "
            f"reasoning_content 长度={len(reasoning)}。"
            "可能是 max_tokens 被推理内容占满或瞬时故障。"
        )

    return content


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


def build_correction_message(error: Exception) -> str:
    """根据上一次输出的错误，生成发给模型去修正的提示词。"""

    return f"""
你上一次返回的内容不合法，请修正后重新输出一个合法的 GenerationPlan JSON。

错误原因：

{error}

请按上述原因修正，并特别注意：

- elements 必须是非空数组，至少包含一个元素
- 同一主体多次出现时（如多宫格同一只猫）只拆为一个元素：appearance 写一次固定外观，layout/action 用列表逐项对齐（第 i 项位置对应第 i 项行为），每次出现特有的事物写进对应 action 项
- name 只写主体身份（'橘猫'），不要写成'一格橘猫睡觉的画面'这种包含其他字段信息的描述；'哪一格/位置'写进 layout
- action 每项是一次出现的行为/互动/当格特有细节；主语就是元素本身，不要重述名称（例如写'正在睡觉'，不写'主语正在睡觉'）
- 依附于主体的随身/互动道具（伞、剑、书等）并入持有者的 action；对象本身就是主体则独立成元素
- 多宫格/分镜排版的整体框架（等大画框、留白边框、几行几列）写进 overall 一次
- 字段类型：count/appearance 是字符串或 null；layout/action 是字符串数组（每项一次出现）或 null
- layout 与 action 都填写时数组长度必须一致（逐项对齐）
- overall 必须是字符串或 null；只写画面级全局属性，不要复述元素级细节或整段照抄用户原文
- 可选项尽量填合理默认值，只有完全无法推断时才用 null
- 只能返回 JSON，不要输出解释
"""

def create_generation_plan(
    user_input: str,
    max_retries: int = 2,
) -> GenerationPlan:
    """
    Convert natural language into a validated GenerationPlan.

    每次拿到模型返回后，先打印返回内容，再调用 parse_generation_plan 校验；
    不合法就带着具体错误原因请模型修正，然后进入下一次调用。
    """

    client = create_client()

    model = os.getenv(
        "LLM_MODEL",
        "deepseek-v4-flash",
    )

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

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):

        try:
            content = call_deepseek(
                client=client,
                model=model,
                messages=messages,
            )

        except (RuntimeError, OpenAIError) as exc:
            # 空内容 / 网络等传输级失败：瞬时故障，直接重试同一次请求，不追加修正
            last_error = exc

            print(f"[planner] DeepSeek 调用失败（第 {attempt + 1} 次）：{exc}")

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
        "DeepSeek failed to produce a valid GenerationPlan "
        f"after {max_retries + 1} attempts.\n\n"
        f"Last error:\n{last_error}"
    )