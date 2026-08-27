import json
import os
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from schemas.generation_plan import GenerationPlan

SYSTEM_PROMPT = """
你是一个 Generative AI 图像提示词规划器。

你的任务是把用户的自然语言图片生成需求转换为严格的
GenerationPlan JSON。

你是一个结构化提取器，不是 Prompt Enhancer。

不要生成最终图片 Prompt。
不要输出解释。
只能输出 JSON。

规则：

1. 必须严格遵循提供的 JSON Schema。
2. 不要增加 Schema 中不存在的字段。
3. 不要删除必要字段。
4. 把画面拆解为多个基本元素（elements），每个元素是画面中一个可指认的事物。
5. 每个元素必须包含 name。
6. 用户没有明确给出的可选字段（count/appearance/position/size/action/relation），
   不要留 null，应根据画面场景与元素类型，用最合理、最常见的默认值自动补齐。
   例如：单一主体默认放在画面中央、占据主体；人物默认补全常见的中性外观与姿态
   （如"少女，长发，浅色上衣，自然表情"）。
   自动补齐必须与用户需求一致，不得冲突。
7. 动作引发的瞬态效果（如：跳跃溅起的水花、扬起的尘土）写入该元素的 action。
8. 元素外观（appearance）应按元素类型覆盖完整：
   人物→年龄感、体型、发型、发色、五官、肤色、服饰、表情；
   物体→形状、颜色、材质、纹理、破损或新旧，及物体上印刻的文字（保持原文）；
   自然元素→形态、颜色、光照感。
   用户未提到的方面用该类元素最常见的中性默认补全。
9. 描述元素关系时，引用其他元素的 name（如"站在草坪上"），
   也可引用'镜头/天空/远方/画面外'等全局参照（如"凝视镜头"）。
10. 画面级属性（overall）用一段简洁连贯的中文描述，只写画面级（全局）属性，按需覆盖：
    风格（写实摄影/Cinematic/2D动画/3D CG/水彩/水墨/赛博朋克/复古胶片等）、
    构图与镜头（景别、机位角度、透视、景深）、光线（时间、光源、方向）、
    天气与环境、色调与氛围、画幅比、视觉特效（光斑、倒影、颗粒感、长曝光）、
    画质与细节、不应出现的内容。
    注意：
    - overall 只放画面级属性；元素的细节（外观/动作/位置/服饰/表情/五官等）只写进对应 element，
      不要在 overall 中复述或整段照抄用户原文；
    - 写成简洁的一到两句话，不要用"构图：""光线：""色调："之类的标签罗列；
    - 用户未提到的方面按画面最合理的方式默认补齐，只有完全无法推断时才用 null。
11. 元素上出现的文字（如招牌、书名、标语）属于该元素的描述，必须保持用户原文逐字保留，不得改写或翻译。
12. 用户明确提供的信息必须尽可能完整地保留。
13. 自动补齐的默认信息不得与用户需求冲突；不要凭空编造与画面无关的新元素或实体。

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
        max_tokens=4000,
        stream=False,
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "DeepSeek returned empty content."
        )

    return content


def parse_generation_plan(content: str) -> GenerationPlan:
    """Parse JSON and validate it against GenerationPlan."""

    try:
        data: Any = json.loads(content)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "DeepSeek returned invalid JSON."
        ) from exc

    try:
        return GenerationPlan.model_validate(data)

    except ValidationError as exc:
        raise exc

def create_generation_plan(
    user_input: str,
    max_retries: int = 2,
) -> GenerationPlan:
    """
    Convert natural language into a validated GenerationPlan.

    If DeepSeek returns valid JSON but the structure does not match
    GenerationPlan, automatically ask DeepSeek to correct the output.
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

            plan = parse_generation_plan(content)

            return plan

        except ValidationError as exc:

            last_error = exc

            print(
                f"[planner] 第 {attempt + 1} 次输出不符合 Schema"
                f"（{exc}），已请求模型修正。"
            )

            if attempt >= max_retries:
                break

            correction_message = f"""
你上一次返回的 JSON 不符合 GenerationPlan Schema。

验证错误如下：

{exc}

请修正 JSON。

特别注意：

- elements 必须是非空数组，至少包含一个元素
- 每个 element 必须包含 name
- element 的字段（count/appearance/position/size/action/relation）必须是字符串或 null
- overall 必须是字符串或 null
- 可选项尽量填合理默认值，只有完全无法推断时才用 null
- overall 只写画面级全局属性（风格/构图/光线/氛围等），不要复述元素级细节或整段照抄用户原文
- 不要把数组/对象字段写成字符串
- 只能返回 JSON
- 不要输出解释
"""

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": correction_message,
                }
            )

        except RuntimeError as exc:

            last_error = exc

            print(f"[planner] DeepSeek 调用失败：{exc}")

            break

    raise RuntimeError(
        "DeepSeek failed to produce a valid GenerationPlan "
        f"after {max_retries + 1} attempts.\n\n"
        f"Last error:\n{last_error}"
    )