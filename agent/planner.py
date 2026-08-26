import json
import os
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from schemas.generation_plan import GenerationPlan

SYSTEM_PROMPT = """
你是一个 Generative AI Generation Planner。

你的任务是将用户的自然语言生成需求转换为严格的
GenerationPlan JSON。

你不是 Prompt Enhancer。

不要生成最终图片 Prompt。
不要生成最终视频 Prompt。
不要输出解释。
只能输出 JSON。

规则：

1. 必须严格遵循提供的 JSON Schema。
2. 不要增加 Schema 中不存在的字段。
3. 不要删除必要字段。
4. object 类型字段必须输出 object，不能压缩成字符串。
5. 如果字段没有相关信息：
   - Optional object 可以使用 null。
   - object 内部字段可以使用 null。
6. 用户明确提供的信息必须尽可能完整地保留。
7. 用户要求出现的文字必须保持原文。
8. 不要凭空添加与用户需求冲突的信息。

任务分类：

text_to_image:
用户要求从文字生成图片。

image_to_image:
用户要求修改、重绘、转换已有图片。

text_to_video:
用户要求从文字生成视频。

image_to_video:
用户提供已有图片，并要求让图片动起来或生成视频。

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

            if attempt >= max_retries:
                break

            correction_message = f"""
你上一次返回的 JSON 不符合 GenerationPlan Schema。

验证错误如下：

{exc}

请修正 JSON。

特别注意：

- subject 必须是 object
- environment 必须是 object
- camera 必须是 object 或 null
- composition 必须是 object 或 null
- style 必须是 object 或 null
- text 必须是 object 或 null
- generation 必须存在
- generation 必须包含 model 和 duration
- 不要把 object 类型字段写成字符串
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
            break

    raise RuntimeError(
        "DeepSeek failed to produce a valid GenerationPlan "
        f"after {max_retries + 1} attempts.\n\n"
        f"Last error:\n{last_error}"
    )