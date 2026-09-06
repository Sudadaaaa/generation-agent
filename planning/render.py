"""Plan → 最终提示词：把 GenerationPlan 交给 LLM 渲染成自然语言提示词。

与 schema.to_prompt_text（确定性拼接）不同，这里让大模型按渲染规则
（语言随 plan、每元素 位置→称呼→外表→动作、同一主体多元素归并等）
自由组织措辞，输出可直接投喂生图模型的文本。
"""

from openai import OpenAIError

from planning.llm import PlanLLM
from planning.prompts import RENDER_SYSTEM_PROMPT
from planning.schema import GenerationPlan


def build_final_prompt(
    plan: GenerationPlan,
    llm: PlanLLM,
    user_input: str | None = None,
) -> str:
    """把 GenerationPlan 渲染成最终生图提示词。

    llm 渲染需要自由文本输出（调用 complete(json_mode=False)）；
    传入原始 user_input 可帮助模型把握语言与措辞。
    渲染失败/返回空内容时回退到 plan.to_prompt_text()（确定性兜底），
    打印 [render] 警告，不阻塞出图。
    """

    content_parts = ["请根据下面的 GenerationPlan 生成最终提示词：\n"]

    if user_input:
        content_parts.insert(0, f"用户的图片需求：\n{user_input}\n\n")

    content_parts.append(plan.model_dump_json(indent=2))

    messages = [
        {"role": "system", "content": RENDER_SYSTEM_PROMPT},
        {"role": "user", "content": "".join(content_parts)},
    ]

    try:
        text = llm.complete(messages, json_mode=False)
    except (RuntimeError, OpenAIError) as exc:
        print(f"[render] LLM 渲染失败，回退到确定性拼接：{exc}")
        return plan.to_prompt_text()

    text = (text or "").strip()

    if not text:
        print("[render] LLM 渲染返回空内容，回退到确定性拼接。")
        return plan.to_prompt_text()

    return text
