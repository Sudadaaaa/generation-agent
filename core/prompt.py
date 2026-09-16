"""全项目提示词的唯一真源。

## 为什么必须集中在这里

plan 任务的 system 提示词有两个消费方，分属两层：

  - 推理侧: agent/planagent.py  → PlanAgent 每次请求带上的 system 消息
  - 训练侧: mydatasets/plan_sft.py → 织进 SFT 语料的那条 system 消息

两边必须逐字相同，否则模型学到的和线上看到的是两套东西（train/serve 偏斜）。
从前两边各存一份：推理侧现拼，训练侧冻结成一个字符串字面量，靠注释提醒同步。

**这个做法已经失效过一次**：core/schema.py 里 GenerationPlan 的类 docstring 被改过
措辞，而 pydantic 会把它嵌成 schema 顶层的 description，于是训练语料用的提示词
（sha1 510f9d91）与推理用的（3b8c905a）从此不一致——没有任何机制会报错，
是后来比对哈希才发现的。

现在两边都 import 本模块，副本在物理上不存在了。

## 只放文本，不放逻辑

本模块只依赖 core.schema（拼提示词要用 pydantic 生成的 JSON Schema）。
**刻意不 import core.llm / core.config / core.agent**：训练侧的
mydatasets/plan_sft.py 也要 import 这里，不能让一个数据加载器被拖进
openai 客户端、环境变量读取那一整套 agent 依赖。

## 散文与 schema 的分工（改提示词前必读）

两边**不重复**写同一个意思：同一个规则写两遍，模型会花注意力去对齐两处措辞，
还可能各听一半。分工是：

  散文 → 只写 **JSON Schema 表达不了的决策**
         拆解粒度（几条算一个元素）、什么并进什么（随身道具归属）、
         缺省值怎么补、信息忠实性
  schema → 字段级的语义与边界（identity 怎么写、action 与 appearance 的分界）

判断一条规则该放哪边，就看它是否跨字段：跨字段的放散文，单字段的放 description。
"""

import json

from .schema import GenerationPlan


def _for_prompt(node):
    """把 pydantic 生成的 JSON Schema 收拾成适合塞进提示词的样子。三件事：

    1. 删 `title`：只是字段名的副本（`identity` → `"Identity"`），对模型是纯噪声。
    2. 删 `default`：可选字段带着 `"default": null`，与散文规则「可选字段不要
       留 null」是**相反**的信号，两个都发过去模型会犹豫。去掉它，
       「必须补齐」就只剩一种说法。
    3. 把 `Optional[str]` 生成的
           {"anyOf": [{"type": "string"}, {"type": "null"}]}
       收成 {"type": ["string", "null"]}——JSON Schema 两种写法等价，但前者 5 行、
       后者 1 行，appearance/layout/action 三个字段合计省约 290 字符。

    只影响拼进提示词的这份文本；`GenerationPlan.model_json_schema()` 本身不动
    （planagent 里走 response_format 的那条升级路径用的仍是原样 schema）。

    实测：3761 字符 → 2619。
    """
    if isinstance(node, dict):
        out = {k: _for_prompt(v) for k, v in node.items() if k not in ("title", "default")}

        # 收 Optional：只认「正好两个分支、每个只有 type、且其中一个是 null」这一种形状，
        # 不满足就原样留着——宁可少省几个字符，也不去猜别的 anyOf 能不能合并。
        variants = out.get("anyOf")
        if (isinstance(variants, list)
                and len(variants) == 2
                and all(isinstance(v, dict) and set(v) == {"type"} for v in variants)
                and "null" in (v["type"] for v in variants)):
            out.pop("anyOf")
            out["type"] = [v["type"] for v in variants]

        return out
    if isinstance(node, list):
        return [_for_prompt(x) for x in node]
    return node


# ===== plan 子 agent 的散文规则 =====
# 只写 schema 表达不了的跨字段决策，理由见模块 docstring 的「分工」一节。
PLAN_BASE_SYSTEM_PROMPT = """
你是生图需求规划器：把用户的自然语言需求，转成严格符合下方 JSON Schema 的
GenerationPlan JSON。

你是结构化提取器，不是 Prompt Enhancer——只做拆解，不写最终图片 Prompt，
不解释，只输出 JSON。

你需要分析用户的需求，然后拆解成elements：主要人物、核心物体、前景关键对象、
需要独立描述的自然主体（太阳、山、建筑）。不要为每个名词都拆一个元素——
背景、天气、氛围、环境特效一律写进 overall。元素个数由画面内容决定，不设上限。
依附于主体的随身/互动道具（拿在手里、背在身上、作为配饰的伞、剑、书、包、乐器、
杯子、帽子）与仅一笔带过的非主要对象，都不独立成元素，
并入持有主体的 action（含道具外观）。

要注意的是只用 Schema 里列出的字段，不要自创一个新的字段。
可选字段（appearance/layout/action）不要留 null：
按画面场景与元素类型，用最合理、最常见的默认值补齐，且不得与用户需求冲突。
例如需求是特写，没写在哪，常见情况下layout='画面中央，占据主体'、
动作如果没描述，可以写action='静止'（思考此时最常见的状态）。
用户明确给出的信息尽量完整保留；不要编造画面外的新主体。

遵守上面的要求，最终输出一个符合 JSON Schema 的 JSON object。
"""

MAIN_SYSTEM_PROMPT = """
你是图片生成助手。用户给你图片需求，你负责把它变成一张真实的图片。

怎么做：
1. 自己把用户的需求写成一段高质量的中文提示词——主体、外观、动作、构图、光线、风格
   都要具体，不要照抄用户原话。
2. 调用 generate_image，把这段提示词传给它（它会真实出图并返回保存路径）。
3. 拿到路径后，用简短中文向用户汇报：你写的提示词 + 图片路径。

需求含糊时（缺主体、缺风格、两种理解都说得通），先把你的理解讲给用户听并反问，
不要瞎猜着出图。一次只推进必要的一步，不要重复调用同一个工具。
"""


def build_plan_system_prompt() -> str:
    """散文规则 + 这份 schema —— plan 子 agent 完整的 system 提示词。

    schema 必须拼进提示词：pydantic 的 Field description 不会自己发给模型，
    不拼进来，core/schema.py 里那几百字字段说明对模型等于不存在。
    """
    return (
        PLAN_BASE_SYSTEM_PROMPT
        + "\n\n以下是必须严格遵循的 JSON Schema：\n\n"
        + json.dumps(_for_prompt(GenerationPlan.model_json_schema()),
                     ensure_ascii=False, indent=2)
    )

def build_correction_message(error: Exception) -> str:
    """把格式错误变成一条要求重发的消息。

    只管格式，不管内容——计划拆得好不好、理解对不对是 review 的事，
    这里只保证拿到的是一份能解析、合 schema 的 GenerationPlan。
    """
    return f"""
你上一次返回的内容不合法，请修正后重新输出。

错误原因：

{error}

要求：
- 只改上面错误原因指出的字段，其余内容逐字保持不变。
- 重新输出【完整】的 JSON object，不要只输出改动的那一小段。
- 只能返回 JSON，不要输出解释，不要 Markdown 代码围栏。
"""
