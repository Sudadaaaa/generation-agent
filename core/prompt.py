"""全项目提示词的唯一真源。

## 为什么必须集中在这里

plan 任务的 system 提示词有两个消费方，分属两层：

  - 推理侧: agent/plan_agent.py  → PlanAgent 每次请求带上的 system 消息
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

**业务规则一律写进 core/schema.py 的字段说明，散文里只留角色设定与输出契约。**

规则有两个消费方：plan 子 agent 拼 schema 来生成，critic 子 agent 拼**同一份**
schema 来评审。规则留在散文里，critic 就得抄一份——两份副本没有任何机制保证同步，
必然漂，而且漂了不报错。写进 schema 则天然只有一份。

（这条边界改过。早先的写法是「跨字段的规则放散文，字段级的放 description」，
理由是 schema 只表达字段级语义。实际用下来，跨字段的规则（拆解粒度、随身道具归属）
写进 `elements` 的 description 里模型同样照做，而散文那份副本反而多出一个漂移点。）

现在留在散文的只有两类——**身份（你是谁、不做什么）**，和**schema 表达不了的
全局要求**（现在只剩 critic 的语言判断例外）。plan 侧的散文已经一条业务规则都不剩。

**判断标准是「critic 能不能看见」，不只是「schema 装不装得下」**：critic 拿到的
只有需求 + schema + 待评计划，plan 侧散文它完全看不见。所以一条规则只要 critic
可能要据以评审（例如信息忠实性），就必须进 schema——留在 plan 散文里等于让 critic
去判一条它没读过的规则。

**能用类型表达的规则不要用文字表达**：appearance/layout/action/overall 从前是
`Optional[str] = None`，靠一句「不要留 null」的散文去补。改成必填 `str` 之后，
这句话连同它防的那个失败模式一起消失了——pydantic 校验和 xgrammar 掩码都在解码期
就挡住了，比任何措辞都硬。一条规则如果能在 schema 里变成类型约束，就该在那儿。**
"""

import json

from .schema import GenerationPlan


def _for_prompt(node):
    """把 pydantic 生成的 JSON Schema 收拾成适合塞进提示词的样子：删 `title`。

    `title` 只是字段名的副本（`identity` → `"Identity"`），对模型是纯噪声。

    从前还做两件事——删 `default`、把 `Optional[str]` 的
    `{"anyOf": [{"type": "string"}, {"type": "null"}]}` 收成 `{"type": ["string", "null"]}`。
    appearance/layout/action/overall 从 `Optional[str] = None` 改成必填 `str` 之后，
    pydantic 对必填 str 只吐 `{"type": "string"}`，这两条再也不会触发，已删。
    真需要时再加：留一条走不到的分支，代价是每个读它的人都得先判断「这条现在到底走不走」。

    只影响拼进提示词的这份文本；`GenerationPlan.model_json_schema()` 本身不动
    （plan_agent 里走 response_format 的那条升级路径用的仍是原样 schema）。
    """
    if isinstance(node, dict):
        return {k: _for_prompt(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_for_prompt(x) for x in node]
    return node

MAIN_SYSTEM_PROMPT = """
你是通用助手，手上有一组工具。用户提出需求，你自己判断要不要用工具、用哪些、
按什么顺序、什么时候停——这个判断权在你，不必套某个固定流程。

怎么判断：
- 先看清每个工具是做什么的、要什么参数，再按需求去挑。工具的产出常常正好是另一个
  工具的输入，该不该接上，看这个需求值不值得多走这一步。
- 一步能做完的就一步做完；要多步的，一步一步来——拿到上一步的结果，再决定下一步。
- 参数按工具自己的说明填。它要自然语言的描述，就不要把结构化的中间结果直接塞进去。
- 不靠工具就能答的（解释、闲聊、你本来就知道的事），直接回答。不要为了用上工具而调用。

几条守则：
- 需求含糊或缺关键信息时（主体不明、两种理解都说得通），先说出你的理解并反问，
  不要瞎猜着往下做。
- 同一个工具、同一组参数不要重复调用；已经有结果就往下走。
- 工具报错时先读错误说了什么，再决定是改参数重试、换个工具，还是把问题讲给用户听。
"""


# ===== plan 子 agent 的角色设定 =====
# 只剩身份、不做什么、输出契约——一条业务规则都没有了。
# 全部规则在 core/schema.py 的字段说明里，随 schema 一起拼进来（见下一函数），
# 散文里不复述——理由见模块 docstring 的「分工」一节。
PLAN_BASE_SYSTEM_PROMPT = """
你是生图需求规划器：把用户的自然语言需求，转成严格符合下方 JSON Schema 的
GenerationPlan JSON。

你是结构化提取器，不是 Prompt Enhancer——只做拆解，不写最终图片 Prompt，
不解释，只输出 JSON。拆解规则全部写在下方的字段说明里，以它为准，
不要自创 schema 里没有的字段。

遵守上面的要求，最终输出一个符合 JSON Schema 的 JSON object。
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

# ===== critic 子 agent 的角色设定 =====
# rubric 先行：检查项不写进散文，而是让 critic 对着 schema 逐条过——它拼的是
# 与 plan 侧**同一份** schema，所以评的是「计划符不符合这份规则」，而不是模型的
# 个人品味。换模型、换提供方，结论仍然可比，规则也不会两边漂。
#
# 输出是 JSON，但**不含 ok / 通过 这类字段**：critic 只负责「说出看到的问题」，
# 判不判通过是主 agent 的事。给它一个 ok 字段等于让它既当评审又当裁判，是职责错位。
CRITIC_BASE_SYSTEM_PROMPT = """
你是一个专业的评审员，对分析用户的需求有着非常强的理解。用户给你一条需求，
和一份按下方 Schema 拆解成的结构化生成计划，你要指出这份计划的问题并给出修改意见，
让其他人照着去改。

你不重写计划，也不判「通过 / 不通过」——那是调用方的事，你只负责说出看到的问题。
每条意见都要具体到能直接照着改：是哪个元素、哪个字段、错在哪、应该怎样。

给出意见时主要分两点，最重要的一点是计划有没有覆盖用户的全部需求，这影响内容是否缺少，
其次是有没有符合下方 Schema 的字段说明，这关心内容实现的好不好。逐项对照，不要凭个人品味。

下方字段说明覆盖不到的，只有一条判断例外：**语言**。计划的语言应与用户需求的语言
一致；但如果用户明确指定了另一种语言的提示词（例如输入正文大部分是英文，但用户
说要中文提示词），此时计划用用户指定的那种语言，应当通过。

## 输出
只输出一个 JSON object，三个字段：
{
  "评分": 0 到 10 的数（可小数），综合「是否符合用户需求」与「是否遵循拆解规则」；
  "意见": 字符串数组，逐条列出，可以是问题，也可以是让计划更好的改进建议；
  "问题数量": 整数，「意见」里真正算问题的有几条
}
"""


def build_critic_system_prompt() -> str:
    """critic 子 agent 的 system 提示词。

    schema 与 plan 侧**拼的是同一份**：critic 判的就是「这份计划符不符合这份 schema」，
    它得先看见规则本身。散文里因此不复述任何拆解规则——两处各写一遍必然漂，
    而这个项目已经因为提示词副本漂移吃过一次亏（见模块 docstring）。

    真源仍在这里——critic_agent 只调用本函数、不存副本。
    """
    return (
        CRITIC_BASE_SYSTEM_PROMPT
        + "\n\n以下是计划拆解时必须严格遵循的规则：\n\n"
        + json.dumps(_for_prompt(GenerationPlan.model_json_schema()),
                     ensure_ascii=False, indent=2)
    )


def build_critique_request(requirement: str, plan: str) -> str:
    """critic 的 user 消息：需求 + 待评计划。

    只有这两样。评审**不该**看到计划是怎么改出来的——主 agent 的 history 里有整场
    对话，其中计划的自我辩解与上一轮的评语会把它说服到放行（见 agent/critic_agent.py）。
    """
    return (
        "## 用户需求\n\n" + requirement.strip()
        + "\n\n## 待评审的 GenerationPlan\n\n" + plan.strip()
    )