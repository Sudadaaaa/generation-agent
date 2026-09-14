"""plan 子 agent：自然语言需求 → 结构化 GenerationPlan。

组合进 tools/plan.py 的 PlanTool，与 GenerateImageTool 持有 ImageGenerator 同构：
工具只管参数与回执，拆解这件事整个在这里。

⚠️ SYSTEM_PROMPT 不是现写的，是逐字取自 git show 826d8a4:planning/prompts.py。
   它和 sft/ 那 3088 条训练语料的系统提示词是同一份，sft/build_sft.py 明确写着
   「与 generation-agent 推理逐字一致」，sft/out/report.txt 还记着它的指纹：

       旧（826d8a4 的 planning/schema.py）  sha1(build_plan_system_prompt())[:8] = 510f9d91
       现（core/schema.py，类 docstring 改过措辞）                    = 3b8c905a

   两者只差 GenerationPlan 的类 docstring——pydantic 会把它嵌成 schema 顶层的
   description，所以改一句注释就等于换了提示词。改本文件的 SYSTEM_PROMPT、
   或改 core/schema.py 里任何字段说明/类 docstring，sft/out/ 的语料即过期。
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
from core.schema import GenerationPlan

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
2. 独立 vs 并入：对画面中每个对象，先判断「用户有没有把它当作主体」。
   - 用户给了独立描述与关注的对象 → 独立成元素。
   - 依附于主体的随身/互动道具（拿在手里、背在身上、作为配饰的伞、剑、书、包、
     乐器、杯子、帽子），以及仅被一笔带过、简单提及、非主要对象 →
     不独立成元素；随身道具并入持有主体的 action（含道具外观）。
3. 多个同类对象：被当作独立主体分别描述的对象，即使同类也各自成元素——
   单个元素只能承载一种外观。同一类别存在多个需区分的独立主体时，identity 用
   类别+拉丁字母编号：如「两个小女孩，一个穿红衣一个穿蓝衣」→ identity='小女孩A'
   与 identity='小女孩B' 两个元素，红衣/蓝衣写进各自 appearance。编号用大写拉丁
   字母 A/B/C（语言中立，不用 1/2/3 数字以免与整体数量混淆），只区分"谁"、不绑定
   外观，顺序可随需求提及先后，一旦分配就全图逐字稳定。
   只有被当作一个整体呈现、未逐个区分的群体（如「并肩的两只金毛犬」）才合并为
   一个元素，数量写进 identity——那是作为一个整体的一个元素，与给多个个体编号不同。
4. 同一主体在画面中多处出现（如多宫格同一只猫、同一人物在不同格）：
   把它拆成多个元素（每个出现位置一个），并且这些元素的 identity 必须逐字完全
   相同——渲染正靠 identity 一致把它们合并为同一主体。例如九宫格同一只橘猫
   睡觉/打哈欠/偷零食，九个元素的 identity 都必须是 '橘猫'，动作分别写进各自 action。
   appearance 不要求相同：同一主体在不同位置可以穿不同衣服、换不同发型等，
   按各元素分别写该次出现的外观即可（若各次外观恰好一致，保持逐字相同）。

# 字段

5. 每个元素统一用同一套字段：identity、appearance、layout、action。
   各字段的语义、示例与填写边界以文末 JSON Schema 的字段描述为准，
   不要自创字段。所有字段都是字符串或 null，不是数组；
   count 已并入 identity，没有单独的 count 字段。
6. identity 是称呼，同时是跨元素引用令牌：某个元素的 action 提到画面中另一个
   主体时，必须逐字使用那个主体的 identity（含编号，如 '牛仔A'）。例如另一元素
   identity='两只金毛犬'，本元素的 action 就写'弯腰抱着两只金毛犬'，不要用别的
   说法重述对方外观。identity 一旦确定，全图保持一致，不要在同义词之间切换。
   同一类别多个独立主体用拉丁字母编号区分（规则 3），引用时带上编号。
   identity 只写 类别(+整体数量+字母编号)：不要混入外观（写 appearance）、动作
   （写 action）。禁止用相对位置/景别/所在格区分主体——'左侧牛仔' 是错的，他在
   别的格可能站在中央，位置信息请写进 layout。

# 补全与约束

7. 可选字段（appearance/layout/action）不要留 null：按画面场景与元素类型，
   用最合理、最常见的默认值自动补齐，且不得与用户需求冲突。
   例如：缺省 layout='画面中央，占据主体'、action='静止'（或该类元素最常见状态）。
8. overall 只写画面级（全局）属性（风格/构图/光线/天气/氛围/画幅/特效/画质/
   整体排版布局框架），只在需要时写一次；多宫格/分镜的整体框架
   （等大画框、留白边框、几行几列）只写进 overall；
   元素的细节（外观/动作/位置/服饰/表情等）只写进对应 element，overall 不重复。
9. 元素上出现的文字（招牌、书名、标语）保持用户原文逐字保留。
10. 用户明确提供的信息必须尽可能完整保留；自动补齐不得与用户需求冲突；
    不要编造画面外的新主体。

最终只能输出符合 JSON Schema 的 JSON object。
"""


def build_plan_system_prompt() -> str:
    """SYSTEM_PROMPT + 这份 schema。

    schema 必须拼进提示词：pydantic 的 Field description 不会自己发给模型，
    不拼进来，core/schema.py 里那几百字字段说明对模型等于不存在。
    """
    return (
        SYSTEM_PROMPT
        + "\n\n以下是必须严格遵循的 JSON Schema：\n\n"
        + json.dumps(GenerationPlan.model_json_schema(), ensure_ascii=False, indent=2)
    )


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


# 只要求模型「输出的是 JSON」，不要求「合 GenerationPlan 的 schema」——schema 那层由
# 下面的 parse + 修正循环兜。这个字段在两家上是同一个写法，不需要按 provider 翻译：
#
#   DeepSeek：只有 json_object（保证语法合法的 JSON，不保证合 schema）。
#             json_schema 会 HTTP 400 "This response_format type is unavailable now"；
#             guided_json 是 vLLM 的私有扩展，DeepSeek 根本没有这个字段。
#   vLLM：    OpenAI 兼容服务【原生支持 response_format】，收到 json_schema 会自己
#             转成约束解码（xgrammar 等）。所以以后换 vLLM 只需改 .env 的
#             PLAN_BASE_URL / PLAN_MODEL_ID，这里一行不动。
#
# 想升级成 schema 级约束时，只动这一个常量：
#     {"type": "json_schema",
#      "json_schema": {"name": "GenerationPlan",
#                      "schema": GenerationPlan.model_json_schema()}}
# vLLM 上直接生效；DeepSeek 上会 400，所以现在不用。
#
# 另外，json_object 模式要求提示词里出现 "json" 这个词（否则模型可能一直吐空白直到
# 撞 max_tokens）——build_plan_system_prompt() 里既有「只能输出 JSON」又有整份
# JSON Schema，这个前提已经满足。
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
    ) -> None:
        super().__init__(name, llm, system_prompt or build_plan_system_prompt(), config)
        self.max_repairs = max_repairs

    def run(self, requirement: str) -> str:
        self.add_message(Message("user", requirement))

        last: Exception | None = None
        for _ in range(self.max_repairs + 1):
            content = self._complete()
            self.add_message(Message("assistant", content))
            try:
                plan = parse_generation_plan(content)
            except PlanFormatError as e:
                last = e
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
        response = self.llm.invoke(self.get_messages(), response_format=RESPONSE_FORMAT)
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

    用 `python -m agent.planagent`（在仓库根跑）或 `python agent/planagent.py`。
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
