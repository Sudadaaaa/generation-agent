from typing import Optional

from pydantic import BaseModel, Field


_PUNCT = "，。；！？、：…"


def _normalize_clause(s: str) -> str:
    """去掉片段首尾空白与末尾标点，保证拼接时符号间隔正确。"""

    return s.strip().rstrip(_PUNCT)


class Element(BaseModel):
    """画面中的一个基本元素：一次出现、一个需独立控制的身份。

    四个维度描述它：identity（称呼/身份，兼作全图引用令牌）、
    appearance（外表）、layout（布局/位置）、action（动作/互动）。
    元素间互动写进 action，提及他主体时逐字用其 identity。
    """

    identity: str = Field(
        description=(
            "该元素的称呼（身份），同时是全图唯一的引用令牌：别的元素要与它互动时，"
            "在 action 里逐字使用本称呼代指它，不再重述其外观。"
            "写法：可直接嵌进中文句子的自然名词短语，结构为 类别(+整体数量)(+字母编号)，"
            "如 '一只橘猫'、'金发女孩'、'并肩的两只金毛犬'、'牛仔A'。"
            "整体数量只写在被当作一个整体呈现、未逐个区分的群体上（'并肩的两只"
            "金毛犬'），此时整个群体算一个元素；被当作独立主体分别描述的对象各成元素"
            "（单个元素只能承载一种外观）。同一类别存在多个需区分的独立主体时，"
            "identity 用 类别+拉丁字母编号：'小女孩A'/'小女孩B'、'牛仔A'/'牛仔B'。"
            "编号用大写拉丁字母 A/B/C（语言中立），不用 1/2/3 数字以免与整体数量混淆；"
            "它只负责分清谁是谁、不绑定外观，顺序可随用户需求中提及的先后，一旦分配"
            "就全图逐字不变。个体之间的外观差异（红衣/蓝衣、帽色等）写进各自 "
            "appearance，不写进 identity。"
            "禁止用相对位置、景别或所在格来区分主体（'左侧牛仔' 是错的——他在另一格"
            "可能站在中央）：位置是 layout 的职责，不是身份。"
            "称呼保持简短、全图稳定。同一主体在画面中多处出现（如多宫格同一只猫、"
            "同一人物在不同格）时，把它拆成多个元素，且这些元素的 identity 必须逐字"
            "完全相同（都写 '橘猫'/'牛仔A'）——渲染正是靠 identity 一致把它们合并为"
            "同一主体。appearance 不必相同：同一主体在不同位置可以穿不同衣服、换不同"
            "发型等，按各元素分别写。identity 只写主体本身（类别+整体数量+字母编号），"
            "动作写进 action、位置写进 layout、每次出现的外表写进 appearance，"
            "都不要混进 identity。"
        )
    )

    appearance: Optional[str] = Field(
        default=None,
        description=(
            "该元素在本次出现中的外表细节：形状、颜色、材质、纹理、服饰、发型发色、"
            "五官、体型、神态表情、装饰、印刻文字（保持原文逐字）等。"
            "按元素类型覆盖（举例，不限于）："
            "人物→发型发色、五官、体型、服饰、神态表情；"
            "物体→形状、颜色、材质、纹理、新旧破损、印字；"
            "自然元素→形态、颜色、光照感。"
            "不含动作、位置与关系。同一主体多处出现时，本 appearance 只写该次出现的"
            "外观，允许与其它出现不同（例如不同格穿不同衣服、换不同发型）；"
            "若与其它出现恰好相同，保持逐字一致更便于渲染归并。"
        )
    )

    layout: Optional[str] = Field(
        default=None,
        description=(
            "该元素在画面中的位置、景深层次与相对大小；多宫格/分镜时写所在格。"
            "例如：'画面中央，占据主体'、'右侧草地上'、'九宫格左上第一格'。"
            "多宫格/分镜的共有框架（行数列数、等大画框、留白边框）写进 overall，"
            "不写在这里。"
        )
    )

    action: Optional[str] = Field(
        default=None,
        description=(
            "该元素的行为、姿态、状态与互动。内容包括："
            "自身动作与状态（奔跑、跳跃、打盹、站立、发光、静止）；"
            "带宾语的互动/关系（弯腰抱着'两只金毛犬'、站在草坪上、面对太阳）；"
            "随身/互动道具的持有方式（右手举着一把透明雨伞，伞面撑开）；"
            "动作引发的瞬态效果（跳跃溅起的水花、奔跑扬起的尘土）。"
            "行为主语就是该元素本身，不要重述称呼（写'正在酣睡'，不写'橘猫正在酣睡'）。"
            "与画面中其他主体的互动，必须逐字引用对方的 identity"
            "（写'弯腰抱着两只金毛犬'，其中'两只金毛犬'是另一元素的 identity），"
            "不要用别的说法重述对方。"
            "某次出现无明确行为时写'静止'或该类元素最常见的状态。"
        )
    )

    def to_prompt_text(self) -> str:
        """兜底渲染该元素为一句提示词：位置→称呼→外表→动作。"""

        parts: list[str] = []

        if self.layout:
            parts.append(f"在{_normalize_clause(self.layout)}")

        head = self.identity

        if self.appearance:
            head = f"{head}，{_normalize_clause(self.appearance)}"

        parts.append(head)

        if self.action:
            parts.append(_normalize_clause(self.action))

        return "，".join(parts) + "。"


class GenerationPlan(BaseModel):
    """
    一张图片的完整结构化提示词。

    把画面拆解为多个基本元素，每个元素是一次出现，用一套通用字段
    （identity/appearance/layout/action）描述；overall 用一段字符串描述
    画面级（全局）属性。最终提示词由 planning.render 交给 LLM 渲染，
    to_prompt_text 仅在 LLM 失败时作确定性兜底。
    """

    elements: list[Element] = Field(
        min_length=1,
        description="画面中的基本元素列表，至少一个。",
    )

    overall: Optional[str] = Field(
        default=None,
        description=(
            "画面整体（全局）属性，用一段简洁的中文描述，只写画面级属性，按需覆盖："
            "①风格（写实摄影、电影感Cinematic、2D动画、3D CG、水彩、水墨、油画、"
            "赛博朋克、复古胶片）；"
            "②构图与镜头（景别：特写/近景/中景/全景/远景；"
            "机位角度：平视/仰视/俯视/顶视；透视：广角/长焦/鱼眼/单点透视；"
            "景深：浅景深/背景虚化）；"
            "③光线（时间、光源、方向，如：黄昏暖光、顶光、侧逆光）；"
            "④天气与环境（如：雨后、微风吹拂、雾气、夜色）；"
            "⑤色调与氛围；⑥画幅比（如：横构图、竖构图、方形构图）；"
            "⑦镜头/视觉特效（如：光斑、倒影、颗粒感、长曝光）；"
            "⑧画质与细节；⑨不应出现的内容；"
            "⑩整体排版/布局框架（多宫格、分镜、拼贴：如 3x3 九宫格、等大方形画框、"
            "格子间留白边框），只写一次。"
            "注意：元素的细节（外观/动作/位置/服饰/表情等）只写在对应 element 中，"
            "overall 不重复。"
        ),
    )

    def to_prompt_text(self) -> str:
        """把结构化计划拼成一段提示词文本（确定性兜底）。

        仅供 LLM 渲染失败时兜底；正常路径用 planning.render.build_final_prompt，
        由大模型生成更自然的最终提示词。每元素按 布局→称呼→外表→动作 拼接，
        overall 附在末尾；不输出字段名、编号等结构标记。
        """

        segments: list[str] = [el.to_prompt_text() for el in self.elements]

        if self.overall:
            segments.append(f"{_normalize_clause(self.overall)}。")

        return "".join(segments)
