from pydantic import BaseModel, Field


_PUNCT = "，。；！？、：…"


def _normalize_clause(s: str) -> str:
    """去掉片段首尾空白与末尾标点，保证拼接时符号间隔正确。"""

    return s.strip().rstrip(_PUNCT)


class Element(BaseModel):
    """画面中的一个基本元素：一次出现、一个需独立控制的身份。

    identity 给称呼，appearance/layout/action 分别描述外表、位置、动作。
    元素间互动写进 action，提及他主体时逐字用其 identity。

    四个字段都必填。需求没写明的，按画面场景与元素类型用最合理、最常见的默认值补齐，
    不要留空——留空等于这一维的信息整个丢掉。例如需求是特写又没写在哪，layout 常见为
    '画面中央，占据主体'；没写动作，action 可为 '静止'（思考此时最常见的状态）。
    """

    identity: str = Field(
        description=(
            "这个元素的称呼（身份），兼作全图唯一的引用令牌：别的元素与它相关时，"
            "就可以使用本称呼代指它，而不再重述其外观，动作等。"
            "如果和其他元素的身份重复，可以用大写拉丁字母编号区分（'小女孩A'/'小女孩B'，"
            "不用 1/2/3 以免与整体数量混淆）；编号只负责分清谁是谁、不绑定外观，"
            "不要使用位置来称呼identity，例如'左侧牛仔'是错的，如果多次出现他可能在右侧，"
            "identity 只写主体本身：外观进 appearance、动作进 action、位置进 layout；"
        )
    )

    appearance: str = Field(
        description=(
            "该元素在本次出现中的外表细节：形状、颜色、材质、纹理、服饰、发型发色、"
            "五官、体型、神态表情、装饰、印刻文字（保持原文逐字）等；"
            "不含动作、位置与关系。按元素类型取舍（人物重服饰神态，物体重形状材质，"
            "自然元素重形态光照）。同一主体多处出现时，这里只写该次出现的外观，"
            "如果外观相同，可以直接说和之前出现的外观一样，也允许各次不同（不同格穿不同衣服、换发型）。"
        )
    )

    layout: str = Field(
        description=(
            "该元素在画面中的位置信息、景深层次与相对大小等，"
            "如 '在画面左侧'、'画面中间的远处'、'九宫格左上第一格'。"
            "如果元素本身位置与其他元素或者场景中的物体位置有关，可以使用相对位置，"
            "如 '在女孩的肩上，在桌子上（尽管桌子可能不是一个独立元素）'。"
        )
    )

    action: str = Field(
        description=(
            "该元素的行为、姿态、状态与互动：自身动作（奔跑、跳跃、打盹、发光）；"
            "与其他元素或者场景的交互（抱着两只金毛犬，摘树叶，追蝴蝶，拿手机）；"
            "动作引发的效果（吐寒气，溅起水花，喷出火焰等）。"
            "与画面中其他物体互动时，如果它被拆成了一个元素，要逐字引用对方的 identity，"
            "如 '抱着金发女孩'、'托着一段文字'，要称呼其identity。"
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
    一张图片的完整结构化提示词：画面拆成多个元素（每个元素是一次出现，
    用 identity/appearance/layout/action 描述），overall 补充画面级属性。

    这份结构不直接出图——它由下游 LLM 渲染成最终提示词；
    to_prompt_text 是渲染失败时的确定性兜底。
    """

    elements: list[Element] = Field(
        min_length=1,
        description=(
            "画面中的主要元素列表，至少一个。"
            "只拆用户需求里出现过的主体，不要编造需求中没有的新主体；"
            "用户明确给出的信息尽量完整保留。"
            "基本拆解规则为用户需求中有特别描述的主要人物、核心物体、前景关键对象"
            "如 '一只橘猫'、'金发女孩'、'几朵鲜花'、'牛仔A/B'、'一段中文'。"
            "如果出现整体描述，例如用户说有一群五彩斑斓的牛，几只小狗，"
            "不用分别拆解，直接当作一个整体元素，如果用户进行了更详细的描述，"
            "如几只小狗，一只黄色的在笑，一只在翻滚等，则需要拆成几个元素。"
            "注意，不要每个名词都拆一个元素，依附于主体的随身/互动道具"
            "（拿在手里、背在身上、作为配饰的伞、包、杯子、帽子），"
            "与仅一笔带过的非主要对象，都不独立成元素，可以并入外观或者动作中，具体两者的关系是否因动作而发生。"
            "如'踢足球'、'追蝴蝶'等跟动作相关，'戴着帽子'、'背着书包'可以看作主体的外观。"
            "同一主体在画面中多处出现（如多宫格/多视角图）时拆成多个元素，"
            "且这些元素的 identity 必须逐字完全相同——渲染正是靠它把多次出现合并为同一主体。"
        ),
    )

    overall: str = Field(
        description=(
            "需求中与主体无关的一些画面级别的整体描述，统一都写在这里。"
            "例如指定整体风格（写实摄影、2D动画、水彩、赛博朋克等）、"
            "整体的构图与镜头（景别、机位角度、透视、景深）、"
            "天气与环境、色调氛围、多宫格/分镜布局、不应出现的内容等等"
            "需求里提到什么就写什么，需求没提的不要为了凑类别硬写。"
            "本字段的语言与用户需求的语言一致（需求是英文就写英文）。"
            "反过来，属于某个元素的都不进这里，只写进对应 element"
        ),
    )

    def to_prompt_text(self) -> str:
        """把结构化计划拼成一段提示词文本（确定性兜底）。

        兜底路径：正常路径是由大模型渲染出更自然的最终提示词，这里只在那个
        渲染失败时用。每元素按 布局→称呼→外表→动作 拼接，overall 附在末尾；
        不输出字段名、编号等结构标记。
        """

        segments: list[str] = [el.to_prompt_text() for el in self.elements]

        if self.overall:
            segments.append(f"{_normalize_clause(self.overall)}。")

        return "".join(segments)
