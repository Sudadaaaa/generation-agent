from typing import Optional

from pydantic import BaseModel, Field, model_validator


_PUNCT = "，。；！？、：…"


def _normalize_clause(s: str) -> str:
    """去掉片段首尾空白与末尾标点，保证拼接时符号间隔正确。"""

    return s.strip().rstrip(_PUNCT)


class Element(BaseModel):
    """画面中的一个基本元素。一个元素 = 一个主体；layout/action 是它的出现列表。"""

    name: str = Field(
        description=(
            "元素是什么（类别或身份），只写主体身份本身，例如：橘猫、女孩、木船、草坪、太阳。"
            "不要混入框架词或场景标题（如'一格橘猫睡觉的画面'）——"
            "'哪一格/哪个位置'写进 layout，'在做什么'写进 action。"
        )
    )

    count: Optional[str] = Field(
        default=None,
        description="数量，例如：一个、两只、三朵、一片、成群。",
    )

    appearance: Optional[str] = Field(
        default=None,
        description=(
            "该元素在所有出现位置一致的固定外观：形状、颜色、材质、纹理、服饰、"
            "发型发色、五官、体型、神态表情、装饰、印刻文字（保持原文逐字）。"
            "按元素类型覆盖（举例，不限于）："
            "人物→发型发色、五官、体型、肤色、服饰、神态表情；"
            "物体→形状、颜色、材质、纹理、新旧破损、印字；"
            "自然元素→形态、颜色、光照感。"
            "例如：金发白裙、翠绿、橙色发光、漆皮剥落、红底白字'营业中'。"
            "不含动作与关系。即使元素多次出现也只写一次固定外观；"
            "每次出现特有的事物写进对应的 action 项。"
        ),
    )

    layout: Optional[list[str]] = Field(
        default=None,
        description=(
            "该元素每次出现的位置、景深层次与相对大小，列表的每一项是一次出现，"
            "与 action 逐项对齐（第 i 项位置对应第 i 项行为）。"
            "只出现一次时是单元素列表，如 ['画面中央，占据主体']；"
            "多宫格/分镜时每格一项，如 ['九宫格左上角第一格', '九宫格正上方中间第二格']。"
            "多宫格时只写所在格子；格子大小、形状、间距、边框等共有框架信息写进 overall。"
        ),
    )

    action: Optional[list[str]] = Field(
        default=None,
        description=(
            "该元素每次出现的行为、姿态、状态与互动关系，列表与 layout 逐项对齐"
            "（第 i 项对应第 i 项出现）。每一项包括："
            "自身动作与状态（奔跑、跳跃、挥手、发光、飘动、静止、站立）、"
            "带宾语的互动/关系（举着一把透明雨伞、站在草坪上、面对太阳、"
            "凝视镜头、追着前面的蝴蝶）、"
            "随身/互动道具的外观与持有方式（右手举着一把透明雨伞，伞面撑开）、"
            "动作引发的瞬态效果（跳跃溅起的水花、奔跑扬起的尘土）、"
            "该次出现特有的事物（身下的垫子、桌边的鱼、窗外的鸟、树干的枝叶）。"
            "行为主语就是该元素本身，不要在项里重述名称（写'正在酣睡'，不写'橘猫正在酣睡'）。"
            "某次出现无明确行为时该项填'静止'或该类元素最常见的状态。"
        ),
    )

    @model_validator(mode="after")
    def _check_occurrence_alignment(self) -> "Element":
        """layout 与 action 都填写时长度必须一致，逐项对齐。"""

        n_layout = len(self.layout) if self.layout else 0
        n_action = len(self.action) if self.action else 0

        if n_layout and n_action and n_layout != n_action:
            raise ValueError(
                f"layout 与 action 列表长度不一致（{n_layout} vs {n_action}）："
                "两者必须逐项对齐，第 i 项位置对应第 i 项行为。"
            )

        return self

    def to_prompt_text_line(self) -> str:
        """
        把该元素渲染成一句提示词：数量+名字+外表，随后是布局与动作。

        只出现一次：数量+名字+外表+布局+动作。
        多次出现：先拼数量+名字+外表，再标注「共出现N次」，
        然后逐次给出第N次出现的布局与行为。
        """

        # 头部：数量+名字+外表（外表只写一次）
        head = f"{self.count}{self.name}" if self.count else self.name

        if self.appearance:
            head = f"{head}，{_normalize_clause(self.appearance)}"

        layouts = self.layout or []
        actions = self.action or []

        if not layouts and not actions:
            return head + "。"

        n = max(len(layouts), len(actions), 1)

        if n == 1:
            # 单次出现：头部直接接布局与行为
            parts = [head]

            if layouts:
                parts.append(layouts[0])

            if actions:
                parts.append(actions[0])

            return "，".join(_normalize_clause(p) for p in parts) + "。"

        # 多次出现：标注次数，逐次给出「第N次+布局+行为」
        items: list[str] = []

        for i in range(n):
            clause = f"第{i + 1}次"

            if i < len(layouts):
                clause += f"，{_normalize_clause(layouts[i])}"

            if i < len(actions):
                clause += f"，{_normalize_clause(actions[i])}"

            items.append(clause)

        return f"{head}，共出现{n}次：" + "；".join(items) + "。"

    def to_prompt_text(self) -> str:
        """兼容：该元素渲染成一句提示词。"""

        return self.to_prompt_text_line()


class GenerationPlan(BaseModel):
    """
    一张图片的完整结构化提示词。

    把画面拆解为多个基本元素，每个元素用一套通用字段描述；
    overall 用一段字符串描述画面级（全局）属性。
    可直接序列化渲染为生图模型的 Prompt 文本。
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
        """把结构化计划渲染成一段连贯的中文提示词文本。"""

        lines: list[str] = [el.to_prompt_text_line() for el in self.elements]

        segments: list[str] = ["生成一幅图片。"]

        if self.overall:
            segments.append(f"画面整体：{_normalize_clause(self.overall)}。")

        if len(lines) == 1:
            segments.append("图片的主要元素：")
            segments.append(lines[0])
        else:
            segments.append("图片包含以下元素：")

            for i, line in enumerate(lines, 1):
                segments.append(f"元素{i}：{line}")

        return "".join(segments)
