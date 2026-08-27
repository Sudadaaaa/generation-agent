from typing import Optional

from pydantic import BaseModel, Field


class Element(BaseModel):
    """画面中的一个基本元素。"""

    name: str = Field(
        description=(
            "元素是什么（类别或身份），例如：女孩、成年男性、橘猫、木船、草坪、太阳。"
        )
    )

    count: Optional[str] = Field(
        default=None,
        description="数量，例如：一个、两只、三朵、一片、成群。",
    )

    appearance: Optional[str] = Field(
        default=None,
        description=(
            "外观：形状、颜色、材质、纹理、新旧/状态等。"
            "按元素类型覆盖：人物→年龄感、体型、发型、发色、五官、肤色、服饰、表情；"
            "物体→形状、颜色、材质、纹理、破损或新旧，及物体上印刻的文字（保持原文）；"
            "自然元素→形态、颜色、光照感。"
            "例如：金发白裙、翠绿、橙色发光、漆皮剥落、红底白字'营业中'。"
        ),
    )

    position: Optional[str] = Field(
        default=None,
        description=(
            "元素在画面中的位置与景深层次，例如：中央、前景、中景、背景、"
            "左上方、远处。不要包含'画面'二字。单一主体默认'中央'。"
        ),
    )

    size: Optional[str] = Field(
        default=None,
        description="相对大小，例如：大、小、占据画面一半。",
    )

    action: Optional[str] = Field(
        default=None,
        description=(
            "动作、姿态或状态，例如：张开双手、奔跑、飘动、发光。"
            "动作引发的瞬态效果也写入此字段，如：跳跃溅起的水花、奔跑扬起的尘土。"
            "没有明确动作时填'静止'或该类元素最常见的状态。"
        ),
    )

    relation: Optional[str] = Field(
        default=None,
        description=(
            "该元素与其他元素的空间/语义关系，引用元素名称描述，"
            "也可引用'镜头/天空/远方/画面外'等全局参照。"
            "例如：站在草坪上、面对太阳、在房子后面、手里拿着一本书、凝视镜头。"
        ),
    )

    def to_prompt_text(self) -> str:
        """把单个元素渲染成一个自然的中文句子。"""

        parts: list[str] = []

        subject = f"{self.count}{self.name}" if self.count else self.name
        parts.append(subject)

        if self.appearance:
            parts.append(self.appearance)

        if self.position:
            parts.append(f"位于{self.position}")

        if self.size:
            parts.append(self.size)

        if self.relation:
            parts.append(self.relation)

        if self.action:
            parts.append(self.action)

        return "，".join(parts) + "。"


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
            "⑧画质与细节；⑨不应出现的内容。"
            "注意：元素的细节（外观/动作/位置/服饰/表情等）只写在对应 element 中，"
            "用户未提到的方面按画面最合理的方式默认补齐，无法推断才为 null。"
        ),
    )

    def to_prompt_text(self) -> str:
        """把结构化计划渲染成一段连贯的中文提示词文本。"""

        segments: list[str] = []

        intro = "生成一幅图片"

        if self.overall:
            intro += f"，画面整体：{self.overall.rstrip('，。； ')}"

        segments.append(intro + "。")

        if len(self.elements) == 1:
            segments.append(self.elements[0].to_prompt_text())
        else:
            segments.append("画面包含以下元素：")

            for i, el in enumerate(self.elements, 1):
                segments.append(f"元素{i}：{el.to_prompt_text()}")

        return "".join(segments)
