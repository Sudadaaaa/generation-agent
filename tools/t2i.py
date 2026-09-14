"""generate_image 工具：真实出图。

生图后端的实现都并在这里——旧 generation/ 那层只为这一个工具存在（外部唯一调用方
是 main.py 里已删的旧 build_session），拆成 base + zimage + flux + factory 四个文件，
是给一个消费者发四个抽屉。

懒加载：模块顶层只有标准库 + pydantic，torch / diffusers / PIL 都在方法里 import。
所以 tools/__init__.py 可以无条件 import 本模块：没配生图的环境启动仍是秒开，
模型加载推迟到第一次 generate()，装配期不占显存。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.tool import Args, Tool


@dataclass(frozen=True)
class _ModelSpec:
    """一个生图模型的全部差异。

    这些是「模型怎么跑」的配方，属于代码，不进环境变量——塞进 .env 只会变成看不见的
    隐藏开关（本仓库已有的教训：AGENT_WORKER_PROVIDER=qwen 被接受、然后被无视）。
    """

    pipeline: str                  # diffusers 里的 pipeline 类名；存字符串，_load() 时才取
    model_id: str                  # 本地 HF 缓存里的模型 id
    steps: int                     # 蒸馏模型的推荐步数
    guidance: float | None = None  # None = 调用时不传 guidance_scale


MODELS: dict[str, _ModelSpec] = {
    "Z-Image-Turbo": _ModelSpec("ZImagePipeline", "Tongyi-MAI/Z-Image-Turbo", steps=9, guidance=0.0),
    "FLUX.2-klein-9B": _ModelSpec("Flux2KleinPipeline", "black-forest-labs/FLUX.2-klein-9B", steps=12),
}


class ImageGenerator:
    """把一段提示词渲染成一张本机 PNG。

    spec 的四个值就是属性，要改直接改（gen.steps = 16），不必新建子类。
    """

    def __init__(self, model: str | None = None, output_dir: str | None = None,
                 device: str | None = None) -> None:
        # 不填 → 缺省模型。键必须是 MODELS 里有的名字（.env 的 T2I_MODEL_ID 同理）。
        spec = MODELS[model or "Z-Image-Turbo"]
        self.pipeline = spec.pipeline
        self.model_id = spec.model_id
        self.steps = spec.steps
        self.guidance = spec.guidance

        # 走 main.py 时 device 一定由 T2I_MODEL_DEVICE 传进来，落不到这个缺省；
        # 它只给直接 new ImageGenerator() 的场合（测试 / REPL）兜底。
        self.device = device or "cuda:0"
        self.save_dir = Path(output_dir or "outputs")
        self.seed = 42
        self._pipe = None

    def _load(self) -> None:
        """加载并缓存 pipeline——惰性，首次 generate() 才走这里。

        pipeline 类用字符串名在这里取：模块顶层因此不必 import diffusers，
        本机 diffusers 缺某个 pipeline 也只坏那一个模型，而不是 import 就炸。
        """
        if self._pipe is not None:
            return

        import diffusers
        import torch

        try:
            pipeline_cls = getattr(diffusers, self.pipeline)
        except AttributeError as e:
            raise RuntimeError(
                f"当前 diffusers（{diffusers.__version__}）里没有 {self.pipeline}，"
                f"加载不了 {self.model_id}"
            ) from e

        # local_files_only：本机网络不可达 HF hub，必须只用本地缓存，
        # 否则 from_pretrained 会卡在联网检查上。
        #
        # bf16：组件原生精度（用 fp16 会因中间张量溢出出全黑图）。
        #
        # enable_model_cpu_offload：权重放 CPU、计算时按需搬上 GPU，峰值显存降到单个
        # 模块大小（FLUX.2-klein-9B 全量 bf16 约 34GB，远超单卡）。
        self._pipe = pipeline_cls.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        self._pipe.enable_model_cpu_offload(device=self.device)

    def _generator(self):
        """默认随机源，保证同一提示词结果可复现。"""
        import torch

        return torch.Generator(device=self.device).manual_seed(self.seed)

    def _render(self, prompt: str):
        kwargs: dict[str, Any] = dict(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=self.steps,
            generator=self._generator(),
        )
        # guidance=None 表示不传：FLUX.2-klein-9B 是 is_distilled 模型，传了会被忽略。
        # Z-Image-Turbo 传 0：实测传 >0（启用 CFG）会明显变糊——同提示词同种子，
        # 拉普拉斯方差从 6.6（guidance=2.0）提升到 102.8（guidance=0.0）。
        if self.guidance is not None:
            kwargs["guidance_scale"] = self.guidance
        return self._pipe(**kwargs).images[0]

    def generate(self, prompt: str, tag: str = "") -> Path:
        """生成一张图片并保存，返回文件路径。tag 用于区分输出文件。"""
        self._load()

        image = self._render(prompt)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        tag_part = f"_{tag}" if tag else ""
        # 毫秒级时间戳 + tag，避免同一次运行连续出图时文件名冲突
        path = self.save_dir / f"img_{datetime.now():%Y%m%d_%H%M%S%f}{tag_part}.png"
        image.save(path)

        return path


class GenerateImageArgs(BaseModel):
    """generate_image 的参数。"""

    prompt: str = Field(description="要出图的最终提示词（自然语言，中文）")


@dataclass
class GenerateImageTool(Tool):
    name: str = "generate_image"
    description: str = "用一段提示词真实生成一张图片并保存到本地，返回保存路径"
    args_schema: Args = GenerateImageArgs

    # 以下都是可选装配项；全不填 = Z-Image-Turbo / outputs / cuda:0，与 AddTool() 一样裸用可用
    model: str | None = None      # MODELS 的键：Z-Image-Turbo 或 FLUX.2-klein-9B
    output_dir: str | None = None
    device: str | None = None
    tag: str = "raw"              # 产物文件名标签（阶段标记，不是模型该判断的东西）
    generator: Any = None         # 逃生口：测试注入假 generator，不拉起 torch

    def __post_init__(self) -> None:
        # 构造很轻：ImageGenerator 只存字段，_load() 是惰性的，装配期不加载模型。
        if self.generator is None:
            self.generator = ImageGenerator(
                model=self.model,
                output_dir=self.output_dir,
                device=self.device,
            )

    def run(self, parameters: GenerateImageArgs) -> str:
        path = self.generator.generate(parameters.prompt, tag=self.tag)
        return f"图片已生成：{path}"
