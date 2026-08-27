"""本地 diffusers 生图：Z-Image-Turbo。"""

import os
import torch
from datetime import datetime
from pathlib import Path

from diffusers import ZImagePipeline
class ImageGenerator:
    """把一段中文提示词渲染成一张图片。"""

    def __init__(self) -> None:
        self.model_id = os.getenv("GEN_MODEL", "Tongyi-MAI/Z-Image-Turbo")
        self.device = os.getenv("GEN_DEVICE", "cuda:0")
        self.save_dir = Path(os.getenv("OUTPUT_DIR", "outputs"))
        self._pipe: ZImagePipeline | None = None

    def _load(self) -> ZImagePipeline:
        if self._pipe is None:
            # local_files_only：本机网络不可达 HF hub，
            # 必须强制只用本地缓存，否则 from_pretrained 会卡在联网检查上。
            #
            # bf16：模型原生精度（text_encoder/vae 为 bf16，transformer 为 fp32），
            # 用 fp16 会因中间张量溢出产生 NaN，输出全黑图。
            self._pipe = ZImagePipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )

            # 权重放 CPU，计算时按需搬上 GPU，峰值显存降到单个模块大小，
            # 适合与其它进程共享、显存实时浮动的机器。
            self._pipe.enable_model_cpu_offload(device=self.device)
        return self._pipe

    def generate(self, prompt: str, tag: str = "") -> Path:
        """生成一张图片并保存，返回文件路径。tag 用于区分输出文件。"""

        pipe = self._load()

        image = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=8,
            guidance_scale=2.0,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]

        self.save_dir.mkdir(parents=True, exist_ok=True)

        tag_part = f"_{tag}" if tag else ""
        # 毫秒级时间戳 + tag，避免同一次运行连续出图时文件名冲突
        fname = f"img_{datetime.now():%Y%m%d_%H%M%S%f}{tag_part}.png"
        path = self.save_dir / fname
        image.save(path)

        return path
