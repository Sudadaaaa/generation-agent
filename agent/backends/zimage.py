"""Z-Image-Turbo 生图后端。"""

import torch
from PIL import Image

from diffusers import ZImagePipeline

from agent.backends.base import BaseImageBackend


class ZImageBackend(BaseImageBackend):
    """Z-Image-Turbo：中文原生支持的轻量 turbo 模型。"""

    def _load(self) -> None:
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

    def _generate(self, prompt: str) -> Image.Image:
        assert self._pipe is not None
        # guidance_scale=0：turbo 是蒸馏模型，官方配方不走 CFG。
        # 实测传 guidance>0（启用 CFG）会明显变糊：同提示词同种子，
        # 拉普拉斯方差从 6.6（guidance=2.0）提升到 102.8（guidance=0.0）。
        return self._pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=9,
            guidance_scale=0,
            generator=self._generator(),
        ).images[0]
