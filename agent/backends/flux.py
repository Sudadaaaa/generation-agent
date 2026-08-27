"""FLUX.2-klein-9B 生图后端。"""

import torch
from PIL import Image

from diffusers import Flux2KleinPipeline

from agent.backends.base import BaseImageBackend

#: 蒸馏 klein 模型的推荐步数（少步数出图，兼顾质量与速度）
NUM_INFERENCE_STEPS = 12


class FluxBackend(BaseImageBackend):
    """FLUX.2-klein-9B：BFL 蒸馏 klein 变体，约 9B transformer。"""

    def _load(self) -> None:
        if self._pipe is None:
            # local_files_only：本机网络不可达 HF hub，必须只用本地缓存。
            #
            # bf16：组件原生精度；全量 bf16 约 34GB，远超单卡显存，必须开
            # CPU offload，峰值显存降到最大单模块（transformer 约 18GB）。
            self._pipe = Flux2KleinPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )

            self._pipe.enable_model_cpu_offload(device=self.device)

    def _generate(self, prompt: str) -> Image.Image:
        assert self._pipe is not None
        # 不传 guidance_scale：is_distilled=True 的蒸馏模型会忽略 CFG
        return self._pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=self._generator(),
        ).images[0]
