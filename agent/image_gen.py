"""本地生图统一入口：按参数/环境变量选择后端模型。

用法：
    from agent.image_gen import create_image_generator
    gen = create_image_generator(model="flux")   # "zimage" / "flux" / 完整模型 id
    path = gen.generate("一个女孩微笑", tag="plan")
"""

import os
from pathlib import Path

from agent.backends import BaseImageBackend, FluxBackend, ZImageBackend

#: 别名 -> 后端类
BACKENDS: dict[str, type[BaseImageBackend]] = {
    "zimage": ZImageBackend,
    "flux": FluxBackend,
}

#: 别名 -> 默认本地模型 id
MODEL_IDS: dict[str, str] = {
    "zimage": "Tongyi-MAI/Z-Image-Turbo",
    "flux": "black-forest-labs/FLUX.2-klein-9B",
}


def create_image_generator(
    model: str | None = None,
    device: str | None = None,
    output_dir: str | None = None,
) -> BaseImageBackend:
    """创建选中的生图后端。

    model 可以是别名（zimage/flux）或完整本地模型 id；
    未指定时依次取 GEN_MODEL 环境变量，缺省用 zimage。
    后端按模型名自动识别，未知 id 默认 ZImage。
    """

    model = model or os.getenv("GEN_MODEL") or "zimage"
    device = device or os.getenv("GEN_DEVICE") or "cuda:0"
    output_dir = output_dir or os.getenv("OUTPUT_DIR") or "outputs"

    alias = model.strip().lower()

    if alias in BACKENDS:
        backend_cls = BACKENDS[alias]
        model_id = MODEL_IDS[alias]
    else:
        # 完整本地模型 id：按名字自动识别后端
        model_id = model
        backend_cls = FluxBackend if "flux" in model.lower() else ZImageBackend

    return backend_cls(
        model_id=model_id,
        device=device,
        save_dir=Path(output_dir),
    )
