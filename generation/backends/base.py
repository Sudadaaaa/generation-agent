"""生图后端统一接口：负责加载 pipeline、生成图片并保存。"""

import torch
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from PIL import Image


class BaseImageBackend(ABC):
    """不同生图模型后端的统一接口。

    子类只需实现 _load()（加载并缓存 pipeline）和 _generate()（调用
    pipeline 生成一张图）；保存文件、文件名等公共逻辑都在 generate() 里。
    新增模型 = 新建一个后端文件 + 在 generation/factory.py 注册一行。
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        save_dir: str | Path,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.save_dir = Path(save_dir)
        self._pipe = None

    @abstractmethod
    def _load(self) -> None:
        """加载并缓存 pipeline（惰性，首次 generate 时才调用）。"""

    @abstractmethod
    def _generate(self, prompt: str) -> Image.Image:
        """用已加载的 pipeline 把 prompt 渲染成一张 PIL 图片。"""

    def _generator(self) -> torch.Generator:
        """默认随机源，保证同一提示词结果可复现；子类可覆写。"""
        return torch.Generator(device=self.device).manual_seed(42)

    def generate(self, prompt: str, tag: str = "") -> Path:
        """生成一张图片并保存，返回文件路径。tag 用于区分输出文件。"""

        self._load()

        image = self._generate(prompt)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        tag_part = f"_{tag}" if tag else ""
        # 毫秒级时间戳 + tag，避免同一次运行连续出图时文件名冲突
        fname = f"img_{datetime.now():%Y%m%d_%H%M%S%f}{tag_part}.png"
        path = self.save_dir / fname
        image.save(path)

        return path
