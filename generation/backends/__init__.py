"""生图后端：每个模型一个文件，统一接口见 base.BaseImageBackend。"""

from generation.backends.base import BaseImageBackend
from generation.backends.flux import FluxBackend
from generation.backends.zimage import ZImageBackend

__all__ = ["BaseImageBackend", "ZImageBackend", "FluxBackend"]
