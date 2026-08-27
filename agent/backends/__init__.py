"""生图后端：每个模型一个文件，统一接口见 base.BaseImageBackend。"""

from agent.backends.base import BaseImageBackend
from agent.backends.zimage import ZImageBackend
from agent.backends.flux import FluxBackend

__all__ = ["BaseImageBackend", "ZImageBackend", "FluxBackend"]
