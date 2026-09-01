"""图片生成层：按参数选择生图后端。"""

from generation.factory import create_image_generator

__all__ = ["create_image_generator"]
