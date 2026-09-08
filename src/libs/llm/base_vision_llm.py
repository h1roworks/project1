"""Vision LLM 抽象基类。

多模态 LLM（文本 + 图片）的统一接口，为阶段 C 的 ImageCaptioner 提供底层抽象。
图片输入支持两种形式：文件路径（str）或原始字节（bytes，自动 base64 编码）。

子类可通过重写 ``_prepare_image``（读取/压缩/格式转换）扩展图片预处理逻辑。
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from typing import Any

from libs.llm.base_llm import ChatResponse


class BaseVisionLLM(ABC):
    provider: str = "base_vision"

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """根据文本提示与图片生成回复。

        Args:
            text: 文本提示（如 "请描述这张图片"）。
            image_path: 图片文件路径，或原始图片字节（bytes）。
            trace: 可选的 TraceContext（阶段 F 落地，目前透传忽略）。
            **kwargs: provider 特定参数（temperature/max_tokens 等覆盖项）。

        Returns:
            ChatResponse，其中 ``content`` 为模型生成的文本。
        """
        raise NotImplementedError

    def _prepare_image(self, image_path: str | bytes) -> bytes:
        """图片预处理扩展点：返回编码前的原始字节（默认原样读取）。"""
        if isinstance(image_path, bytes):
            return image_path
        with open(image_path, "rb") as f:
            return f.read()

    def _encode_image(self, image_path: str | bytes) -> str:
        """把图片（经 ``_prepare_image`` 预处理后）编码为 base64 字符串。"""
        return base64.b64encode(self._prepare_image(image_path)).decode("ascii")
