"""OpenAI 兼容协议 Vision LLM 的共享实现基类。

Azure Vision（GPT-4o 等）与 OpenAI/DashScope 兼容服务都遵循 OpenAI 多模态
消息协议，差异只在客户端构造参数。本模块实现图片读取/压缩、多模态消息构造与
chat 调用，子类只需实现 ``_get_client()``。
"""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image

from libs.llm.base_llm import ChatResponse, usage_to_dict
from libs.llm.base_vision_llm import BaseVisionLLM


class OpenAICompatVisionLLM(BaseVisionLLM):
    """基于 OpenAI SDK 的多模态 LLM 公共逻辑。

    图片最长边超过 ``settings.max_image_size``（默认 2048px）时自动等比压缩。
    """

    _default_base_url = ""
    _timeout = 60.0

    def __init__(self, settings: Any, http_client: Any = None) -> None:
        self.settings = settings
        self.model = settings.deployment_name or settings.model
        self._http_client = http_client
        self._client: Any = None

    def _get_client(self) -> Any:
        """按 provider 构造 OpenAI/Azure SDK 客户端（由子类实现）。"""
        raise NotImplementedError

    def _read_image(self, image_path: str | bytes) -> bytes:
        if isinstance(image_path, bytes):
            return image_path
        with open(image_path, "rb") as f:
            return f.read()

    def _image_to_data_url(self, image_path: str | bytes) -> str:
        """读取图片并构造 data URL；超限图片压缩后统一输出 PNG。

        图片未超限时保留原始字节与对应 mime（PNG/JPEG），避免无损图被二次编码。
        """
        raw = self._read_image(image_path)
        image = Image.open(io.BytesIO(raw))
        src_format = (image.format or "PNG").upper()
        mime = "image/png" if src_format == "PNG" else "image/jpeg"

        max_size = getattr(self.settings, "max_image_size", 2048)
        if max(image.size) <= max_size:
            return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

        image.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        image.save(buf, format="PNG")  # 统一 PNG，避免 JPEG 无法保存 RGBA 等限制
        payload = buf.getvalue()
        return f"data:image/png;base64,{base64.b64encode(payload).decode('ascii')}"

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": self._image_to_data_url(image_path)}},
                ],
            }
        ]
        request: dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            "max_tokens": kwargs.pop(
                "max_tokens", getattr(self.settings, "max_tokens", 1024)
            ),
        }
        temperature = kwargs.pop("temperature", None)
        if temperature is not None:
            request["temperature"] = temperature
        request.update(kwargs)

        try:
            resp = self._get_client().chat.completions.create(**request)
        except Exception as exc:  # noqa: BLE001 - 统一包装为含 provider 与错误类型的可读错误
            raise RuntimeError(
                f"Vision LLM 调用失败 (provider={self.provider}, "
                f"错误类型={type(exc).__name__}): {exc}"
            ) from exc

        return ChatResponse(
            content=(resp.choices[0].message.content or ""),
            model=getattr(resp, "model", "") or self.model,
            usage=usage_to_dict(getattr(resp, "usage", None)),
            raw=resp,
        )
