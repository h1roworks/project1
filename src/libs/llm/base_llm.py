"""LLM 抽象基类。

定义所有 LLM provider 必须实现的最小接口。上层业务代码只依赖 BaseLLM，
不关心底层是 OpenAI / Azure / DeepSeek / Ollama 还是其他后端。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatResponse:
    """一次 LLM 调用的统一返回。

    ``content`` 是主要返回值（生成文本）；``model``/``usage``/``raw`` 为可选
    元信息，供调试与 Trace 使用。
    """

    content: str
    model: str = ""
    usage: dict[str, Any] | None = None
    raw: Any = None


class BaseLLM(ABC):
    """所有 LLM provider 的抽象基类。

    消息格式采用 OpenAI 兼容的 dict 列表：``[{"role": "user", "content": "..."}]``。
    """

    provider: str = "base"

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """根据消息列表生成回复。

        Args:
            messages: OpenAI 兼容的消息列表。
            trace: 可选的 TraceContext（阶段 F 落地，目前透传忽略）。
            **kwargs: provider 特定参数（temperature/max_tokens 等覆盖项）。

        Returns:
            ChatResponse，其中 ``content`` 为模型生成的文本。
        """
        raise NotImplementedError
