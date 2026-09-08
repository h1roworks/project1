"""Splitter 抽象基类。

定义文本切分策略的最小接口：把一段长文本切成若干片段。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSplitter(ABC):
    """所有文本切分器的抽象基类。

    注意：这里只负责 ``str -> List[str]`` 的纯文本切分，
    不涉及 Document/Chunk 等业务对象（那是 ingestion.chunking 的职责）。
    """

    strategy: str = "base"

    @abstractmethod
    def split_text(self, text: str, trace: Any = None) -> list[str]:
        """把一段文本切分为若干片段。

        Args:
            text: 待切分的文本。
            trace: 可选的 TraceContext（阶段 F 落地，目前透传忽略）。

        Returns:
            切分后的文本片段列表。
        """
        raise NotImplementedError
