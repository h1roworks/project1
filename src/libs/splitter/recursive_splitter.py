"""Recursive Splitter：封装 LangChain 的 RecursiveCharacterTextSplitter 作为默认切分器。

按 Markdown 结构递归切分：优先在标题 / 代码块边界处断开，尽量保持结构完整，
再逐步退化到段落、行、词、字符。
"""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory

# 优先在 Markdown 结构边界切分（标题、代码块围栏），保证这些结构不被打断
_MARKDOWN_SEPARATORS = [
    "\n\n## ",
    "\n\n# ",
    "\n\n### ",
    "\n\n#### ",
    "\n\n##### ",
    "\n```",
    "\n\n",
    "\n",
    " ",
    "",
]


class RecursiveSplitter(BaseSplitter):
    strategy = "recursive"

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(settings.chunk_size, 1),
            chunk_overlap=max(settings.chunk_overlap, 0),
            separators=settings.separators or _MARKDOWN_SEPARATORS,
        )

    def split_text(self, text: str, trace: Any = None) -> list[str]:
        if not text:
            return []
        return self._splitter.split_text(text)


SplitterFactory.register("recursive", RecursiveSplitter)
