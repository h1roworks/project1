"""Document → Chunks 转换（C4：Splitter 集成，调用 Libs）。

职责边界（对齐 DEV_SPEC 3.1.1 Splitter / 5.3.4 document_chunker）：
- 调用 ``libs.splitter``（SplitterFactory）把 ``Document.text`` 切分为若干片段；
- 将每个片段封装为 ``core.types.Chunk``，携带稳定 ID、字符定位偏移与溯源链接；
- ``metadata`` 从 Document 继承，并追加 ``chunk_index``、``image_refs``；
- 只做"切分 + 封装"，不涉及 Embedding / 存储（那是 C8~C12 的职责）。
"""

from __future__ import annotations

import hashlib
import re

from core.settings import SplitterSettings
from core.types import Chunk, Document, extract_image_ids
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory


class DocumentChunker:
    """把 Loader 产出的 ``Document`` 切分为若干带定位信息的 ``Chunk``。"""

    def __init__(
        self,
        settings: SplitterSettings | None = None,
        splitter: BaseSplitter | None = None,
    ) -> None:
        """初始化。

        Args:
            settings: 切分配置；缺省用默认值（recursive / chunk_size=1000）。
            splitter: 可注入的 BaseSplitter（测试可传 Fake，缺省经工厂按配置创建）。
        """
        self._settings = settings or SplitterSettings()
        self._splitter = splitter or SplitterFactory.create(self._settings)

    def chunk(self, document: Document) -> list[Chunk]:
        """把文档切分为 Chunk 列表（空文本返回空列表，纯空白片段被跳过）。"""
        if not document.text:
            return []

        chunks: list[Chunk] = []
        search_from = 0
        pieces = [p for p in self._splitter.split_text(document.text) if p.strip()]
        for index, text in enumerate(pieces):
            start, end = _find_text_span(document.text, text, search_from)
            search_from = end
            metadata = dict(document.metadata)
            # Document-level MCP tools need a stable way to group all stored
            # chunks back into their source document after ingestion.
            metadata["doc_id"] = document.id
            metadata["chunk_index"] = index
            metadata["image_refs"] = extract_image_ids(text)
            chunks.append(
                Chunk(
                    id=_make_chunk_id(document.id, index, text),
                    text=text,
                    metadata=metadata,
                    start_offset=start,
                    end_offset=end,
                    source_ref=document.id,
                )
            )
        return chunks


def _make_chunk_id(doc_id: str, index: int, text: str) -> str:
    """生成稳定 Chunk ID：``{doc_id}_{index:04d}_{content_hash 前 8 位}``。"""
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}_{index:04d}_{content_hash}"


def _find_text_span(text: str, chunk: str, search_from: int) -> tuple[int, int]:
    """从 ``search_from`` 起，在 ``text`` 中定位 ``chunk`` 的字符区间 ``[start, end)``。

    优先精确匹配；LangChain 切分会对空白归位，导致片段与原文在空白处不一致，
    因此失败时回退到"空白不敏感"的模糊匹配（把片段中的空白序列视作 ``\\s+``）。
    极端情况仍找不到时，退回 ``[search_from, ...]`` 并钳制在文末作为兜底。
    """
    start = text.find(chunk, search_from)
    if start != -1:
        return start, start + len(chunk)

    pattern = _fuzzy_pattern(chunk)
    if pattern is not None:
        match = pattern.search(text, search_from)
        if match:
            return match.start(), match.end()

    return search_from, min(search_from + len(chunk), len(text))


def _fuzzy_pattern(chunk: str) -> re.Pattern[str] | None:
    """把片段构建为空白不敏感的正则（空白序列 → ``\\s+``）；无有效字符返回 None。"""
    tokens = chunk.strip().split()
    if not tokens:
        return None
    return re.compile(r"\s+".join(re.escape(tok) for tok in tokens), re.DOTALL)
