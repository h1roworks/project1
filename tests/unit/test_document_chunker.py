"""C4: Splitter 集成（Document → Chunks 转换）单元测试。

验证：
- DocumentChunker 默认经 SplitterFactory 创建 recursive splitter（调用 Libs）
- 切分结果正文内容不丢失（空白归位不影响内容）
- Chunk ID 格式 ``{doc_id}_{index:04d}_{hash}``，内容相关且确定性
- source_ref / start_offset / end_offset 定位信息正确、有序且不重叠
- metadata 从 Document 继承且逐 chunk 隔离；追加 chunk_index / image_refs
- 图片占位符随切分保留，image_refs 只含本 chunk 的图片
- 边界：空文本 → 空列表；纯空白片段被跳过；定位的精确/模糊/兜底匹配
"""

import re

import pytest

from core.settings import SplitterSettings
from core.types import Chunk, Document
from ingestion.chunking.document_chunker import (
    DocumentChunker,
    _find_text_span,
    _make_chunk_id,
)
from libs.splitter.base_splitter import BaseSplitter

# ---------- fixture ----------


def make_document(text: str | None = None) -> Document:
    return Document(
        id="doc_001",
        text=text if text is not None else "# 标题\n\n第一段。\n\n第二段。\n\n第三段。",
        metadata={
            "source_path": "/data/documents/sample.pdf",
            "doc_type": "pdf",
            "title": "示例文档",
        },
    )


class ParagraphSplitter(BaseSplitter):
    """按空行切分（边界可控），用于验证定位 / ID / metadata 逻辑。"""

    strategy = "paragraph"

    def split_text(self, text, trace=None):
        return [p for p in text.split("\n\n") if p]


class NoisySplitter(BaseSplitter):
    """返回含纯空白片段的结果，验证被跳过且 chunk_index 连续。"""

    strategy = "noisy"

    def split_text(self, text, trace=None):
        return ["有效第一段", "   \n  ", "有效第二段"]


# ---------- 调用 Libs（SplitterFactory） ----------


def test_default_chunker_uses_libs_splitter_factory() -> None:
    """无参构造 → 经 SplitterFactory 创建默认 recursive splitter。"""
    chunker = DocumentChunker()
    chunks = chunker.chunk(make_document())
    assert chunks
    assert isinstance(chunks[0], Chunk)


def test_chunker_accepts_splitter_settings() -> None:
    settings = SplitterSettings(strategy="recursive", chunk_size=15, chunk_overlap=0)
    chunks = DocumentChunker(settings=settings).chunk(make_document())
    assert len(chunks) >= 2


def test_chunker_accepts_injected_splitter() -> None:
    chunks = DocumentChunker(splitter=ParagraphSplitter()).chunk(make_document())
    assert len(chunks) == 4  # 标题 + 三个段落


# ---------- 内容与 ID ----------


def test_content_preserved_whitespace_insensitive() -> None:
    """LangChain 切分会对空白归位，但正文内容一个字符都不能丢。"""
    text = "# 项目介绍\n\n" + "这是第一段正文，用于验证切分时正文内容不丢失。\n\n" * 6
    chunker = DocumentChunker(
        settings=SplitterSettings(chunk_size=60, chunk_overlap=10)
    )
    chunks = chunker.chunk(make_document(text=text))

    def strip_ws(s: str) -> str:
        return re.sub(r"\s", "", s)

    assert "".join(strip_ws(c.text) for c in chunks) == strip_ws(text)


def test_chunk_id_format_and_determinism() -> None:
    text = "第一段内容。\n\n第二段内容。"
    chunker = DocumentChunker(splitter=ParagraphSplitter())
    chunks = chunker.chunk(make_document(text=text))
    assert chunks
    pattern = re.compile(r"^doc_001_\d{4}_[0-9a-f]{8}$")
    assert all(pattern.match(c.id) for c in chunks)

    # 同输入 → 同 ID（确定性）；内容变更 → hash 段变化（ID 随之不同）
    again = chunker.chunk(make_document(text=text))
    assert [c.id for c in chunks] == [c.id for c in again]
    changed = chunker.chunk(make_document(text=text + "\n\n新增第三段。"))
    assert [c.id for c in chunks] != [c.id for c in changed]


# ---------- 溯源与定位信息 ----------


def test_source_ref_points_to_document() -> None:
    chunks = DocumentChunker(splitter=ParagraphSplitter()).chunk(make_document())
    assert chunks
    assert all(c.source_ref == "doc_001" for c in chunks)


def test_offsets_in_bounds_and_ordered() -> None:
    text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
    chunker = DocumentChunker(splitter=ParagraphSplitter())
    chunks = chunker.chunk(make_document(text=text))
    assert chunks
    for c in chunks:
        assert 0 <= c.start_offset <= c.end_offset <= len(text)
    for prev, cur in zip(chunks, chunks[1:]):
        assert prev.end_offset <= cur.start_offset
    assert chunks[0].start_offset == 0
    # 片段文本与原文在该区间精确一致（分隔符对齐时）
    for c in chunks:
        assert text[c.start_offset : c.end_offset] == c.text


def test_offsets_valid_on_real_recursive_splitter() -> None:
    """真实 recursive splitter + 模糊匹配回退时，偏移仍有序且在界内。"""
    text = (
        "# 项目介绍\n\n"
        + ("这是第一段正文内容，用于测试切分边界与偏移定位是否正确。\n\n" * 8)
        + "## 深入细节\n\n"
        + ("这是第二段更长的正文，用于验证切分与偏移在更复杂文本下依然可靠。" * 20)
    )
    chunker = DocumentChunker(
        settings=SplitterSettings(chunk_size=80, chunk_overlap=20)
    )
    chunks = chunker.chunk(make_document(text=text))
    assert len(chunks) >= 2
    for c in chunks:
        assert 0 <= c.start_offset <= c.end_offset <= len(text)
    for prev, cur in zip(chunks, chunks[1:]):
        assert prev.end_offset <= cur.start_offset


# ---------- metadata 继承与增强 ----------


def test_metadata_inherited_and_added() -> None:
    doc = make_document()
    chunks = DocumentChunker(splitter=ParagraphSplitter()).chunk(doc)
    assert len(chunks) == 4  # 标题 + 三个段落
    for i, c in enumerate(chunks):
        assert c.metadata["source_path"] == doc.metadata["source_path"]
        assert c.metadata["doc_type"] == "pdf"
        assert c.metadata["chunk_index"] == i
        assert c.metadata["image_refs"] == []


def test_metadata_isolated_per_chunk() -> None:
    doc = make_document()
    chunks = DocumentChunker(splitter=ParagraphSplitter()).chunk(doc)
    chunks[0].metadata["extra"] = "mutated"
    assert "extra" not in chunks[1].metadata
    assert "extra" not in doc.metadata


def test_image_placeholders_preserved_and_refs_extracted() -> None:
    text = (
        "前文段落 [IMAGE: img_a_0] 继续。\n\n"
        "中段无图。\n\n"
        "后文 [IMAGE: img_c_1] 结束。"
    )
    chunks = DocumentChunker(splitter=ParagraphSplitter()).chunk(make_document(text=text))
    assert len(chunks) == 3
    assert "[IMAGE: img_a_0]" in chunks[0].text
    assert chunks[0].metadata["image_refs"] == ["img_a_0"]
    assert chunks[1].metadata["image_refs"] == []
    assert chunks[2].metadata["image_refs"] == ["img_c_1"]


# ---------- 边界处理 ----------


def test_empty_text_returns_empty() -> None:
    assert DocumentChunker(splitter=ParagraphSplitter()).chunk(make_document(text="")) == []


def test_whitespace_only_pieces_skipped() -> None:
    chunks = DocumentChunker(splitter=NoisySplitter()).chunk(make_document(text="任意文本"))
    assert [c.text for c in chunks] == ["有效第一段", "有效第二段"]
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1]


# ---------- 与 Loader 的链路（C3 → C4） ----------


def test_chunker_chains_with_pdf_loader(tmp_path) -> None:
    """真实 PDF 经 PDFLoader → DocumentChunker 产出带定位信息的 Chunk。"""
    from reportlab.pdfgen import canvas

    from libs.loader.pdf_loader import PDFLoader

    pdf = tmp_path / "chain.pdf"
    c = canvas.Canvas(str(pdf))
    c.setFont("Helvetica", 14)
    c.drawString(72, 800, "# RAG Overview")
    c.drawString(72, 770, "RAG combines retrieval with generation.")
    c.drawString(72, 740, "BM25 matches exact keywords.")
    c.save()

    doc = PDFLoader().load(str(pdf))
    chunker = DocumentChunker(
        settings=SplitterSettings(chunk_size=40, chunk_overlap=0)
    )
    chunks = chunker.chunk(doc)

    assert chunks
    for chunk in chunks:
        assert chunk.source_ref == doc.id
        assert chunk.metadata["source_path"] == str(pdf)
        assert "chunk_index" in chunk.metadata
        assert 0 <= chunk.start_offset <= chunk.end_offset <= len(doc.text)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


# ---------- 内部定位工具（精确 / 模糊 / 兜底） ----------


def test_make_chunk_id_format() -> None:
    assert re.fullmatch(r"doc_\d{4}_[0-9a-f]{8}", _make_chunk_id("doc", 3, "正文"))


def test_find_text_span_exact() -> None:
    assert _find_text_span("hello world hello", "world", 0) == (6, 11)


def test_find_text_span_respects_search_from() -> None:
    assert _find_text_span("ab ab", "ab", 1) == (3, 5)


def test_find_text_span_fuzzy_whitespace() -> None:
    """空白归位导致与原文不一致时，按空白不敏感方式仍能定位整段。"""
    text = "alpha   beta\n\n gamma"
    assert _find_text_span(text, "alpha beta\ngamma", 0) == (0, len(text))


def test_find_text_span_fallback_clamped() -> None:
    text = "完全不同的文本"
    start, end = _find_text_span(text, "不存在的片段内容", 0)
    assert start == 0 and end == len(text)  # 兜底钳制在文末，不越界
