"""C9: SparseEncoder 单元测试。

验证：
- 输出契约：结构与 chunks 等长、顺序对齐，每条为 {term: tf}
- 分词行为：英文小写化、标点/连字符切分、中文单字切分、停用词过滤
- 词频统计：重复词计数正确
- 空文本行为：空/空白/无词项文本返回空 dict
- 边界行为：空 chunks、自定义停用词、输入不被修改、结果确定性
- 与 Chunker 的链路（C4 → C9）
"""

import hashlib

from core.types import Chunk
from ingestion.embedding.sparse_encoder import SparseEncoder

# ---------- fixtures / helpers ----------


def make_chunk(text: str, doc_id: str = "doc_001", index: int = 0) -> Chunk:
    meta = {"source_path": "/data/sample.pdf", "chunk_index": index, "image_refs": []}
    content_hash_short = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return Chunk(
        id=f"{doc_id}_{index:04d}_{content_hash_short}",
        text=text,
        metadata=meta,
        end_offset=len(text),
        source_ref=doc_id,
    )


# ---------- 输出契约：等长 / 顺序对齐 ----------


def test_output_count_matches_input() -> None:
    chunks = [make_chunk(f"chunk {i} content", index=i) for i in range(5)]
    result = SparseEncoder().encode(chunks)
    assert len(result) == 5  # 结构与 chunks 等长
    assert all(isinstance(d, dict) for d in result)  # 每条都是 {term: tf}


def test_output_order_aligned_with_input() -> None:
    chunks = [make_chunk("alpha beta"), make_chunk("gamma delta")]
    result = SparseEncoder().encode(chunks)
    assert result[0] == {"alpha": 1, "beta": 1}
    assert result[1] == {"gamma": 1, "delta": 1}


def test_input_chunks_not_mutated() -> None:
    chunks = [make_chunk("正文内容。")]
    before = [(c.id, c.text) for c in chunks]
    SparseEncoder().encode(chunks)
    assert [(c.id, c.text) for c in chunks] == before


# ---------- 分词行为 ----------


def test_tokenize_lowercases_english() -> None:
    assert SparseEncoder().tokenize("Hello World RAG") == ["hello", "world", "rag"]


def test_tokenize_splits_on_punctuation_and_hyphens() -> None:
    enc = SparseEncoder()
    assert enc.tokenize("text-embedding-3-small") == ["text", "embedding", "3", "small"]
    assert enc.tokenize("RAG, PDF & Markdown!") == ["rag", "pdf", "markdown"]


def test_tokenize_removes_stopwords() -> None:
    enc = SparseEncoder()
    assert enc.tokenize("the RAG system is good") == ["rag", "system", "good"]


def test_tokenize_handles_chinese_single_chars() -> None:
    assert SparseEncoder().tokenize("检索增强") == ["检", "索", "增", "强"]


def test_tokenize_mixed_cjk_and_ascii() -> None:
    assert SparseEncoder().tokenize("BM25检索") == ["bm25", "检", "索"]


# ---------- 词频统计 ----------


def test_term_freq_counts_duplicates() -> None:
    assert SparseEncoder().term_freqs("apple apple orange apple") == {
        "apple": 3,
        "orange": 1,
    }


# ---------- 空文本行为 ----------


def test_empty_text_returns_empty_dict() -> None:
    assert SparseEncoder().encode([make_chunk("")]) == [{}]


def test_whitespace_only_text_returns_empty_dict() -> None:
    assert SparseEncoder().encode([make_chunk("   \n\t ")]) == [{}]


def test_no_token_text_returns_empty_dict() -> None:
    assert SparseEncoder().encode([make_chunk("!!! ??? ---")]) == [{}]


def test_stopword_only_text_returns_empty_dict() -> None:
    assert SparseEncoder().encode([make_chunk("the and of")]) == [{}]


# ---------- 边界行为 ----------


def test_empty_chunks_returns_empty() -> None:
    assert SparseEncoder().encode([]) == []


def test_custom_stopwords() -> None:
    enc = SparseEncoder(stopwords={"apple"})
    assert enc.tokenize("apple banana") == ["banana"]
    enc_none = SparseEncoder(stopwords=set())
    assert enc_none.tokenize("the apple") == ["the", "apple"]


def test_deterministic() -> None:
    enc = SparseEncoder()
    assert enc.encode([make_chunk("RAG retrieval augmented generation")]) == enc.encode(
        [make_chunk("RAG retrieval augmented generation")]
    )


# ---------- 与 Chunker 的链路（C4 → C9） ----------


def test_sparse_encoder_chains_with_document_chunker(tmp_path) -> None:
    """真实 PDF 经 PDFLoader → DocumentChunker → SparseEncoder 产出词频结构。"""
    from reportlab.pdfgen import canvas

    from core.settings import SplitterSettings
    from ingestion.chunking.document_chunker import DocumentChunker
    from libs.loader.pdf_loader import PDFLoader

    pdf = tmp_path / "sparse_chain.pdf"
    c = canvas.Canvas(str(pdf))
    c.setFont("Helvetica", 14)
    c.drawString(72, 800, "RAG Overview")
    c.drawString(72, 770, "Sparse retrieval matches keywords exactly.")
    c.save()

    doc = PDFLoader().load(str(pdf))
    chunker = DocumentChunker(settings=SplitterSettings(chunk_size=200, chunk_overlap=0))
    chunks = chunker.chunk(doc)
    assert chunks

    result = SparseEncoder().encode(chunks)
    assert len(result) == len(chunks)
    assert all(isinstance(d, dict) for d in result)
    # 原文中的实义词应出现在词频结构里
    assert any("retrieval" in d for d in result)
    assert any("keywords" in d for d in result)
