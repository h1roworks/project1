"""C8: DenseEncoder 单元测试。

验证：
- 输出契约：向量数量与 chunks 一致、维度一致、与输入顺序对齐
- 内容哈希缓存（增量编码）：相同内容复用缓存向量，不重复调用 Embedding API
- 分批控制：batch_size 生效，Embedding 按批调用且批次顺序稳定
- 边界行为：空 chunks、禁用缓存、provider 返回数量不匹配时报错、内容哈希确定性
- 与 Chunker 的链路（C4 → C8）
"""

import hashlib

import pytest

from core.types import Chunk
from ingestion.embedding.dense_encoder import DenseEncoder, content_hash
from libs.embedding.base_embedding import BaseEmbedding

# ---------- fixtures / helpers ----------


class FakeEmbedding(BaseEmbedding):
    """可注入的假 Embedding：返回内容长度派生的确定性向量，并记录每次调用。"""

    provider = "c8fake"

    def __init__(self, dimensions: int = 4) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []  # 每次 embed 收到的文本批次
        self.traces: list[object] = []  # 每次 embed 收到的 trace

    def embed(self, texts, trace=None) -> list[list[float]]:
        self.calls.append(list(texts))
        self.traces.append(trace)
        return [[float(len(t)) % 100] * self.dimensions for t in texts]


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


# ---------- 输出契约：数量 / 维度 / 顺序对齐 ----------


def test_output_matches_input_count_and_dimension() -> None:
    emb = FakeEmbedding(dimensions=4)
    chunks = [make_chunk(f"chunk {i} content", index=i) for i in range(5)]
    vectors = DenseEncoder(emb).encode(chunks)
    assert len(vectors) == 5  # 向量数量与 chunks 数量一致
    assert all(len(v) == 4 for v in vectors)  # 维度一致


def test_output_order_aligned_with_input() -> None:
    emb = FakeEmbedding()
    chunks = [make_chunk("first"), make_chunk("second"), make_chunk("third")]
    vectors = DenseEncoder(emb).encode(chunks)
    assert vectors[0] == emb.embed(["first"])[0]
    assert vectors[1] == emb.embed(["second"])[0]
    assert vectors[2] == emb.embed(["third"])[0]


def test_input_chunks_not_mutated() -> None:
    chunks = [make_chunk("正文内容。")]
    before = [(c.id, c.text) for c in chunks]
    DenseEncoder(FakeEmbedding()).encode(chunks)
    assert [(c.id, c.text) for c in chunks] == before


# ---------- 分批控制 ----------


def test_batch_size_splits_embed_calls() -> None:
    emb = FakeEmbedding()
    chunks = [make_chunk(f"c{i}", index=i) for i in range(5)]
    DenseEncoder(emb, batch_size=2).encode(chunks)
    assert [len(call) for call in emb.calls] == [2, 2, 1]  # 5 条文本按 2 一批 → 3 批
    # 批次拼接后与输入顺序一致
    assert [t for call in emb.calls for t in call] == [c.text for c in chunks]


def test_batch_size_from_settings() -> None:
    from core.settings import EmbeddingSettings

    emb = FakeEmbedding()
    chunks = [make_chunk(f"c{i}", index=i) for i in range(5)]
    DenseEncoder(emb, settings=EmbeddingSettings(batch_size=2)).encode(chunks)
    assert [len(call) for call in emb.calls] == [2, 2, 1]


# ---------- 内容哈希缓存（增量编码） ----------


def test_cache_reuses_identical_content_across_calls() -> None:
    """增量编码：第二次 encode 内容未变（如改名重摄取）时复用缓存向量。"""
    emb = FakeEmbedding()
    encoder = DenseEncoder(emb)
    chunks = [make_chunk("same content", index=i) for i in range(3)]

    first = encoder.encode(chunks)
    assert encoder.embedded_count == 3
    assert encoder.cache_hit_count == 0
    assert len(emb.calls) == 1  # 第一次全量编码

    second = encoder.encode(chunks)  # 内容未变：全部命中缓存，不再调用 API
    assert encoder.embedded_count == 3
    assert encoder.cache_hit_count == 3
    assert len(emb.calls) == 1
    assert second == first


def test_cache_disabled_reembeds_all() -> None:
    emb = FakeEmbedding()
    encoder = DenseEncoder(emb, use_cache=False)
    chunks = [make_chunk("same content", index=i) for i in range(3)]
    vectors = encoder.encode(chunks)
    assert encoder.embedded_count == 3
    assert encoder.cache_hit_count == 0
    assert len(emb.calls) == 1  # 仍为一批送入，但不做缓存去重
    assert all(v == vectors[0] for v in vectors)


# ---------- 边界行为 ----------


def test_empty_chunks_returns_empty() -> None:
    emb = FakeEmbedding()
    vectors = DenseEncoder(emb).encode([])
    assert vectors == []
    assert emb.calls == []


def test_mismatched_output_raises() -> None:
    class ShortEmbedding(BaseEmbedding):
        provider = "c8short"

        def embed(self, texts, trace=None) -> list[list[float]]:
            return [[0.0]] * (len(texts) - 1)  # 少返回一条向量

    with pytest.raises(ValueError, match="不一致"):
        DenseEncoder(ShortEmbedding()).encode([make_chunk("a"), make_chunk("b")])


def test_content_hash_deterministic() -> None:
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("world")


def test_trace_passed_through() -> None:
    emb = FakeEmbedding()
    DenseEncoder(emb).encode([make_chunk("hello")], trace="t0")
    assert emb.traces == ["t0"]


# ---------- 与 Chunker 的链路（C4 → C8） ----------


def test_dense_encoder_chains_with_document_chunker(tmp_path) -> None:
    """真实 PDF 经 PDFLoader → DocumentChunker → DenseEncoder 产出对齐向量。"""
    from reportlab.pdfgen import canvas

    from core.settings import SplitterSettings
    from ingestion.chunking.document_chunker import DocumentChunker
    from libs.loader.pdf_loader import PDFLoader

    pdf = tmp_path / "dense_chain.pdf"
    c = canvas.Canvas(str(pdf))
    c.setFont("Helvetica", 14)
    c.drawString(72, 800, "RAG Overview")
    c.drawString(72, 770, "Dense retrieval finds semantic neighbors.")
    c.save()

    doc = PDFLoader().load(str(pdf))
    chunker = DocumentChunker(settings=SplitterSettings(chunk_size=200, chunk_overlap=0))
    chunks = chunker.chunk(doc)
    assert chunks

    emb = FakeEmbedding(dimensions=8)
    vectors = DenseEncoder(emb, batch_size=2).encode(chunks)
    assert len(vectors) == len(chunks)
    assert all(len(v) == 8 for v in vectors)
