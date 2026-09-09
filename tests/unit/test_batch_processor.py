"""C10: BatchProcessor 单元测试。

验证：
- 输出契约：process(chunks) 返回等长、顺序对齐的 ChunkRecord 列表
- ChunkRecord 组装：id/text/metadata 保留，dense_vector/sparse_vector 填实
- 分批控制：batch_size 生效，Dense/Sparse 按批调用且批次顺序稳定
- 默认批大小：继承 dense_encoder.batch_size
- 边界行为：空 chunks、输入不被修改、双路输出与批次不匹配时报错
- 与 Chunker 的链路（C4 → C8 → C9 → C10）
"""

import hashlib

import pytest

from core.types import Chunk, ChunkRecord
from ingestion.embedding.batch_processor import BatchProcessor
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder
from libs.embedding.base_embedding import BaseEmbedding

# ---------- fixtures / helpers ----------


class FakeEmbedding(BaseEmbedding):
    """可注入的假 Embedding：返回内容长度派生的确定性向量，并记录每次调用。"""

    provider = "c10fake"

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


def make_processor(
    dimensions: int = 4,
    batch_size: int | None = None,
    dense_batch_size: int | None = None,
) -> tuple[BatchProcessor, FakeEmbedding]:
    """构造 BatchProcessor + 其内部 FakeEmbedding（供断言调用记录）。"""
    emb = FakeEmbedding(dimensions=dimensions)
    dense = DenseEncoder(emb, batch_size=dense_batch_size)
    sparse = SparseEncoder()
    return BatchProcessor(dense, sparse, batch_size=batch_size), emb


# ---------- 输出契约：等长 / 顺序对齐 ----------


def test_output_count_matches_input() -> None:
    proc, _ = make_processor()
    chunks = [make_chunk(f"chunk {i} content", index=i) for i in range(5)]
    records = proc.process(chunks)
    assert len(records) == 5  # ChunkRecord 数量与 chunks 一致
    assert all(isinstance(r, ChunkRecord) for r in records)


def test_output_order_aligned_with_input() -> None:
    proc, emb = make_processor()
    chunks = [make_chunk("first"), make_chunk("second"), make_chunk("third")]
    records = proc.process(chunks)
    assert [r.id for r in records] == [c.id for c in chunks]  # id 顺序一致
    # 稠密向量与 SparseEncoder 单条编码结果一致
    assert records[0].dense_vector == emb.embed(["first"])[0]
    assert records[1].dense_vector == emb.embed(["second"])[0]


def test_chunk_record_fields_preserved() -> None:
    proc, _ = make_processor()
    chunk = make_chunk("alpha beta gamma", index=3)
    records = proc.process([chunk])
    rec = records[0]
    assert rec.id == chunk.id  # 原 id 保留
    assert rec.text == chunk.text  # 原文保留
    assert rec.metadata == chunk.metadata  # metadata 保留（含 source_path/chunk_index）
    assert isinstance(rec.dense_vector, list)  # 稠密向量填实
    assert rec.sparse_vector == {"alpha": 1, "beta": 1, "gamma": 1}  # 关键词权重填实


def test_input_chunks_not_mutated() -> None:
    chunks = [make_chunk("正文内容。")]
    before = [(c.id, c.text) for c in chunks]
    proc, _ = make_processor()
    proc.process(chunks)
    assert [(c.id, c.text) for c in chunks] == before


# ---------- 分批控制 ----------


def test_batch_size_splits_encoding() -> None:
    proc, emb = make_processor(batch_size=2)
    chunks = [make_chunk(f"c{i}", index=i) for i in range(5)]
    records = proc.process(chunks)
    assert [len(call) for call in emb.calls] == [2, 2, 1]  # 5 条按 2 一批 → 3 批
    # 批次拼接后与输入顺序一致
    assert [t for call in emb.calls for t in call] == [c.text for c in chunks]
    # 结果仍与输入等长对齐
    assert [r.id for r in records] == [c.id for c in chunks]


def test_batch_ranges_splits_ranges() -> None:
    proc, _ = make_processor(batch_size=3)
    assert proc.batch_ranges(7) == [(0, 3), (3, 6), (6, 7)]
    assert proc.batch_ranges(3) == [(0, 3)]
    assert proc.batch_ranges(0) == []


def test_batch_size_default_from_dense_encoder() -> None:
    proc, emb = make_processor(dense_batch_size=3)
    chunks = [make_chunk(f"c{i}", index=i) for i in range(5)]
    proc.process(chunks)
    assert [len(call) for call in emb.calls] == [3, 2]  # 继承 dense_encoder 的批大小


# ---------- trace 透传 ----------


def test_trace_passed_through() -> None:
    proc, emb = make_processor(batch_size=1)
    chunks = [make_chunk("hello"), make_chunk("world")]
    proc.process(chunks, trace="t0")
    assert emb.traces == ["t0", "t0"]  # 每批透传同一 trace


# ---------- 边界行为 ----------


def test_empty_chunks_returns_empty() -> None:
    proc, emb = make_processor()
    assert proc.process([]) == []
    assert emb.calls == []  # 空输入不触发任何编码


def test_mismatched_output_raises() -> None:
    class OkDense:
        batch_size = 2

        def encode(self, chunks, trace=None) -> list[list[float]]:
            return [[0.0]] * len(chunks)

    class ShortSparse:
        def encode(self, chunks, trace=None) -> list[dict[str, int]]:
            return [{}] * (len(chunks) - 1)  # 少返回一条，破坏对齐

    proc = BatchProcessor(OkDense(), ShortSparse(), batch_size=2)
    with pytest.raises(ValueError, match="长度不一致"):
        proc.process([make_chunk("a"), make_chunk("b")])


def test_whitespace_text_still_aligns() -> None:
    proc, _ = make_processor()
    chunks = [make_chunk("   \n\t "), make_chunk("real content")]
    records = proc.process(chunks)
    assert len(records) == 2
    assert records[0].sparse_vector == {}  # 空白文本 → 空关键词权重
    assert records[1].sparse_vector != {}


# ---------- 与 Chunker 的链路（C4 → C8 → C9 → C10） ----------


def test_batch_processor_chains_with_document_chunker(tmp_path) -> None:
    """真实 PDF 经 PDFLoader → DocumentChunker → BatchProcessor 产出对齐 ChunkRecord。"""
    from reportlab.pdfgen import canvas

    from core.settings import SplitterSettings
    from ingestion.chunking.document_chunker import DocumentChunker
    from libs.loader.pdf_loader import PDFLoader

    pdf = tmp_path / "batch_chain.pdf"
    c = canvas.Canvas(str(pdf))
    c.setFont("Helvetica", 14)
    c.drawString(72, 800, "RAG Overview")
    c.drawString(72, 770, "Batch processor embeds and sparsifies chunks together.")
    c.save()

    doc = PDFLoader().load(str(pdf))
    chunker = DocumentChunker(settings=SplitterSettings(chunk_size=200, chunk_overlap=0))
    chunks = chunker.chunk(doc)
    assert chunks

    emb = FakeEmbedding(dimensions=8)
    proc = BatchProcessor(
        DenseEncoder(emb, batch_size=2), SparseEncoder(), batch_size=2
    )
    records = proc.process(chunks)
    assert len(records) == len(chunks)
    assert all(len(r.dense_vector) == 8 for r in records)
    assert all(r.sparse_vector for r in records)  # 每个 chunk 都有关键词权重
    # 原文中的实义词应出现在稀疏结构里
    assert any("overview" in r.sparse_vector for r in records)
