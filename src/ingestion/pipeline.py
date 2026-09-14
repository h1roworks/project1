"""Ingestion Pipeline 主流程编排（C14：MVP 串起来）。

把 C1~C13 各阶段零件串成一条完整的离线摄取链路：

    FileIntegrity → Loader → Splitter → Transform → Embed → Upsert
      （检查/增量）   （加载）   （切分）   （增强）   （编码）  （存储）

对齐 DEV_SPEC：
- 5.3.4 pipeline.py：串行执行、异常处理、增量更新；支持 ``on_progress`` 回调；
  统一使用 ``core.types`` 的数据契约；
- 3.1.1 "Pipeline 进度回调"：``on_progress(stage_name, current, total)``，
  ``None`` 时行为与不传完全一致，不影响 CLI 与测试；
- 3.1.1 "文档生命周期管理"：摄取成功后 FileIntegrity 记录 success 状态，
  支持零成本增量跳过；异常时记录 failed 供 Dashboard 展示。

**collection 的 MVP 边界**：``run()`` 的 collection 用于 BM25 索引文件
（``{bm25_dir}/{collection}.pkl``）与图片目录（``data/images/{collection}/``）；
向量库仍写入 settings.vector_store.collection 绑定的集合（ChromaStore 构造时
绑定集合，这是既有接口限制，MVP 阶段不做跨集合向量拆分）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from core.settings import Settings
from core.trace import TraceCollector, TraceContext
from core.types import Document, ImageRef
from ingestion.chunking.document_chunker import DocumentChunker
from ingestion.embedding.batch_processor import BatchProcessor
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from ingestion.transform.base_transform import BaseTransform
from ingestion.transform.chunk_refiner import ChunkRefiner
from ingestion.transform.image_captioner import ImageCaptioner
from ingestion.transform.metadata_enricher import MetadataEnricher
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.loader.base_loader import BaseLoader
from libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker
from libs.loader.pdf_loader import PDFLoader
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory
from observability.logger import write_trace

DEFAULT_COLLECTION = "default"
DEFAULT_BM25_DIR = "data/db/bm25"

# 图片扩展名 → MIME 类型（图片落盘时推断格式，避免一律存成 PNG）
_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


@dataclass
class IngestionResult:
    """一次 ``run()`` 的摄取结果，供日志 / 测试 / Dashboard 展示。

    - ``skipped``: 文件哈希已成功处理过 → 零成本跳过（增量更新）。
    - ``stages``: 各阶段统计（数量 + 耗时 ms），结构见 pipeline 模块 docstring。
    - ``error``: 失败原因（成功为 None）。异常时不重新抛出，由调用方读取。
    """

    source_path: str
    collection: str
    file_hash: str
    skipped: bool = False
    doc_id: str = ""
    total_chunks: int = 0
    total_images: int = 0
    stages: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "collection": self.collection,
            "file_hash": self.file_hash,
            "skipped": self.skipped,
            "doc_id": self.doc_id,
            "total_chunks": self.total_chunks,
            "total_images": self.total_images,
            "stages": self.stages,
            "error": self.error,
        }


class IngestionPipeline:
    """串行编排 load → split → transform → embed → upsert 的摄取流水线。

    支持两种装配方式（对齐既有模块的可注入风格）：
    - **配置装配**：只传 ``settings``，其余组件经工厂自动创建；
    - **注入装配**：显式传入组件实例（测试传 Fake，生产可部分注入覆盖）。
    未注入且无法从 settings 装配的组件（embedding/vector_store）在真正用到时
    才失败（见 ``_encode`` / ``_upsert``），空文档等无需该组件的场景不受影响。
    """

    name = "ingestion_pipeline"

    def __init__(
        self,
        settings: Settings | None = None,
        loaders: list[BaseLoader] | None = None,
        embedding: BaseEmbedding | None = None,
        vector_store: BaseVectorStore | None = None,
        bm25_indexer: BM25Indexer | None = None,
        integrity: FileIntegrityChecker | None = None,
        image_storage: ImageStorage | None = None,
        transforms: list[BaseTransform] | None = None,
        bm25_dir: str | Path = DEFAULT_BM25_DIR,
        trace_writer: Callable[[dict[str, Any]], None] = write_trace,
    ) -> None:
        """初始化。

        Args:
            settings: 完整 Settings；提供时用于经工厂装配 embedding/vector_store，
                并读取 splitter/transform 子段配置。
            loaders: 文档 Loader 列表（按 ``can_handle`` 匹配文件类型）。
                缺省 ``[PDFLoader()]``。
            embedding: BaseEmbedding 实现；缺省经 EmbeddingFactory 从 settings 创建。
            vector_store: BaseVectorStore 实现；缺省经 VectorStoreFactory 从 settings 创建。
            bm25_indexer: BM25Indexer 实例；注入后跨 collection 复用（测试用），
                缺省按 ``bm25_dir/{collection}.pkl`` 懒创建并自动持久化。
            integrity: 文件完整性检查器；缺省 SQLiteIntegrityChecker（默认库路径）。
            image_storage: 图片存储；缺省 ImageStorage（默认目录/库路径）。
            transforms: 增强链；缺省按 settings 装配
                ChunkRefiner → MetadataEnricher → ImageCaptioner。
            bm25_dir: 未注入 bm25_indexer 时的 BM25 索引落盘目录。
        """
        self._settings = settings
        self._loaders = loaders or [PDFLoader()]
        self._integrity = integrity or SQLiteIntegrityChecker()
        self._image_storage = image_storage or ImageStorage()
        self._default_collection = (
            settings.vector_store.collection
            if settings is not None and settings.vector_store.collection
            else DEFAULT_COLLECTION
        )

        self._vector_store = vector_store
        if self._vector_store is None and settings is not None:
            self._vector_store = VectorStoreFactory.create(settings.vector_store)

        self._chunker = DocumentChunker(settings.splitter if settings is not None else None)
        self._transforms = transforms or [
            ChunkRefiner(settings),
            MetadataEnricher(settings),
            ImageCaptioner(settings),
        ]

        self._embedding = embedding
        if self._embedding is None and settings is not None:
            self._embedding = EmbeddingFactory.create(settings.embedding)
        self._dense_encoder = (
            DenseEncoder(self._embedding, settings=settings)
            if self._embedding is not None
            else None
        )
        self._sparse_encoder = SparseEncoder()
        self._batch_processor = (
            BatchProcessor(self._dense_encoder, self._sparse_encoder)
            if self._dense_encoder is not None
            else None
        )

        self._injected_bm25 = bm25_indexer
        self._bm25_dir = Path(bm25_dir)
        self._bm25_indexers: dict[str, BM25Indexer] = {}
        self._upserters: dict[str, VectorUpserter] = {}
        self._trace_writer = trace_writer

    # ---------- 主流程 ----------

    def run(
        self,
        source_path: str,
        collection: str | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> IngestionResult:
        """摄取单个文件，返回统计结果。

        Args:
            source_path: 待解析文件的路径。
            collection: 目标集合（影响 BM25 文件与图片目录）；None 用
                settings.vector_store.collection（缺省 "default"）。
            on_progress: 进度回调 ``(stage_name, current, total)``；None 不回调。

        Returns:
            IngestionResult。文件未变更时 ``skipped=True``；处理失败时不抛出，
            由 ``error`` 携带原因（并已向 integrity 记录 failed 状态）。

        Raises:
            FileNotFoundError: 文件不存在（不进完整性表）。
        """
        path = str(source_path)
        if not Path(path).exists():
            raise FileNotFoundError(f"文件不存在: {source_path}")

        collection = collection or self._default_collection
        file_hash = self._integrity.compute_sha256(path)
        trace = TraceContext(trace_type="ingestion")
        stages: dict[str, Any] = {}
        current_stage = "load"
        stage_started = perf_counter()
        try:
            if self._integrity.should_skip(file_hash):
                _record_trace_stage(
                    trace,
                    "load",
                    0.0,
                    method="integrity_check",
                    provider=type(self._integrity).__name__,
                    details={"skipped": True, "reason": "already_ingested"},
                )
                return IngestionResult(
                    source_path=path,
                    collection=collection,
                    file_hash=file_hash,
                    skipped=True,
                )

            self._integrity.mark_processing(file_hash, path)

            # ---- load ----
            self._report(on_progress, "load", 0, 1)
            stage_started = perf_counter()
            doc = self._load(path)
            self._report(on_progress, "load", 1, 1)
            elapsed_ms = _ms(stage_started)
            stages["load"] = {
                "doc_id": doc.id,
                "images": len(doc.metadata.get("images", [])),
                "latency_ms": elapsed_ms,
            }
            _record_trace_stage(
                trace,
                "load",
                elapsed_ms,
                method=str(doc.metadata.get("loader", "loader")),
                provider=type(next(
                    loader for loader in self._loaders if loader.can_handle(path)
                )).__name__,
                details=stages["load"],
            )

            # ---- split ----
            current_stage = "split"
            stage_started = perf_counter()
            chunks = self._chunker.chunk(doc, trace=trace)
            # Chroma is a single physical collection in this MVP.  Retaining
            # the logical collection in each chunk's metadata lets the
            # DocumentManager browse and delete documents safely by collection.
            for chunk in chunks:
                chunk.metadata["collection"] = collection
            self._report(on_progress, "split", 1, 1)
            elapsed_ms = _ms(stage_started)
            stages["split"] = {"chunks": len(chunks), "latency_ms": elapsed_ms}
            splitter = self._chunker._splitter
            _record_trace_stage(
                trace,
                "split",
                elapsed_ms,
                method=getattr(splitter, "strategy", "splitter"),
                provider=type(splitter).__name__,
                details=stages["split"],
            )

            # ---- transform ----
            current_stage = "transform"
            stage_started = perf_counter()
            enriched = chunks
            for idx, transform in enumerate(self._transforms, start=1):
                enriched = transform.transform_many(enriched, trace=trace)
                self._report(on_progress, "transform", idx, len(self._transforms))
            elapsed_ms = _ms(stage_started)
            stages["transform"] = {
                "applied": len(self._transforms),
                "latency_ms": elapsed_ms,
            }
            _record_trace_stage(
                trace,
                "transform",
                elapsed_ms,
                method="transform_chain",
                provider=",".join(
                    getattr(transform, "name", type(transform).__name__)
                    for transform in self._transforms
                ),
                details={**stages["transform"], "chunk_count": len(enriched)},
            )

            # ---- embed ----
            current_stage = "embed"
            stage_started = perf_counter()
            records = self._encode(enriched, on_progress, stages, trace)
            _record_trace_stage(
                trace,
                "embed",
                stages["embed"]["latency_ms"],
                method="dense_sparse",
                provider=getattr(
                    self._embedding, "provider", type(self._embedding).__name__
                ),
                details=stages["embed"],
            )

            # ---- upsert（含图片落盘） ----
            current_stage = "upsert"
            stage_started = perf_counter()
            images_saved = self._save_images(doc, collection, file_hash)
            self._upsert(records, collection, images_saved, on_progress, stages, trace)
            stages["upsert"]["latency_ms"] = _ms(stage_started)
            _record_trace_stage(
                trace,
                "upsert",
                stages["upsert"]["latency_ms"],
                method="vector_bm25_upsert",
                provider=getattr(
                    self._vector_store, "provider", type(self._vector_store).__name__
                ),
                details=stages["upsert"],
            )

            self._integrity.mark_success(file_hash, path, len(chunks))
            return IngestionResult(
                source_path=path,
                collection=collection,
                file_hash=file_hash,
                skipped=False,
                doc_id=doc.id,
                total_chunks=len(chunks),
                total_images=images_saved,
                stages=stages,
            )
        except Exception as exc:  # noqa: BLE001 - 兜底：记录 failed 并返回错误结果
            _record_trace_stage(
                trace,
                current_stage,
                _ms(stage_started),
                method="failed",
                provider="pipeline",
                details={"error": str(exc)},
            )
            self._integrity.mark_failed(file_hash, str(exc))
            return IngestionResult(
                source_path=path,
                collection=collection,
                file_hash=file_hash,
                skipped=False,
                stages=stages,
                error=str(exc),
            )
        finally:
            try:
                TraceCollector(persist=self._trace_writer).collect(trace)
            except Exception:
                # 日志目录不可写等观测故障，不能影响已完成的文档摄取。
                pass

    # ---------- 阶段执行 ----------

    def _load(self, path: str) -> Document:
        """按扩展名选择 Loader 解析文件；无匹配 Loader 时抛 ValueError。"""
        loader = next((l for l in self._loaders if l.can_handle(path)), None)
        if loader is None:
            suffixes = [l.supported_extensions for l in self._loaders]
            raise ValueError(f"不支持的文档类型: {Path(path).suffix}，可用 Loader: {suffixes}")
        return loader.load(path)

    def _encode(
        self,
        chunks: list[Any],
        on_progress: Callable[[str, int, int], None] | None,
        stages: dict[str, Any],
        trace: TraceContext,
    ) -> list[Any]:
        """Dense + Sparse 双路编码（逐批上报进度），返回 ChunkRecord 列表。"""
        if not chunks:
            batch_size = self._batch_processor.batch_size if self._batch_processor else 0
            stages["embed"] = {"records": 0, "batch_size": batch_size, "latency_ms": 0}
            return []
        if self._batch_processor is None:
            raise ValueError(
                "无法进行 embedding：未注入 embedding，且 settings 未配置 embedding.provider"
            )

        t0 = perf_counter()
        records: list[Any] = []
        ranges = self._batch_processor.batch_ranges(len(chunks))
        for idx, (start, end) in enumerate(ranges, start=1):
            records.extend(self._batch_processor.process(chunks[start:end], trace=trace))
            self._report(on_progress, "embed", idx, len(ranges))
        stages["embed"] = {
            "records": len(records),
            "batch_size": self._batch_processor.batch_size,
            "latency_ms": _ms(t0),
        }
        return records

    def _save_images(self, doc: Document, collection: str, file_hash: str) -> int:
        """把 Document 引用的图片落盘到 ImageStorage，返回成功保存的图片数。

        当前 PDFLoader 产出 ``metadata.images = []``，此环节自然空转；
        为未来 Loader 升级预留（引用文件缺失时跳过，不阻塞摄取）。
        """
        entries = doc.metadata.get("images", [])
        if not entries:
            return 0

        saved = 0
        for entry in entries:
            ref = _to_image_ref(entry)
            if ref is None or not ref.path:
                continue
            src = Path(ref.path)
            if not src.exists():
                continue
            try:
                mime_type = _MIME_BY_EXT.get(src.suffix.lower(), "image/png")
                self._image_storage.save_image(
                    ref.id,
                    src.read_bytes(),
                    collection=collection,
                    doc_hash=file_hash,
                    page_num=ref.page,
                    mime_type=mime_type,
                )
                saved += 1
            except OSError:  # noqa: BLE001 - 单图失败跳过，不影响其余
                continue
        return saved

    def _upsert(
        self,
        records: list[Any],
        collection: str,
        images_saved: int,
        on_progress: Callable[[str, int, int], None] | None,
        stages: dict[str, Any],
        trace: TraceContext,
    ) -> None:
        """把 ChunkRecord 写入向量库 + BM25（逐批上报进度），并填入 stages。"""
        if not records:
            stages["upsert"] = {
                "vector_store_count": 0,
                "bm25_count": 0,
                "images_saved": images_saved,
            }
            return
        if self._vector_store is None:
            raise ValueError(
                "无法写入向量库：未注入 vector_store，且 settings 未配置 vector_store"
            )

        upserter = self._upserters.get(collection)
        if upserter is None:
            bm25 = self._bm25_indexers.get(collection) or self._get_bm25_indexer(collection)
            upserter = VectorUpserter(self._vector_store, bm25)
            self._upserters[collection] = upserter

        ranges = self._batch_processor.batch_ranges(len(records)) if self._batch_processor else [(0, len(records))]
        vector_count = 0
        bm25_count = 0
        for idx, (start, end) in enumerate(ranges, start=1):
            result = upserter.upsert(records[start:end], trace=trace)
            vector_count += result.vector_store_count
            bm25_count += result.bm25_count
            self._report(on_progress, "upsert", idx, len(ranges))
        stages["upsert"] = {
            "vector_store_count": vector_count,
            "bm25_count": bm25_count,
            "images_saved": images_saved,
        }

    def _get_bm25_indexer(self, collection: str) -> BM25Indexer:
        """按 collection 返回（懒创建）BM25Indexer；注入的实例跨 collection 复用。"""
        if self._injected_bm25 is not None:
            self._bm25_indexers[collection] = self._injected_bm25
            return self._injected_bm25
        indexer = self._bm25_indexers.get(collection)
        if indexer is None:
            indexer = BM25Indexer(index_path=self._bm25_dir / f"{collection}.pkl")
            self._bm25_indexers[collection] = indexer
        return indexer

    # ---------- 进度回调 ----------

    @staticmethod
    def _report(
        on_progress: Callable[[str, int, int], None] | None,
        stage: str,
        current: int,
        total: int,
    ) -> None:
        """回调进度；观测层回调失败不能中断文档摄取。"""
        if on_progress is None:
            return
        try:
            on_progress(stage, current, total)
        except Exception:
            pass


# ---------- 私有工具函数 ----------


def _ms(t0: float) -> int:
    """距 ``t0`` 的毫秒耗时（整型，便于日志/展示）。"""
    return int((perf_counter() - t0) * 1000)


def _record_trace_stage(
    trace: TraceContext,
    stage_name: str,
    elapsed_ms: float,
    *,
    method: str,
    provider: str,
    details: dict[str, Any],
) -> None:
    """记录阶段数据；观测功能本身不能使摄取链路失败。"""
    try:
        trace.record_stage(
            stage_name,
            elapsed_ms,
            method=method,
            provider=provider,
            details=details,
        )
    except Exception:
        pass


def _to_image_ref(entry: Any) -> ImageRef | None:
    """把 ``metadata.images`` 条目规整为 ImageRef；非法条目返回 None。"""
    if isinstance(entry, ImageRef):
        return entry
    if isinstance(entry, dict):
        try:
            return ImageRef.from_dict(entry)
        except Exception:  # noqa: BLE001 - 脏条目跳过，不阻塞
            return None
    return None
