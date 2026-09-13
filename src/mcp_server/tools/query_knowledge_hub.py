"""MCP tool that runs HybridSearch + Reranker over the local knowledge base."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import Fusion
from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.reranker import Reranker
from core.query_engine.sparse_retriever import SparseRetriever
from core.response.multimodal_assembler import MultimodalAssembler
from core.response.response_builder import MCPResponse, ResponseBuilder
from core.settings import Settings, load_settings
from ingestion.storage.bm25_indexer import BM25Indexer


QUERY_KNOWLEDGE_HUB_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "要在知识库中检索的完整问题。",
        },
        "top_k": {
            "type": "integer",
            "minimum": 1,
            "description": "返回的最多片段数，默认使用配置中的 retrieval.top_k。",
        },
        "collection": {
            "type": "string",
            "description": "可选的知识库集合名称。",
        },
    },
    "required": ["query"],
}

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"
_BM25_DIRECTORY = _PROJECT_ROOT / "data" / "db" / "bm25"
_IMAGES_DIRECTORY = _PROJECT_ROOT / "data" / "images"
_IMAGE_INDEX_PATH = _PROJECT_ROOT / "data" / "db" / "image_index.db"


def build_query_engine(
    settings: Settings, collection: str | None = None
) -> tuple[HybridSearch, Reranker]:
    """Assemble the existing D1--D6 query pipeline for one collection."""
    collection_name = collection or settings.vector_store.collection
    sparse_retriever = SparseRetriever(
        settings,
        bm25_indexer=BM25Indexer(_BM25_DIRECTORY / f"{collection_name}.pkl"),
    )
    hybrid_search = HybridSearch(
        settings,
        query_processor=QueryProcessor(),
        dense_retriever=DenseRetriever(settings),
        sparse_retriever=sparse_retriever,
        fusion=Fusion(k=settings.retrieval.fusion_k),
    )
    return hybrid_search, Reranker(settings)


class QueryKnowledgeHubTool:
    """Lazily execute the project query pipeline and format its MCP response."""

    def __init__(
        self,
        *,
        hybrid_search: HybridSearch | None = None,
        reranker: Reranker | None = None,
        default_top_k: int = 10,
        engine_factory: Callable[[str | None], tuple[HybridSearch, Reranker]] | None = None,
        response_builder: ResponseBuilder | None = None,
        multimodal_assembler: MultimodalAssembler | None = None,
    ) -> None:
        if (hybrid_search is None) != (reranker is None):
            raise ValueError("hybrid_search and reranker must be provided together.")
        if hybrid_search is None and engine_factory is None:
            raise ValueError("Provide query components or an engine_factory.")
        if not isinstance(default_top_k, int) or isinstance(default_top_k, bool) or default_top_k <= 0:
            raise ValueError("default_top_k must be a positive integer.")

        self._static_engine = (hybrid_search, reranker) if hybrid_search is not None else None
        self._engine_factory = engine_factory
        self._engines: dict[str | None, tuple[HybridSearch, Reranker]] = {}
        self.default_top_k = default_top_k
        self.response_builder = response_builder or ResponseBuilder()
        self.multimodal_assembler = multimodal_assembler or MultimodalAssembler(
            storage_factory=lambda: _build_image_storage()
        )

    @classmethod
    def from_config(cls, config_path: str | Path = _DEFAULT_CONFIG_PATH) -> "QueryKnowledgeHubTool":
        """Create a tool whose expensive retrieval clients are initialized on use."""
        settings = load_settings(config_path)
        return cls(
            default_top_k=settings.retrieval.top_k,
            engine_factory=lambda collection: build_query_engine(settings, collection),
        )

    def __call__(
        self,
        query: str,
        top_k: int | None = None,
        collection: str | None = None,
    ) -> MCPResponse:
        """Run the query and return Markdown plus structured citations."""
        normalized_query = self._validate_query(query)
        limit = self._validate_top_k(top_k)
        normalized_collection = self._validate_collection(collection)
        hybrid_search, reranker = self._get_engine(normalized_collection)
        filters = {"collection": normalized_collection} if normalized_collection else None
        candidates = hybrid_search.search(normalized_query, top_k=limit, filters=filters)
        results = reranker.rerank(normalized_query, candidates)
        selected_results = results[:limit]
        response = self.response_builder.build(selected_results, normalized_query)
        return self.multimodal_assembler.assemble(response, selected_results)

    def _get_engine(self, collection: str | None) -> tuple[HybridSearch, Reranker]:
        if self._static_engine is not None:
            return self._static_engine
        if collection not in self._engines:
            assert self._engine_factory is not None
            self._engines[collection] = self._engine_factory(collection)
        return self._engines[collection]

    def _validate_top_k(self, top_k: int | None) -> int:
        if top_k is None:
            return self.default_top_k
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        return top_k

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        return query.strip()

    @staticmethod
    def _validate_collection(collection: str | None) -> str | None:
        if collection is None:
            return None
        if not isinstance(collection, str) or not collection.strip():
            raise ValueError("collection must be a non-empty string when supplied.")
        return collection.strip()


def query_knowledge_hub(
    query: str,
    top_k: int | None = None,
    collection: str | None = None,
) -> MCPResponse:
    """Convenience entry point for direct, one-off use outside the MCP server."""
    return QueryKnowledgeHubTool.from_config()(query, top_k, collection)


def _build_image_storage():
    """Open the project image index only when a query actually references images."""
    from ingestion.storage.image_storage import ImageStorage

    return ImageStorage(images_dir=_IMAGES_DIRECTORY, db_path=_IMAGE_INDEX_PATH)
