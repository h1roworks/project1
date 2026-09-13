"""BM25 稀疏索引构建（C11：BM25Indexer）。

对 C10 产出的 ChunkRecord（含 ``sparse_vector={term: tf}``）建立 BM25 检索所需的
倒排索引与 IDF 统计，并把索引状态持久化到本地文件（pickle）。

对齐 DEV_SPEC：
- 3.1.1 "Upsert & Storage"：存储倒排索引 + 文档长度 + 文档计数，为 D3
  SparseRetriever 的 BM25 打分提供数据基础；
- 3.1.1 "文档生命周期管理"：提供 ``remove_document(source)`` 支持跨存储删除；
- 存储说明（"BM25 索引元数据"）：当前使用 pickle 持久化到 ``data/db/bm25/``，
  未来可迁移至 SQLite。

**范围约定**：本模块只负责"建索引 + 算 IDF + 持久化"，BM25 打分/召回属于
D3 SparseRetriever（本模块暴露 postings / 文档长度 / avgdl / idf 供其消费）。
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

# 默认 BM25 参数：k1 控制词频饱和，b 控制文档长度归一化强度。
_DEFAULT_K1 = 1.5
_DEFAULT_B = 0.75
_VERSION = 1


class BM25Indexer:
    """BM25 倒排索引：term → {chunk_id: tf} + 文档长度表，支持幂等写入与持久化。

    Args:
        index_path: 索引 pickle 文件路径（DEV_SPEC 约定 ``data/db/bm25/`` 下）。
            提供且文件已存在时，构造时自动加载既有索引（增量场景）。
        k1: BM25 词频饱和参数（默认 1.5）。
        b: BM25 文档长度归一化参数（默认 0.75）。
    """

    name = "bm25_indexer"

    def __init__(
        self,
        index_path: str | Path | None = None,
        k1: float = _DEFAULT_K1,
        b: float = _DEFAULT_B,
    ) -> None:
        self.index_path = Path(index_path) if index_path else None
        self.k1 = k1
        self.b = b
        # 倒排索引：term -> {chunk_id: tf}。tf 为整型词频（来自 sparse_vector）。
        self.postings: dict[str, dict[str, int]] = {}
        # 文档表：chunk_id -> {"source": 所属文档路径, "doc_len": 词项总数}。
        self.documents: dict[str, dict[str, Any]] = {}
        if self.index_path is not None and self.index_path.exists():
            self._load_from_file(self.index_path)

    # ---------- 索引构建（幂等 Upsert） ----------

    def add(self, records: list[Any]) -> int:
        """把 ChunkRecord 列表写入索引（Upsert 语义），返回处理条数。

        同一 ``chunk_id`` 已存在时先移除旧条目再写入，保证重复摄取不产生重复索引。
        ``sparse_vector`` 为 ``None`` 或空 dict 的 chunk 仍记录进文档表
        （doc_len=0，用于文档计数 N），但不产生任何倒排词项。
        """
        for record in records:
            if record.id in self.documents:
                self._remove_chunk(record.id)
            sparse = record.sparse_vector or {}
            doc_len = sum(tf for tf in sparse.values() if tf > 0)
            self.documents[record.id] = {
                "source": record.metadata.get("source_path", ""),
                "doc_len": doc_len,
            }
            for term, tf in sparse.items():
                if tf <= 0:
                    continue
                self.postings.setdefault(term, {})[record.id] = tf
        return len(records)

    def remove_document(self, source: str) -> int:
        """按文档路径移除该文档的全部 chunk 索引条目，返回移除条数。"""
        chunk_ids = [
            cid
            for cid, info in self.documents.items()
            if info.get("source") == source
        ]
        for cid in chunk_ids:
            self._remove_chunk(cid)
        return len(chunk_ids)

    def _remove_chunk(self, chunk_id: str) -> None:
        """从文档表与倒排索引中移除单个 chunk，并清理空词项。"""
        self.documents.pop(chunk_id, None)
        for term_postings in self.postings.values():
            term_postings.pop(chunk_id, None)
        self.postings = {
            term: term_postings
            for term, term_postings in self.postings.items()
            if term_postings
        }

    # ---------- IDF / 统计 ----------

    @property
    def total_docs(self) -> int:
        """已索引的文档（chunk）总数 N，参与 IDF 计算。"""
        return len(self.documents)

    @property
    def avgdl(self) -> float:
        """平均文档长度，参与 BM25 长度归一化。无文档时返回 0。"""
        if not self.documents:
            return 0.0
        total_len = sum(info["doc_len"] for info in self.documents.values())
        return total_len / len(self.documents)

    def df(self, term: str) -> int:
        """词项 t 的文档频率（出现在多少个 chunk 中）。"""
        return len(self.postings.get(term, {}))

    def idf(self, term: str) -> float:
        """词项 t 的 IDF 权重；未出现在索引中的词项返回 0。

        采用平滑版公式（恒非负，避免经典公式 df 过大时出现负值）：
        ``IDF(t) = ln(1 + (N - df + 0.5) / (df + 0.5))``
        """
        df = self.df(term)
        if df == 0:
            return 0.0
        n = self.total_docs
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def query(
        self,
        keywords: list[str] | dict[str, float],
        top_k: int = 10,
        trace: Any = None,
    ) -> list[dict[str, float | str]]:
        """使用 BM25 对倒排索引中的 chunk 进行关键词召回。

        ``keywords`` 可以是普通关键词列表，也可以是 ``{词: 权重}`` 映射。
        后者供 QueryProcessor 的同义词扩展使用：原始词可保持更高权重，
        扩展词只影响一次稀疏检索，不会额外发起多个查询。

        返回结果只包含 ``chunk_id`` 和 ``score``。正文与元数据由 D3 的
        SparseRetriever 通过 VectorStore.get_by_ids() 补齐。
        """
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not self.documents:
            return []

        term_weights = self._normalize_query_terms(keywords)
        if not term_weights:
            return []

        avgdl = self.avgdl
        if avgdl <= 0:
            return []

        scores: dict[str, float] = {}
        for term, query_weight in term_weights.items():
            idf = self.idf(term)
            if idf <= 0:
                continue
            for chunk_id, term_frequency in self.postings.get(term, {}).items():
                doc_len = self.documents[chunk_id]["doc_len"]
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * doc_len / avgdl
                )
                scores[chunk_id] = scores.get(chunk_id, 0.0) + query_weight * idf * (
                    term_frequency * (self.k1 + 1) / denominator
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            {"chunk_id": chunk_id, "score": score}
            for chunk_id, score in ranked[:top_k]
        ]

    @staticmethod
    def _normalize_query_terms(
        keywords: list[str] | dict[str, float],
    ) -> dict[str, float]:
        """去除空词、合并重复词，并保留最高查询权重。"""
        if isinstance(keywords, dict):
            normalized: dict[str, float] = {}
            for term, weight in keywords.items():
                if not isinstance(term, str) or not term:
                    continue
                try:
                    numeric_weight = float(weight)
                except (TypeError, ValueError):
                    continue
                if numeric_weight > 0:
                    normalized[term] = numeric_weight
            return normalized

        return {
            term: 1.0
            for term in keywords
            if isinstance(term, str) and term
        }

    def stats(self) -> dict[str, int]:
        """索引规模统计：文档数、词项数、倒排条目总数。"""
        return {
            "total_docs": self.total_docs,
            "total_terms": len(self.postings),
            "total_postings": sum(len(p) for p in self.postings.values()),
        }

    # ---------- 持久化（pickle） ----------

    def save(self, path: str | Path | None = None) -> None:
        """把索引状态序列化到 pickle 文件（路径缺省用构造时的 index_path）。"""
        target = Path(path) if path is not None else self.index_path
        if target is None:
            raise ValueError("未配置 index_path，请显式传入保存路径")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _VERSION,
            "postings": self.postings,
            "documents": self.documents,
        }
        with target.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_from(
        cls,
        path: str | Path,
        k1: float = _DEFAULT_K1,
        b: float = _DEFAULT_B,
    ) -> "BM25Indexer":
        """从 pickle 文件加载既有索引，返回新的 BM25Indexer 实例。"""
        return cls(index_path=path, k1=k1, b=b)

    def _load_from_file(self, path: Path) -> None:
        """从磁盘读取索引状态并覆盖当前内存结构。"""
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        if payload.get("version") != _VERSION:
            raise ValueError(
                f"BM25 索引版本不兼容: 期望 {_VERSION}, 实际 {payload.get('version')}"
            )
        self.postings = payload["postings"]
        self.documents = payload["documents"]
