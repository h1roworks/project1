"""文件完整性检查（SHA256）—— 增量摄取的前置去重。

原理：在解析文件前计算其 SHA256 哈希指纹，检索 ``ingestion_history`` 表；
若发现相同哈希且状态为 ``success`` 的记录，则认定该文件未变更，
直接跳过后续所有处理（解析、切分、LLM 重写），实现零成本增量更新。

默认使用 SQLite 持久化（``data/db/ingestion_history.db``，WAL 模式），
架构上通过 ``FileIntegrityChecker`` 抽象接口隔离存储细节，
后续可替换为 Redis（分布式缓存）或 PostgreSQL（企业级中心化存储）。

表结构（与 DEV_SPEC 3.1.1 一致）：
- ``file_hash``   TEXT  PRIMARY KEY  文件 SHA256 指纹
- ``file_path``   TEXT  NOT NULL    已处理文件路径
- ``file_size``   INTEGER           文件大小（字节）
- ``status``      TEXT   success/failed/processing 处理状态
- ``processed_at`` TIMESTAMP         最近一次状态变更时间
- ``error_msg``   TEXT               失败原因
- ``chunk_count`` INTEGER           成功时产出的 chunk 数量
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = "data/db/ingestion_history.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingestion_history (
    file_hash TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_size INTEGER,
    status TEXT NOT NULL CHECK(status IN ('success', 'failed', 'processing')),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_msg TEXT,
    chunk_count INTEGER
);
CREATE INDEX IF NOT EXISTS idx_status ON ingestion_history(status);
CREATE INDEX IF NOT EXISTS idx_processed_at ON ingestion_history(processed_at);
"""


class FileIntegrityChecker(ABC):
    """文件完整性检查的抽象接口。

    只约定"做什么"，不关心"用什么存"：计算哈希、判定是否跳过、记录状态。
    后续新增 Redis/PostgreSQL 实现时只需继承本类并实现这些方法。
    """

    @abstractmethod
    def compute_sha256(self, path: str) -> str:
        """计算文件内容的 SHA256 指纹（十六进制字符串）。"""
        raise NotImplementedError

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """该哈希是否已成功处理过（是则跳过解析，实现增量摄取）。"""
        raise NotImplementedError

    @abstractmethod
    def get_status(self, file_hash: str) -> str | None:
        """查询某哈希当前处理状态，未知哈希返回 None。"""
        raise NotImplementedError

    @abstractmethod
    def mark_processing(self, file_hash: str, file_path: str) -> None:
        """开始处理：写入/更新一条 processing 记录。"""
        raise NotImplementedError

    @abstractmethod
    def mark_success(self, file_hash: str, file_path: str, chunk_count: int = 0) -> None:
        """处理成功：将状态置为 success 并记录 chunk 数。"""
        raise NotImplementedError

    @abstractmethod
    def mark_failed(self, file_hash: str, error_msg: str = "") -> None:
        """处理失败：将状态置为 failed 并记录错误信息。"""
        raise NotImplementedError

    @abstractmethod
    def remove_record(self, file_hash: str) -> int:
        """删除某哈希的处理记录，使文件可重新摄入（供 DocumentManager 协调删除）。"""
        raise NotImplementedError

    @abstractmethod
    def list_processed(self) -> list[dict[str, Any]]:
        """列出全部处理记录（按处理时间倒序），供管理/展示使用。"""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """释放底层资源（如数据库连接）。"""
        raise NotImplementedError


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """SQLite 默认实现：WAL 模式支持多进程并发安全读写。"""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---------- 哈希计算 ----------

    def compute_sha256(self, path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):  # 分块读取，避免大文件占满内存
                digest.update(block)
        return digest.hexdigest()

    # ---------- 查询 ----------

    def should_skip(self, file_hash: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM ingestion_history WHERE file_hash = ? AND status = 'success'",
            (file_hash,),
        ).fetchone()
        return row is not None

    def get_status(self, file_hash: str) -> str | None:
        row = self._conn.execute(
            "SELECT status FROM ingestion_history WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
        return row["status"] if row else None

    # ---------- 状态流转 ----------

    def mark_processing(self, file_hash: str, file_path: str) -> None:
        self._conn.execute(
            """
            INSERT INTO ingestion_history (file_hash, file_path, file_size, status)
            VALUES (?, ?, ?, 'processing')
            ON CONFLICT(file_hash) DO UPDATE SET
                file_path = excluded.file_path,
                file_size = excluded.file_size,
                status = 'processing',
                error_msg = NULL,
                chunk_count = NULL
            """,
            (file_hash, file_path, _file_size_or_none(file_path)),
        )
        self._conn.commit()

    def mark_success(self, file_hash: str, file_path: str, chunk_count: int = 0) -> None:
        self._conn.execute(
            """
            INSERT INTO ingestion_history (file_hash, file_path, file_size, status, chunk_count)
            VALUES (?, ?, ?, 'success', ?)
            ON CONFLICT(file_hash) DO UPDATE SET
                file_path = excluded.file_path,
                file_size = excluded.file_size,
                status = 'success',
                error_msg = NULL,
                chunk_count = excluded.chunk_count
            """,
            (file_hash, file_path, _file_size_or_none(file_path), chunk_count),
        )
        self._conn.commit()

    def mark_failed(self, file_hash: str, error_msg: str = "") -> None:
        self._conn.execute(
            """
            INSERT INTO ingestion_history (file_hash, file_path, status, error_msg)
            VALUES (?, '', 'failed', ?)
            ON CONFLICT(file_hash) DO UPDATE SET
                status = 'failed',
                error_msg = excluded.error_msg
            """,
            (file_hash, error_msg),
        )
        self._conn.commit()

    # ---------- 生命周期管理（供 DocumentManager 复用） ----------

    def remove_record(self, file_hash: str) -> int:
        cur = self._conn.execute(
            "DELETE FROM ingestion_history WHERE file_hash = ?", (file_hash,)
        )
        self._conn.commit()
        return cur.rowcount

    def list_processed(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM ingestion_history ORDER BY processed_at DESC, rowid DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    # ---------- 资源释放 ----------

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None


def _file_size_or_none(file_path: str) -> int | None:
    """读取文件大小；文件缺失时返回 None 而非抛错（不阻塞状态记录）。"""
    try:
        return os.path.getsize(file_path)
    except OSError:
        return None
