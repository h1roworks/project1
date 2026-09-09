"""图片文件存储 + SQLite 索引（C13：ImageStorage）。

把图片二进制落盘到 ``data/images/{collection}/``，并用 SQLite
（``data/db/image_index.db``）记录 ``image_id → file_path`` 映射，支持按
image_id 快速定位、按 collection/doc_hash 批量查询与协调删除。检索命中
Chunk 后，可依据其 ``image_refs`` 通过本模块定位图片文件，用于 MCP 多模态返回。

对齐 DEV_SPEC：
- 3.1.1 "Upsert & Storage / 原始图片存储"：图片文件落盘 + 独立索引表
  （``image_id, file_path, collection, doc_hash, page_num, created_at``）；
- 3.1.1 "文档生命周期管理"：提供 ``delete_images(collection, doc_hash)``
  供 DocumentManager 跨存储协调删除；
- 5.4.3 管理操作流：``list_images(collection, doc_hash)`` 供 Dashboard 展示。

复用 ``file_integrity.py`` 的 SQLite 架构模式：WAL 模式并发安全、自动建表、
``ON CONFLICT`` 幂等写入（同一 image_id 重复保存覆盖旧记录与文件）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_IMAGES_DIR = "data/images"
DEFAULT_DB_PATH = "data/db/image_index.db"

# mime_type → 文件扩展名；未知 mime 回退为 PNG。
_MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS image_index (
    image_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    collection TEXT,
    doc_hash TEXT,
    page_num INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection);
CREATE INDEX IF NOT EXISTS idx_doc_hash ON image_index(doc_hash);
"""


class ImageStorage:
    """图片落盘 + SQLite 索引映射：image_id → 本地文件路径。"""

    name = "image_storage"

    def __init__(
        self,
        images_dir: str | Path = DEFAULT_IMAGES_DIR,
        db_path: str | Path = DEFAULT_DB_PATH,
    ) -> None:
        """初始化。

        Args:
            images_dir: 图片根目录（约定 ``data/images``），图片按
                ``{collection}/`` 子目录存放。
            db_path: SQLite 索引数据库路径（约定 ``data/db/image_index.db``）。
        """
        self.images_dir = Path(images_dir)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---------- 写入 ----------

    def save_image(
        self,
        image_id: str,
        data: bytes,
        collection: str = "default",
        doc_hash: str | None = None,
        page_num: int | None = None,
        mime_type: str = "image/png",
    ) -> str:
        """保存图片文件并记录索引，返回存储的相对路径（Upsert 幂等）。

        同一 ``image_id`` 重复保存：覆盖磁盘文件与数据库记录，不产生重复条目。

        Args:
            image_id: 全局唯一图片标识（建议 ``{doc_hash}_{page}_{seq}``）。
            data: 图片二进制内容。
            collection: 所属集合，图片存放在 ``{images_dir}/{collection}/`` 下。
            doc_hash: 所属文档哈希，供按文档批量查询/删除。
            page_num: 图片在原文档中的页码（可选）。
            mime_type: 图片 MIME 类型，用于推断文件扩展名（默认 PNG）。
        """
        ext = _MIME_TO_EXT.get(mime_type, ".png")
        file_path = self.images_dir / collection / f"{image_id}{ext}"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(data)
        relative = file_path.as_posix()
        self._conn.execute(
            """
            INSERT INTO image_index (image_id, file_path, collection, doc_hash, page_num)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                file_path = excluded.file_path,
                collection = excluded.collection,
                doc_hash = excluded.doc_hash,
                page_num = excluded.page_num
            """,
            (image_id, relative, collection, doc_hash, page_num),
        )
        self._conn.commit()
        return relative

    # ---------- 查询 ----------

    def get_path(self, image_id: str) -> str | None:
        """按 image_id 返回图片文件路径；未知返回 None。"""
        row = self._conn.execute(
            "SELECT file_path FROM image_index WHERE image_id = ?", (image_id,)
        ).fetchone()
        return row["file_path"] if row else None

    def get_record(self, image_id: str) -> dict[str, Any] | None:
        """按 image_id 返回完整索引记录；未知返回 None。"""
        row = self._conn.execute(
            "SELECT image_id, file_path, collection, doc_hash, page_num "
            "FROM image_index WHERE image_id = ?",
            (image_id,),
        ).fetchone()
        return dict(row) if row else None

    def exists(self, image_id: str) -> bool:
        """该 image_id 是否已登记索引。"""
        return self.get_path(image_id) is not None

    def read_image(self, image_id: str) -> bytes | None:
        """读取图片二进制；未知 image_id 或文件缺失时返回 None。"""
        path = self.get_path(image_id)
        if path is None:
            return None
        file_path = Path(path)
        if not file_path.exists():
            return None
        return file_path.read_bytes()

    def list_images(
        self,
        collection: str | None = None,
        doc_hash: str | None = None,
    ) -> list[dict[str, Any]]:
        """按 collection / doc_hash 批量查询（条件均可省略），按写入倒序返回。"""
        sql = (
            "SELECT image_id, file_path, collection, doc_hash, page_num "
            "FROM image_index WHERE 1=1"
        )
        params: list[Any] = []
        if collection is not None:
            sql += " AND collection = ?"
            params.append(collection)
        if doc_hash is not None:
            sql += " AND doc_hash = ?"
            params.append(doc_hash)
        sql += " ORDER BY created_at DESC, rowid DESC"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    # ---------- 生命周期管理（供 DocumentManager 复用） ----------

    def delete_image(self, image_id: str) -> bool:
        """删除单张图片（文件 + 索引记录），返回是否实际删除。"""
        record = self.get_record(image_id)
        if record is None:
            return False
        _remove_file(record["file_path"])
        self._conn.execute("DELETE FROM image_index WHERE image_id = ?", (image_id,))
        self._conn.commit()
        return True

    def delete_images(
        self,
        collection: str,
        doc_hash: str | None = None,
    ) -> int:
        """删除某集合下（可按 doc_hash 限定）的全部图片，返回删除条数。

        协调删除的落盘部分：同时移除磁盘文件与数据库记录。
        """
        records = self.list_images(collection=collection, doc_hash=doc_hash)
        for record in records:
            _remove_file(record["file_path"])
        sql = "DELETE FROM image_index WHERE collection = ?"
        params: list[Any] = [collection]
        if doc_hash is not None:
            sql += " AND doc_hash = ?"
            params.append(doc_hash)
        cur = self._conn.execute(sql, params)
        self._conn.commit()
        return cur.rowcount

    # ---------- 统计 ----------

    def stats(self) -> dict[str, int]:
        """索引规模统计：图片总数、集合数。"""
        total = self._conn.execute(
            "SELECT COUNT(*) FROM image_index"
        ).fetchone()[0]
        collections = self._conn.execute(
            "SELECT COUNT(DISTINCT collection) FROM image_index"
        ).fetchone()[0]
        return {"total_images": total, "total_collections": collections}

    # ---------- 资源释放 ----------

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None


def _remove_file(path: str) -> None:
    """删除单个图片文件；文件缺失时静默跳过（不阻塞删除流程）。"""
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
