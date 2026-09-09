"""Core 层统一出口：re-export 核心数据契约，简化导入路径。"""

from core.types import (
    IMAGE_PLACEHOLDER_PATTERN,
    Chunk,
    ChunkRecord,
    Document,
    ImageRef,
    ProcessedQuery,
    extract_image_ids,
    make_image_placeholder,
)

__all__ = [
    "IMAGE_PLACEHOLDER_PATTERN",
    "Chunk",
    "ChunkRecord",
    "Document",
    "ImageRef",
    "ProcessedQuery",
    "extract_image_ids",
    "make_image_placeholder",
]
