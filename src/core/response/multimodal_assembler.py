"""Append locally stored images to an MCP response when retrieved chunks reference them."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.response.response_builder import MCPResponse
from core.types import RetrievalResult
from ingestion.storage.image_storage import ImageStorage


_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class MultimodalAssembler:
    """Turn ``image_refs`` from retrieval metadata into MCP image content.

    Image lookup is intentionally lazy: ordinary text-only queries neither open
    the image SQLite index nor touch the filesystem.
    """

    def __init__(
        self,
        image_storage: ImageStorage | None = None,
        storage_factory: Callable[[], ImageStorage] | None = None,
    ) -> None:
        if image_storage is None and storage_factory is None:
            raise ValueError("Provide image_storage or storage_factory.")
        self._image_storage = image_storage
        self._storage_factory = storage_factory

    def assemble(
        self,
        response: MCPResponse,
        retrieval_results: list[RetrievalResult],
    ) -> MCPResponse:
        """Return ``response`` with unique, readable images appended to content.

        A missing or unreadable image is skipped.  It must not turn a useful
        text retrieval response into a failed tool call.
        """
        image_ids = self._image_ids(retrieval_results)
        if not image_ids:
            return response

        storage = self._get_storage()
        content = list(response["content"])
        for image_id in image_ids:
            record = storage.get_record(image_id)
            data = storage.read_image(image_id)
            if record is None or data is None:
                continue
            mime_type = _MIME_BY_SUFFIX.get(
                Path(record["file_path"]).suffix.lower(), "image/png"
            )
            content.append(
                {
                    "type": "image",
                    "data": base64.b64encode(data).decode("ascii"),
                    "mimeType": mime_type,
                }
            )
        return {
            "content": content,
            "structuredContent": response["structuredContent"],
        }

    def _get_storage(self) -> ImageStorage:
        if self._image_storage is None:
            assert self._storage_factory is not None
            self._image_storage = self._storage_factory()
        return self._image_storage

    @staticmethod
    def _image_ids(results: list[RetrievalResult]) -> list[str]:
        seen: set[str] = set()
        image_ids: list[str] = []
        for result in results:
            for image_id in _normalise_image_refs(result.metadata.get("image_refs")):
                if image_id not in seen:
                    seen.add(image_id)
                    image_ids.append(image_id)
        return image_ids


def _normalise_image_refs(value: Any) -> list[str]:
    """Accept native lists and JSON strings returned by Chroma metadata."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]
