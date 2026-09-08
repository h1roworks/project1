"""Ollama Embedding 实现（nomic-embed-text、mxbai-embed-large 等本地模型）。

``ollama`` 包在首次真正调用时才导入（懒加载），测试中可注入 Fake client。
``embed(model, input=[...])`` 原生支持批量输入，返回 ``{"embeddings": [...]}``。
"""

from __future__ import annotations

from typing import Any

from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class OllamaEmbedding(BaseEmbedding):
    provider = "ollama"
    _default_host = "http://localhost:11434"

    def __init__(self, settings: Any, client: Any = None) -> None:
        self.settings = settings
        self.model = settings.model
        self._host = settings.base_url or self._default_host
        self._client = client  # 测试注入的 Fake client；None 时懒加载真实客户端

    def _get_client(self) -> Any:
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self._host)
        return self._client

    def _validate_texts(self, texts: Any) -> None:
        if not isinstance(texts, list):
            raise ValueError(f"embed 的 texts 必须是 list（provider={self.provider}）")
        for t in texts:
            if not isinstance(t, str):
                raise ValueError(
                    f"embed 的 texts 每条必须是字符串（provider={self.provider}）"
                )

    def _maybe_truncate(self, texts: list[str]) -> list[str]:
        limit = getattr(self.settings, "max_input_chars", 0)
        if limit <= 0:
            return texts
        return [t[:limit] for t in texts]

    def embed(
        self,
        texts: list[str],
        trace: Any = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        self._validate_texts(texts)
        texts = self._maybe_truncate(texts)

        try:
            resp = self._get_client().embed(model=self.model, input=texts)
        except Exception as exc:  # noqa: BLE001 - 连接失败/超时统一包装为可读错误
            raise RuntimeError(
                f"Ollama Embedding 调用失败 (provider={self.provider}, "
                f"错误类型={type(exc).__name__}): {exc}"
            ) from exc

        return [list(v) for v in resp["embeddings"]]


EmbeddingFactory.register("ollama", OllamaEmbedding)
