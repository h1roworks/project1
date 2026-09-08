"""OpenAI 兼容协议 Embedding 的共享实现基类。

OpenAI / Azure / DashScope 的 Embedding 接口遵循同一协议（``POST /embeddings``，
请求体 ``{"model": ..., "input": [...]}``），差异只在客户端构造参数。本模块把
批量 embed、输入校验、超长截断与错误包装抽成公共基类，各 provider 子类只需
实现 ``_build_client()``。
"""

from __future__ import annotations

from typing import Any

from libs.embedding.base_embedding import BaseEmbedding


class OpenAICompatEmbedding(BaseEmbedding):
    """基于 OpenAI SDK 的兼容协议 Embedding 实现基类。

    构造时可注入 ``http_client``：测试中传入包装 ``httpx.MockTransport`` 的
    客户端即可 mock HTTP，不触发真实网络；生产环境为 None 时由 SDK 自动创建
    真实客户端（懒加载，首次 embed 时才建立）。
    """

    _default_base_url = "https://api.openai.com/v1"
    _timeout = 60.0

    def __init__(self, settings: Any, http_client: Any = None) -> None:
        self.settings = settings
        self.model = settings.model or settings.deployment_name
        self._http_client = http_client
        self._client: Any = None

    def _build_client(self) -> Any:
        """按 provider 构造 OpenAI/Azure SDK 客户端（由子类实现）。"""
        raise NotImplementedError

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
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
        """按配置截断超长输入（max_input_chars>0 时启用，0 表示不截断）。"""
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

        batch_size = getattr(self.settings, "batch_size", 32) or len(texts)
        vectors: list[list[float]] = []
        try:
            client = self._get_client()
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                resp = client.embeddings.create(model=self.model, input=batch)
                ordered = sorted(resp.data, key=lambda d: d.index)
                vectors.extend(d.embedding for d in ordered)
        except Exception as exc:  # noqa: BLE001 - 统一包装为含 provider 与错误类型的可读错误
            raise RuntimeError(
                f"Embedding 调用失败 (provider={self.provider}, "
                f"错误类型={type(exc).__name__}): {exc}"
            ) from exc
        return vectors
