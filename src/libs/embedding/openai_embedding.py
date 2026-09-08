"""OpenAI 官方 Embedding 实现（text-embedding-3-small/large 等）。

复用 OpenAI 兼容协议基类，只补充 OpenAI 客户端构造。
"""

from __future__ import annotations

from typing import Any

from libs.embedding._openai_compat import OpenAICompatEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class OpenAIEmbedding(OpenAICompatEmbedding):
    provider = "openai"

    def _build_client(self) -> Any:
        import openai

        return openai.OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url or self._default_base_url,
            timeout=self._timeout,
            http_client=self._http_client,
        )


EmbeddingFactory.register("openai", OpenAIEmbedding)
