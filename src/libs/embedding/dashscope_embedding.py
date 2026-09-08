"""DashScope（阿里百炼）Embedding 实现。

DashScope 提供 OpenAI 兼容模式，协议与 OpenAI 一致，仅默认 base_url 不同。
"""

from __future__ import annotations

from typing import Any

from libs.embedding._openai_compat import OpenAICompatEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class DashScopeEmbedding(OpenAICompatEmbedding):
    provider = "dashscope"
    _default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def _build_client(self) -> Any:
        import openai

        return openai.OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url or self._default_base_url,
            timeout=self._timeout,
            http_client=self._http_client,
        )


EmbeddingFactory.register("dashscope", DashScopeEmbedding)
