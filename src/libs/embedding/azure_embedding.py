"""Azure OpenAI Embedding 实现（text-embedding-ada-002 等）。

Azure 走 OpenAI 兼容协议，但客户端是 ``AzureOpenAI``，需 endpoint /
api-version / api-key 三要素。请求中的 ``model`` 用 deployment_name。
"""

from __future__ import annotations

from typing import Any

from libs.embedding._openai_compat import OpenAICompatEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class AzureEmbedding(OpenAICompatEmbedding):
    provider = "azure"

    def _build_client(self) -> Any:
        if not self.settings.azure_endpoint and not self.settings.base_url:
            raise ValueError(
                "Azure Embedding 缺少 azure_endpoint 配置"
                "（settings.embedding.azure_endpoint）"
            )
        if not self.settings.api_version:
            raise ValueError(
                "Azure Embedding 缺少 api_version 配置（settings.embedding.api_version）"
            )
        import openai

        return openai.AzureOpenAI(
            azure_endpoint=self.settings.azure_endpoint or self.settings.base_url,
            api_version=self.settings.api_version,
            api_key=self.settings.api_key,
            timeout=self._timeout,
            http_client=self._http_client,
        )


EmbeddingFactory.register("azure", AzureEmbedding)
