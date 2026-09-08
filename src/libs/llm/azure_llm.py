"""Azure LLM：Azure OpenAI 服务的 BaseLLM 实现。

配置差异：Azure 使用 ``azure_endpoint``（资源地址）、``api_version``（如 2024-06-01）、
``api_key``，模型名通过 ``deployment_name`` 指定（也可用 ``model`` 兜底）。
"""

from __future__ import annotations

from typing import Any

from libs.llm._openai_compat import OpenAICompatLLM
from libs.llm.llm_factory import LLMFactory


class AzureLLM(OpenAICompatLLM):
    provider = "azure"

    def _build_client(self) -> Any:
        if not (self.settings.azure_endpoint or self.settings.base_url):
            raise ValueError(
                "Azure LLM 缺少 azure_endpoint 配置（settings.llm.azure_endpoint）"
            )
        if not self.settings.api_version:
            raise ValueError("Azure LLM 缺少 api_version 配置（settings.llm.api_version）")

        import openai

        return openai.AzureOpenAI(
            azure_endpoint=self.settings.azure_endpoint or self.settings.base_url,
            api_version=self.settings.api_version,
            api_key=self.settings.api_key,
            timeout=self._timeout,
            http_client=self._http_client,
        )


LLMFactory.register("azure", AzureLLM)
