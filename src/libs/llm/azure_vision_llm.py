"""Azure Vision LLM：基于 Azure OpenAI 的多模态 LLM 实现。

支持 GPT-4o / GPT-4-Vision-Preview 等模型，接受图片路径或原始字节输入，
超过 ``max_image_size`` 时自动压缩（默认 2048px）。
"""

from __future__ import annotations

from typing import Any

from libs.llm._openai_vision_compat import OpenAICompatVisionLLM
from libs.llm.llm_factory import LLMFactory


class AzureVisionLLM(OpenAICompatVisionLLM):
    provider = "azure"

    def _get_client(self) -> Any:
        if self._client is None:
            if not (self.settings.azure_endpoint or self.settings.base_url):
                raise ValueError(
                    "Azure Vision LLM 缺少 azure_endpoint 配置"
                    "（settings.vision_llm.azure_endpoint）"
                )
            if not self.settings.api_version:
                raise ValueError(
                    "Azure Vision LLM 缺少 api_version 配置（settings.vision_llm.api_version）"
                )

            import openai

            self._client = openai.AzureOpenAI(
                azure_endpoint=self.settings.azure_endpoint or self.settings.base_url,
                api_version=self.settings.api_version,
                api_key=self.settings.api_key,
                timeout=self._timeout,
                http_client=self._http_client,
            )
        return self._client


LLMFactory.register_vision("azure", AzureVisionLLM)
