"""图片自动标注（C7：ImageCaptioner）。

当 Chunk 含图片引用（``metadata.image_refs``）且启用 Vision LLM 时，对每张图片生成
描述性 caption 并写回 ``metadata.image_captions``；禁用/不可用/异常时走降级路径，
标记 ``metadata.has_unprocessed_images``，绝不阻塞摄取流水线（对齐 DEV_SPEC 3.1.1）。

**不变式**：Transform 只改写 ``chunk.metadata``；``id``/``text``/``start_offset`` /
``end_offset`` / ``source_ref`` 保持不变。无图片引用的 chunk 原样返回（幂等）。
``image_refs`` 永不删除——caption 只是附加信息，图片引用始终保留供下游溯源。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from core.settings import TransformSettings
from core.types import Chunk, ImageRef
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.llm_factory import LLMFactory

# src/ingestion/transform/image_captioner.py → 上溯 3 层得到仓库根目录
_DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "prompts" / "image_captioning.txt"
)


class ImageCaptioner(BaseTransform):
    """为 Chunk 中的图片引用生成 caption（可选 Vision LLM + 降级不阻塞）。"""

    name = "image_captioner"

    def __init__(
        self,
        settings: TransformSettings | None = None,
        vision_llm: BaseVisionLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        """初始化。

        Args:
            settings: TransformSettings；也可传完整 Settings（自动取 ``.transform``
                子段，Vision LLM 配置取 ``.vision_llm``）；缺省用默认值
                （caption_with_vision=False，即默认不调用图片模型）。
            vision_llm: 可注入的 Vision LLM 实例（测试传 Fake）；为 None 时从
                settings.vision_llm 懒创建。
            prompt_path: image_captioning prompt 文件路径；测试中可注入替代文本。
        """
        settings = settings or TransformSettings()
        self._settings = settings.transform if hasattr(settings, "transform") else settings
        self._vision_settings = getattr(settings, "vision_llm", None)
        self.vision_llm = vision_llm
        self.prompt_path = Path(prompt_path) if prompt_path else None
        self._prompt_template: str | None = None

    def transform(self, chunk: Chunk, trace: Any = None) -> Chunk:
        image_ids = list(chunk.metadata.get("image_refs", []))
        if not image_ids:
            return chunk  # 无图片引用：无需标注，原样返回

        metadata = dict(chunk.metadata)
        metadata["captioned"] = False

        if not self._settings.caption_with_vision:
            metadata["has_unprocessed_images"] = True
            return replace(chunk, metadata=metadata)

        try:
            vision = self._get_vision_llm()
        except Exception as exc:  # noqa: BLE001 - 工厂创建失败降级，不阻塞
            metadata["caption_error"] = str(exc)
            metadata["has_unprocessed_images"] = True
            return replace(chunk, metadata=metadata)

        if vision is None:
            metadata["has_unprocessed_images"] = True
            return replace(chunk, metadata=metadata)

        lookup = _build_image_lookup(chunk.metadata.get("images", []))
        captions: dict[str, str] = {}
        errors: list[str] = []
        for image_id in image_ids:
            ref = lookup.get(image_id)
            if ref is None:
                errors.append(f"{image_id}: 在 metadata.images 中找不到对应 ImageRef（无法定位图片路径）")
                continue
            try:
                captions[image_id] = self._caption_one(vision, ref, trace)
            except Exception as exc:  # noqa: BLE001 - 单图失败不影响其余图片
                errors.append(f"{image_id}: {exc}")

        if captions:
            metadata["image_captions"] = captions
            metadata["captioned"] = True
        if errors:
            metadata["caption_errors"] = errors
        metadata["has_unprocessed_images"] = bool(errors)
        return replace(chunk, metadata=metadata)

    # ---------- Vision LLM 标注 ----------

    def _get_vision_llm(self) -> BaseVisionLLM | None:
        """返回 Vision LLM 实例；未注入且 settings 无 vision_llm 配置时返回 None。"""
        if self.vision_llm is None:
            vision_settings = self._vision_settings
            if vision_settings is not None and getattr(vision_settings, "provider", ""):
                self.vision_llm = LLMFactory.create_vision_llm(vision_settings)
        return self.vision_llm

    def _caption_one(self, vision: BaseVisionLLM, ref: ImageRef, trace: Any) -> str:
        prompt = self._build_prompt()
        response = vision.chat_with_image(prompt, ref.path, trace=trace)
        caption = (response.content or "").strip()
        if not caption:
            raise ValueError("Vision LLM 返回空 caption")
        return caption

    def _build_prompt(self) -> str:
        if self._prompt_template is None:
            path = self.prompt_path or Path(self._settings.prompt_path or _DEFAULT_PROMPT_PATH)
            self._prompt_template = path.read_text(encoding="utf-8")
        return self._prompt_template


def _build_image_lookup(images: Any) -> dict[str, ImageRef]:
    """把 ``metadata.images``（ImageRef 对象或 dict 列表）构建为 id → ImageRef 映射。"""
    lookup: dict[str, ImageRef] = {}
    if not isinstance(images, (list, tuple)):
        return lookup
    for entry in images:
        try:
            ref = entry if isinstance(entry, ImageRef) else ImageRef.from_dict(entry)
        except Exception:  # noqa: BLE001 - 单个脏条目跳过，不影响其余
            continue
        lookup[ref.id] = ref
    return lookup
