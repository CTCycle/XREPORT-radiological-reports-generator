from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SUPPORTED_ADAPTERS = frozenset(
    {"medgemma", "chexone", "cxrmate_multi", "cxrmate_ed", "cxrmate2"}
)
SUPPORTED_MODEL_LOADERS = frozenset({"auto_model", "image_text_to_text", "causal_lm"})
SUPPORTED_DTYPES = frozenset({"auto", "float32", "float16", "bfloat16"})


def is_pinned_revision(revision: str | None) -> bool:
    return bool(revision and REVISION_PATTERN.fullmatch(revision))


def validate_manifest(
    repository_id: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(manifest)
    normalized["repository_id"] = repository_id
    revision = normalized.get("revision")
    if not is_pinned_revision(revision):
        raise RuntimeError(f"{repository_id} requires a pinned 40-character revision")
    if normalized.get("trust_remote_code") and not normalized.get(
        "remote_code_approved", False
    ):
        raise RuntimeError(
            f"Remote code is not approved for pinned repository {repository_id}"
        )

    required_fields = (
        "adapter",
        "model_loader",
        "processor_loader",
        "max_current_images",
        "preferred_dtype",
    )
    missing_fields = [field for field in required_fields if field not in normalized]
    if missing_fields:
        raise RuntimeError(
            f"{repository_id} manifest is missing required field(s): "
            + ", ".join(missing_fields)
        )

    adapter_name = normalized["adapter"]
    if not isinstance(adapter_name, str):
        raise RuntimeError("Manifest adapter must be a string")
    if adapter_name not in SUPPORTED_ADAPTERS:
        raise RuntimeError(f"Unsupported Hugging Face adapter: {adapter_name}")

    model_loader = normalized["model_loader"]
    if not isinstance(model_loader, str):
        raise RuntimeError("Manifest model_loader must be a string")
    if model_loader not in SUPPORTED_MODEL_LOADERS:
        raise RuntimeError(f"Unsupported Transformers model loader: {model_loader}")

    processor_loader = normalized["processor_loader"]
    if not isinstance(processor_loader, str):
        raise RuntimeError("Manifest processor_loader must be a string")
    if processor_loader != "auto":
        raise RuntimeError(f"Unsupported Transformers processor loader: {processor_loader}")

    max_current_images = normalized["max_current_images"]
    if isinstance(max_current_images, bool) or not isinstance(max_current_images, int):
        raise RuntimeError("max_current_images must be an integer")
    if max_current_images < 1:
        raise RuntimeError("max_current_images must be at least 1")

    preferred_dtype = normalized["preferred_dtype"]
    if not isinstance(preferred_dtype, str):
        raise RuntimeError("Manifest preferred_dtype must be a string")
    if preferred_dtype not in SUPPORTED_DTYPES:
        raise RuntimeError(f"Unsupported preferred dtype: {preferred_dtype}")

    return normalized
