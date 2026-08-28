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


###############################################################################
def is_pinned_revision(revision: str | None) -> bool:
    return bool(revision and REVISION_PATTERN.fullmatch(revision))


###############################################################################
def _validate_pinned_revision(repository_id: str, manifest: Mapping[str, Any]) -> None:
    if not is_pinned_revision(manifest.get("revision")):
        raise RuntimeError(f"{repository_id} requires a pinned 40-character revision")


###############################################################################
def _validate_remote_code(repository_id: str, manifest: Mapping[str, Any]) -> None:
    if manifest.get("trust_remote_code") and not manifest.get(
        "remote_code_approved", False
    ):
        raise RuntimeError(
            f"Remote code is not approved for pinned repository {repository_id}"
        )


###############################################################################
def _require_manifest_fields(
    repository_id: str, manifest: Mapping[str, Any]
) -> None:
    required_fields = (
        "adapter",
        "model_loader",
        "processor_loader",
        "max_current_images",
        "preferred_dtype",
    )
    missing_fields = [field for field in required_fields if field not in manifest]
    if missing_fields:
        raise RuntimeError(
            f"{repository_id} manifest is missing required field(s): "
            + ", ".join(missing_fields)
        )


###############################################################################
def _validate_string_choice(
    manifest: Mapping[str, Any],
    field: str,
    allowed_values: frozenset[str],
    unsupported_label: str,
) -> None:
    value = manifest[field]
    if not isinstance(value, str):
        raise RuntimeError(f"Manifest {field} must be a string")
    if value not in allowed_values:
        raise RuntimeError(f"{unsupported_label}: {value}")


###############################################################################
def _validate_max_current_images(manifest: Mapping[str, Any]) -> None:
    value = manifest["max_current_images"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError("max_current_images must be an integer")
    if value < 1:
        raise RuntimeError("max_current_images must be at least 1")


###############################################################################
def validate_manifest(
    repository_id: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(manifest)
    normalized["repository_id"] = repository_id
    _validate_pinned_revision(repository_id, normalized)
    _validate_remote_code(repository_id, normalized)
    _require_manifest_fields(repository_id, normalized)
    _validate_string_choice(
        normalized, "adapter", SUPPORTED_ADAPTERS, "Unsupported Hugging Face adapter"
    )
    _validate_string_choice(
        normalized,
        "model_loader",
        SUPPORTED_MODEL_LOADERS,
        "Unsupported Transformers model loader",
    )
    _validate_string_choice(
        normalized,
        "processor_loader",
        frozenset({"auto"}),
        "Unsupported Transformers processor loader",
    )
    _validate_max_current_images(normalized)
    _validate_string_choice(
        normalized, "preferred_dtype", SUPPORTED_DTYPES, "Unsupported preferred dtype"
    )

    return normalized
