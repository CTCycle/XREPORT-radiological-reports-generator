from __future__ import annotations

import os
import re


MAX_CHECKPOINT_NAME_LENGTH = 128
CHECKPOINT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

MAX_DATASET_NAME_LENGTH = 128
DATASET_NAME_ALLOWED_CHARS = re.compile(r"[^A-Za-z0-9._ -]+")
DATASET_NAME_EDGE_TRIM = "._ -"


# -----------------------------------------------------------------------------
def validate_checkpoint_name(name: str) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        raise ValueError("Checkpoint name cannot be empty")
    if len(normalized) > MAX_CHECKPOINT_NAME_LENGTH:
        raise ValueError(
            f"Checkpoint name cannot exceed {MAX_CHECKPOINT_NAME_LENGTH} characters"
        )
    if os.path.sep in normalized or (
        os.path.altsep and os.path.altsep in normalized
    ):
        raise ValueError("Checkpoint name must not contain path separators")
    if os.path.isabs(normalized):
        raise ValueError("Checkpoint path must be relative")
    if os.path.splitdrive(normalized)[0]:
        raise ValueError("Checkpoint name must not include a drive letter")
    if not CHECKPOINT_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Checkpoint name can only include letters, numbers, hyphen, and underscore"
        )
    return normalized


# -----------------------------------------------------------------------------
def sanitize_dataset_name(name: str) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        raise ValueError("Dataset name cannot be empty")

    sanitized = DATASET_NAME_ALLOWED_CHARS.sub("_", normalized).strip(
        DATASET_NAME_EDGE_TRIM
    )
    if not sanitized:
        raise ValueError("Dataset name is invalid after sanitization")
    if len(sanitized) > MAX_DATASET_NAME_LENGTH:
        sanitized = sanitized[:MAX_DATASET_NAME_LENGTH].rstrip(DATASET_NAME_EDGE_TRIM)
        if not sanitized:
            raise ValueError("Dataset name is invalid after truncation")
    return sanitized

