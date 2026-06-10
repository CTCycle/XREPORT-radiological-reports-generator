from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath

from server.common.path import CHECKPOINTS_DIR


MAX_CHECKPOINT_NAME_LENGTH = 128
CHECKPOINT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

MAX_DATASET_NAME_LENGTH = 128
DATASET_NAME_ALLOWED_CHARS = re.compile(r"[^A-Za-z0-9._ -]+")
DATASET_NAME_EDGE_TRIM = "._ -"

###############################################################################
def validate_checkpoint_name(name: str) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        raise ValueError("Checkpoint name cannot be empty")
    if len(normalized) > MAX_CHECKPOINT_NAME_LENGTH:
        raise ValueError(
            f"Checkpoint name cannot exceed {MAX_CHECKPOINT_NAME_LENGTH} characters"
        )
    if "/" in normalized or "\\" in normalized:
        raise ValueError("Checkpoint name must not contain path separators")
    if Path(normalized).is_absolute() or PureWindowsPath(normalized).is_absolute():
        raise ValueError("Checkpoint path must be relative")
    if PureWindowsPath(normalized).drive:
        raise ValueError("Checkpoint name must not include a drive letter")
    if not CHECKPOINT_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Checkpoint name can only include letters, numbers, hyphen, and underscore"
        )
    return normalized

###############################################################################
def resolve_checkpoint_path(name: str) -> str:
    checkpoint_name = validate_checkpoint_name(name)
    base_path = CHECKPOINTS_DIR.resolve()
    target_path = (base_path / checkpoint_name).resolve()
    if base_path not in target_path.parents and target_path != base_path:
        raise ValueError("Checkpoint path is outside the checkpoints directory")
    return str(target_path)

###############################################################################
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

