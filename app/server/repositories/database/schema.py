from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any

from sqlalchemy import MetaData

from server.common.path import APP_PROFILE

MIGRATIONS_ROOT = Path(__file__).resolve().parents[2] / "migrations"

###############################################################################
@dataclass(frozen=True)
class ApplicationSchema:
    """Database schema owned by one application profile."""

    profile: str
    title: str
    base: Any
    metadata: MetaData
    settings_model: Any
    checkpoint_model: Any
    checkpoint_history: tuple[Any, ...]
    migrations_dir: Path
    version_table: str


# Each profile is resolved by module name so XREPORT Inference never imports the
# Studio schema (see server.common.profile_modules).
_PROFILE_SCHEMAS = {
    "xreport": (
        "XREPORT Inference",
        "server.repositories.schemas.inference",
        "InferenceBase",
        "InferenceSettingsRecord",
        ("InferenceRun",),
        "inference",
    ),
    "studio": (
        "XREPORT Studio",
        "server.repositories.schemas.studio",
        "StudioBase",
        "StudioSettingsRecord",
        ("CheckpointEvaluation",),
        "studio",
    ),
}
VERSION_TABLES = {
    profile: f"xreport_{stream}_version"
    for profile, (*_, stream) in _PROFILE_SCHEMAS.items()
}
# Version table written by the retired shared migration stream.
SHARED_VERSION_TABLE = "alembic_version"

###############################################################################
def application_schema(profile: str | None = None) -> ApplicationSchema:
    """Return a profile's schema; defaults to the running application's profile."""
    return _load_schema(profile or APP_PROFILE)

###############################################################################
@lru_cache(maxsize=None)
def _load_schema(profile: str) -> ApplicationSchema:
    try:
        title, module_name, base, settings, history, stream = _PROFILE_SCHEMAS[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown application profile: {profile!r}") from exc
    module = import_module(module_name)
    return ApplicationSchema(
        profile=profile,
        title=title,
        base=getattr(module, base),
        metadata=getattr(module, base).metadata,
        settings_model=getattr(module, settings),
        checkpoint_model=module.Checkpoint,
        checkpoint_history=tuple(
            getattr(module, name).checkpoint_id for name in history
        ),
        migrations_dir=MIGRATIONS_ROOT / stream,
        version_table=VERSION_TABLES[profile],
    )

###############################################################################
def profile_title(profile: str) -> str:
    return _PROFILE_SCHEMAS[profile][0]


__all__ = [
    "ApplicationSchema",
    "SHARED_VERSION_TABLE",
    "VERSION_TABLES",
    "application_schema",
    "profile_title",
]
