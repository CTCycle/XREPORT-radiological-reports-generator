from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any

from sqlalchemy import MetaData

from xreport_shared.common.path import APP_PROFILE

# Alembic configuration shared by both migration streams (the streams themselves
# live inside each application package).
ALEMBIC_INI = Path(__file__).resolve().parents[3] / "server" / "alembic.ini"

PROFILE_TITLES = {"xreport": "XREPORT Inference", "studio": "XREPORT Studio"}
VERSION_TABLES = {
    "xreport": "xreport_inference_version",
    "studio": "xreport_studio_version",
}
# Version table written by the retired shared migration stream.
SHARED_VERSION_TABLE = "alembic_version"


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


###############################################################################
@dataclass(frozen=True)
class SchemaRegistration:
    """How an application package describes the schema it owns."""

    module_name: str
    base: str
    settings: str
    history: tuple[str, ...]
    migrations_dir: Path

_REGISTRATIONS: dict[str, SchemaRegistration] = {}


###############################################################################
def register_application_schema(profile: str, registration: SchemaRegistration) -> None:
    """Called by an application package when it is imported.

    The shared package never imports an application; each application announces
    its own schema instead.
    """
    if profile not in PROFILE_TITLES:
        raise ValueError(f"Unknown application profile: {profile!r}")
    _REGISTRATIONS[profile] = registration
    _load_schema.cache_clear()


###############################################################################
def application_schema(profile: str | None = None) -> ApplicationSchema:
    """Return a profile's schema; defaults to the running application's profile."""
    return _load_schema(profile or APP_PROFILE)


###############################################################################
@lru_cache(maxsize=None)
def _load_schema(profile: str) -> ApplicationSchema:
    if profile not in PROFILE_TITLES:
        raise ValueError(f"Unknown application profile: {profile!r}")
    registration = _REGISTRATIONS.get(profile)
    if registration is None:
        raise RuntimeError(
            f"{PROFILE_TITLES[profile]} has not registered its database schema; "
            "import its application package first"
        )
    module = import_module(registration.module_name)
    return ApplicationSchema(
        profile=profile,
        title=PROFILE_TITLES[profile],
        base=getattr(module, registration.base),
        metadata=getattr(module, registration.base).metadata,
        settings_model=getattr(module, registration.settings),
        checkpoint_model=module.Checkpoint,
        checkpoint_history=tuple(
            getattr(module, name).checkpoint_id for name in registration.history
        ),
        migrations_dir=registration.migrations_dir,
        version_table=VERSION_TABLES[profile],
    )


###############################################################################
def profile_title(profile: str) -> str:
    return PROFILE_TITLES[profile]


__all__ = [
    "ALEMBIC_INI",
    "ApplicationSchema",
    "SHARED_VERSION_TABLE",
    "SchemaRegistration",
    "VERSION_TABLES",
    "application_schema",
    "profile_title",
    "register_application_schema",
]
