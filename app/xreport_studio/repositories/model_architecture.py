from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select

from xreport_studio.domain.model_architecture import ModelArchitecture
from xreport_shared.repositories.database import Database, get_database
from xreport_shared.repositories.schemas.normalization import normalize_key
from xreport_studio.repositories.schemas.studio import (
    HiddenArchitecturePreset,
    ModelArchitecturePreset,
)

###############################################################################
@dataclass(frozen=True)
class PresetRecord:
    preset_id: int
    name: str
    description: str
    architecture: ModelArchitecture
    created_at: datetime
    updated_at: datetime

###############################################################################
class PresetNameConflictError(ValueError):
    """Raised when a preset name is already used (case-insensitive)."""

###############################################################################
class ModelArchitectureRepository:
    """Persisted user-defined architecture presets."""

    # -------------------------------------------------------------------------
    def __init__(self, database: Database | None = None) -> None:
        self.database = database or get_database()

    # -------------------------------------------------------------------------
    @staticmethod
    def _record(row: Any) -> PresetRecord:
        return PresetRecord(
            preset_id=int(row.preset_id),
            name=row.name,
            description=row.description,
            architecture=ModelArchitecture.model_validate(row.architecture_json),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _assert_name_free(session: Any, name_key: str, preset_id: int | None) -> None:
        existing = session.execute(
            select(ModelArchitecturePreset).where(
                ModelArchitecturePreset.name_key == name_key
            )
        ).scalar_one_or_none()
        if existing is not None and existing.preset_id != preset_id:
            raise PresetNameConflictError(
                f"A preset named '{existing.name}' already exists"
            )

    # -------------------------------------------------------------------------
    def list_presets(self) -> list[PresetRecord]:
        with self.database.read_session() as session:
            rows = (
                session.execute(
                    select(ModelArchitecturePreset).order_by(
                        ModelArchitecturePreset.name_key
                    )
                )
                .scalars()
                .all()
            )
        return [self._record(row) for row in rows]

    # -------------------------------------------------------------------------
    def get_preset(self, preset_id: int) -> PresetRecord | None:
        with self.database.read_session() as session:
            row = session.get(ModelArchitecturePreset, preset_id)
        return self._record(row) if row is not None else None

    # -------------------------------------------------------------------------
    def create_preset(
        self, name: str, description: str, architecture: ModelArchitecture
    ) -> PresetRecord:
        name = name.strip()
        name_key = normalize_key(name)
        now = datetime.now(timezone.utc)
        with self.database.transaction() as session:
            self._assert_name_free(session, name_key, None)
            row = ModelArchitecturePreset(
                name=name,
                name_key=name_key,
                description=description.strip(),
                architecture_json=architecture.model_dump(),
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.flush()
            return self._record(row)

    # -------------------------------------------------------------------------
    def update_preset(
        self,
        preset_id: int,
        name: str,
        description: str,
        architecture: ModelArchitecture,
    ) -> PresetRecord | None:
        name = name.strip()
        name_key = normalize_key(name)
        with self.database.transaction() as session:
            row = session.get(ModelArchitecturePreset, preset_id)
            if row is None:
                return None
            self._assert_name_free(session, name_key, preset_id)
            row.name = name
            row.name_key = name_key
            row.description = description.strip()
            row.architecture_json = architecture.model_dump()
            row.updated_at = datetime.now(timezone.utc)
            session.flush()
            return self._record(row)

    # -------------------------------------------------------------------------
    def delete_preset(self, preset_id: int) -> bool:
        with self.database.transaction() as session:
            row = session.get(ModelArchitecturePreset, preset_id)
            if row is None:
                return False
            session.delete(row)
        return True

    # -------------------------------------------------------------------------
    def hidden_builtin_keys(self) -> set[str]:
        with self.database.read_session() as session:
            rows = session.execute(select(HiddenArchitecturePreset.preset_key))
            return set(rows.scalars().all())

    # -------------------------------------------------------------------------
    def hide_builtin(self, key: str) -> None:
        with self.database.transaction() as session:
            if session.get(HiddenArchitecturePreset, key) is None:
                session.add(HiddenArchitecturePreset(preset_key=key))

    # -------------------------------------------------------------------------
    def restore_builtins(self) -> None:
        with self.database.transaction() as session:
            session.execute(delete(HiddenArchitecturePreset))


__all__ = [
    "ModelArchitectureRepository",
    "PresetNameConflictError",
    "PresetRecord",
]
