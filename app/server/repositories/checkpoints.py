from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil

from sqlalchemy import exists, select

from server.common.path import CHECKPOINTS_DIR
from server.common.utils.security import validate_checkpoint_name
from server.repositories.database import Database, get_database
from server.repositories.schemas import Checkpoint, CheckpointEvaluation, InferenceRun
from server.repositories.schemas.normalization import normalize_key


CHECKPOINT_ARTIFACT_FILES = (
    "saved_model.keras",
    "configuration/configuration.json",
    "configuration/metadata.json",
    "configuration/session_history.json",
)


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint_id: int
    name: str
    name_key: str
    path: Path
    created_at: datetime
    last_seen_at: datetime

    @property
    def artifact_complete(self) -> bool:
        return self.path.is_dir() and all(
            (self.path / relative_path).is_file()
            for relative_path in CHECKPOINT_ARTIFACT_FILES
        )


class CheckpointRegistryError(RuntimeError):
    """Raised when a checkpoint cannot be registered or deleted safely."""


class CheckpointReferencedError(CheckpointRegistryError):
    """Raised when persisted history still references a checkpoint."""


class CheckpointRepository:
    """Database-owned checkpoint identity with explicit artifact verification."""

    def __init__(self, database: Database | None = None) -> None:
        self.database = database or get_database()

    # -------------------------------------------------------------------------
    @staticmethod
    def _record(row: Checkpoint) -> CheckpointRecord:
        return CheckpointRecord(
            checkpoint_id=int(row.checkpoint_id),
            name=row.name,
            name_key=row.name_key,
            path=Path(row.path),
            created_at=row.created_at,
            last_seen_at=row.last_seen_at,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _canonical_artifact_path(path: str | Path) -> Path:
        resolved = Path(path).expanduser().resolve()
        base = CHECKPOINTS_DIR.resolve()
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise CheckpointRegistryError(
                f"Checkpoint artifact is outside the checkpoints directory: {resolved}"
            ) from exc
        return resolved

    # -------------------------------------------------------------------------
    @staticmethod
    def _assert_complete_artifact(path: Path) -> None:
        if not path.is_dir():
            raise CheckpointRegistryError(f"Checkpoint artifact directory is missing: {path}")
        missing = [
            relative_path
            for relative_path in CHECKPOINT_ARTIFACT_FILES
            if not (path / relative_path).is_file()
        ]
        if missing:
            raise CheckpointRegistryError(
                f"Checkpoint artifact is incomplete: {path} ({', '.join(missing)})"
            )

    # -------------------------------------------------------------------------
    def list_checkpoints(self) -> list[CheckpointRecord]:
        with self.database.read_session() as session:
            rows = session.execute(
                select(Checkpoint).order_by(Checkpoint.name_key)
            ).scalars().all()
        return [self._record(row) for row in rows]

    # -------------------------------------------------------------------------
    def get_checkpoint(self, name: str) -> CheckpointRecord | None:
        checkpoint_name = validate_checkpoint_name(name)
        with self.database.read_session() as session:
            row = session.execute(
                select(Checkpoint).where(
                    Checkpoint.name_key == normalize_key(checkpoint_name)
                )
            ).scalar_one_or_none()
        return self._record(row) if row is not None else None

    # -------------------------------------------------------------------------
    def register_completed_checkpoint(
        self,
        name: str,
        path: str | Path,
    ) -> CheckpointRecord:
        checkpoint_name = validate_checkpoint_name(name)
        artifact_path = self._canonical_artifact_path(path)
        if artifact_path.name != checkpoint_name:
            raise CheckpointRegistryError(
                f"Checkpoint name does not match its artifact directory: {checkpoint_name}"
            )
        self._assert_complete_artifact(artifact_path)
        name_key = normalize_key(checkpoint_name)
        path_key = str(artifact_path).casefold()
        now = datetime.now(timezone.utc)

        with self.database.transaction() as session:
            row = session.execute(
                select(Checkpoint).where(Checkpoint.name_key == name_key)
            ).scalar_one_or_none()
            path_rows = session.execute(select(Checkpoint)).scalars().all()
            for existing in path_rows:
                if str(Path(existing.path).resolve()).casefold() == path_key:
                    if existing.name_key != name_key:
                        raise CheckpointRegistryError(
                            f"Checkpoint artifact path is already registered: {artifact_path}"
                        )
                    row = existing
                    break

            if row is None:
                row = Checkpoint(
                    name=checkpoint_name,
                    name_key=name_key,
                    path=str(artifact_path),
                    created_at=now,
                    last_seen_at=now,
                )
                session.add(row)
                session.flush()
            elif str(Path(row.path).resolve()).casefold() != path_key:
                raise CheckpointRegistryError(
                    f"Checkpoint name is already registered to another artifact: {checkpoint_name}"
                )
            else:
                row.last_seen_at = now
            return self._record(row)

    # -------------------------------------------------------------------------
    def verify_artifact(self, name: str) -> CheckpointRecord:
        record = self.get_checkpoint(name)
        if record is None:
            raise CheckpointRegistryError(f"Checkpoint is not registered: {name}")
        now = datetime.now(timezone.utc)
        if record.artifact_complete:
            with self.database.transaction() as session:
                row = session.get(Checkpoint, record.checkpoint_id)
                if row is not None:
                    row.last_seen_at = now
                    return self._record(row)
        return record

    # -------------------------------------------------------------------------
    def delete_checkpoint(self, name: str) -> CheckpointRecord:
        record = self.get_checkpoint(name)
        if record is None:
            raise CheckpointRegistryError(f"Checkpoint is not registered: {name}")
        with self.database.transaction() as session:
            evaluation_exists = session.execute(
                select(
                    exists().where(
                        CheckpointEvaluation.checkpoint_id == record.checkpoint_id
                    )
                )
            ).scalar()
            inference_exists = session.execute(
                select(
                    exists().where(InferenceRun.checkpoint_id == record.checkpoint_id)
                )
            ).scalar()
            if evaluation_exists or inference_exists:
                raise CheckpointReferencedError(
                    f"Checkpoint cannot be deleted while persisted history references it: {record.name}"
                )
            artifact_path = record.path
            if artifact_path.exists():
                try:
                    shutil.rmtree(artifact_path)
                except OSError as exc:
                    raise CheckpointRegistryError(
                        f"Failed to delete checkpoint artifact: {artifact_path}"
                    ) from exc
            row = session.get(Checkpoint, record.checkpoint_id)
            if row is not None:
                session.delete(row)
        return record


__all__ = [
    "CHECKPOINT_ARTIFACT_FILES",
    "CheckpointRecord",
    "CheckpointRegistryError",
    "CheckpointReferencedError",
    "CheckpointRepository",
]
