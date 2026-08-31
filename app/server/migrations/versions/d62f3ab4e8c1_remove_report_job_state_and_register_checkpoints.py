"""remove persisted job state and register complete checkpoints"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from server.common.path import CHECKPOINTS_DIR
from server.repositories.schemas.normalization import normalize_key


revision: str = "d62f3ab4e8c1"
down_revision: Union[str, None] = "c1e4f1a7b2d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_CHECKPOINT_FILES = (
    "saved_model.keras",
    "configuration/configuration.json",
    "configuration/metadata.json",
    "configuration/session_history.json",
)


def _complete_checkpoint_paths() -> list[Path]:
    if not CHECKPOINTS_DIR.is_dir():
        return []
    return [
        entry
        for entry in sorted(
            CHECKPOINTS_DIR.iterdir(), key=lambda item: item.name.casefold()
        )
        if entry.is_dir()
        and all(
            (entry / relative_path).is_file() for relative_path in _CHECKPOINT_FILES
        )
    ]


def _normalized_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve()).casefold()


def _register_complete_checkpoints() -> None:
    connection = op.get_bind()
    existing_rows = (
        connection.execute(sa.text("SELECT name, name_key, path FROM checkpoints"))
        .mappings()
        .all()
    )
    existing_names = {
        str(row["name_key"]): (str(row["name"]), _normalized_path(str(row["path"])))
        for row in existing_rows
    }
    existing_paths = {
        _normalized_path(str(row["path"])): str(row["name_key"])
        for row in existing_rows
    }

    seen_names: dict[str, str] = {}
    seen_paths: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    now = datetime.now(timezone.utc)
    for checkpoint_path in _complete_checkpoint_paths():
        name = checkpoint_path.name
        name_key = normalize_key(name)
        path = checkpoint_path.resolve()
        path_key = _normalized_path(path)
        prior_name = seen_names.get(name_key)
        if prior_name is not None and prior_name != name:
            raise RuntimeError(
                f"Checkpoint registry migration found duplicate normalized names: "
                f"{prior_name!r} and {name!r}"
            )
        prior_path = seen_paths.get(path_key)
        if prior_path is not None and prior_path != name_key:
            raise RuntimeError(
                f"Checkpoint registry migration found a path collision for {path}"
            )
        seen_names[name_key] = name
        seen_paths[path_key] = name_key

        existing = existing_names.get(name_key)
        if existing is not None:
            if existing[1] != path_key:
                raise RuntimeError(
                    f"Checkpoint registry migration found conflicting paths for {name!r}"
                )
            continue
        registered_name_key = existing_paths.get(path_key)
        if registered_name_key is not None and registered_name_key != name_key:
            raise RuntimeError(
                f"Checkpoint registry migration found a path collision for {path}"
            )
        rows.append(
            {
                "name": name,
                "name_key": name_key,
                "path": str(path),
                "created_at": now,
                "last_seen_at": now,
            }
        )

    if rows:
        checkpoints = sa.table(
            "checkpoints",
            sa.column("name", sa.String()),
            sa.column("name_key", sa.String(255)),
            sa.column("path", sa.Text()),
            sa.column("created_at"),
            sa.column("last_seen_at"),
        )
        op.bulk_insert(checkpoints, rows)


def _drop_request_id_constraint(table_name: str) -> None:
    connection = op.get_bind()
    if connection.dialect.name == "postgresql":
        op.drop_constraint(
            f"{table_name}_request_id_key",
            table_name,
            type_="unique",
        )


def upgrade() -> None:
    _register_complete_checkpoints()
    _drop_request_id_constraint("validation_runs")
    with op.batch_alter_table("validation_runs", schema=None) as batch_op:
        batch_op.drop_constraint("ck_validation_runs_status", type_="check")
        batch_op.drop_column("request_id")
        batch_op.drop_column("status")

    _drop_request_id_constraint("checkpoint_evaluations")
    with op.batch_alter_table("checkpoint_evaluations", schema=None) as batch_op:
        batch_op.drop_constraint("ck_checkpoint_evaluations_status", type_="check")
        batch_op.drop_column("request_id")
        batch_op.drop_column("status")


def downgrade() -> None:
    with op.batch_alter_table("checkpoint_evaluations", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "status",
                sa.String(length=16),
                nullable=False,
                server_default="succeeded",
            )
        )
        batch_op.add_column(
            sa.Column("request_id", sa.String(length=64), nullable=True)
        )
        batch_op.create_unique_constraint(
            "checkpoint_evaluations_request_id_key", ["request_id"]
        )
        batch_op.create_check_constraint(
            "ck_checkpoint_evaluations_status",
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
        )

    with op.batch_alter_table("validation_runs", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "status",
                sa.String(length=16),
                nullable=False,
                server_default="succeeded",
            )
        )
        batch_op.add_column(
            sa.Column("request_id", sa.String(length=64), nullable=True)
        )
        batch_op.create_unique_constraint(
            "validation_runs_request_id_key", ["request_id"]
        )
        batch_op.create_check_constraint(
            "ck_validation_runs_status",
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
        )
