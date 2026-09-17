"""persist application settings and retire the legacy JSON authority"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

import server.repositories.schemas.types


revision: str = "f48a7c2e91b6"
down_revision: Union[str, None] = "d62f3ab4e8c1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_DEFAULTS: dict[str, object] = {
    "global_seed": 42,
    "allow_local_filesystem_access": True,
    "job_polling_interval": 1.0,
    "inference_hf_local_only": True,
    "inference_device": "auto",
    "inference_model_timeout": 600,
}
_ROOT_KEYS = {"global", "features", "jobs", "inference"}


def _legacy_configuration_path() -> Path:
    if os.getenv("XREPORT_DESKTOP", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        data_root = os.getenv("XREPORT_DATA_ROOT", "").strip()
        if data_root:
            return Path(data_root).expanduser() / "settings" / "configurations.json"
    repository_root = Path(__file__).resolve().parents[4]
    return repository_root / "settings" / "configurations.json"


def _required_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Legacy application configuration section {name!r} is invalid")
    return value


def _required_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise RuntimeError(f"Legacy application configuration value {name!r} is invalid")
    return value


def _required_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise RuntimeError(f"Legacy application configuration value {name!r} is invalid")
    return value


def _legacy_payload(path: Path) -> dict[str, dict[str, object]] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load legacy application configuration: {path}") from exc
    if not isinstance(payload, dict) or set(payload) != _ROOT_KEYS:
        raise RuntimeError("Legacy application configuration has an invalid root")

    global_values = _required_mapping(payload["global"], "global")
    feature_values = _required_mapping(payload["features"], "features")
    job_values = _required_mapping(payload["jobs"], "jobs")
    inference_values = _required_mapping(payload["inference"], "inference")
    expected_sections = {
        "global": {"seed"},
        "features": {"allow_local_filesystem_access"},
        "jobs": {"polling_interval"},
        "inference": {
            "hf_local_only",
            "device",
            "max_loaded_models",
            "model_timeout",
        },
    }
    for name, values in (
        ("global", global_values),
        ("features", feature_values),
        ("jobs", job_values),
        ("inference", inference_values),
    ):
        if set(values) != expected_sections[name]:
            raise RuntimeError(f"Legacy application configuration section {name!r} is invalid")
    return {
        "global": global_values,
        "features": feature_values,
        "jobs": job_values,
        "inference": inference_values,
    }


def _legacy_values(path: Path) -> dict[str, object]:
    payload = _legacy_payload(path)
    if payload is None:
        return dict(_DEFAULTS)
    global_values = payload["global"]
    feature_values = payload["features"]
    job_values = payload["jobs"]
    inference_values = payload["inference"]

    seed = _required_int(global_values["seed"], "global.seed")
    if not 0 <= seed <= 4_294_967_295:
        raise RuntimeError("Legacy global.seed is outside the supported range")
    polling = job_values["polling_interval"]
    if isinstance(polling, bool) or not isinstance(polling, (int, float)):
        raise RuntimeError("Legacy jobs.polling_interval is invalid")
    polling_value = float(polling)
    if not 0.25 <= polling_value <= 60:
        raise RuntimeError("Legacy jobs.polling_interval is outside the supported range")
    timeout = _required_int(inference_values["model_timeout"], "inference.model_timeout")
    if timeout < 1:
        raise RuntimeError("Legacy inference.model_timeout is invalid")
    max_loaded_models = _required_int(
        inference_values["max_loaded_models"], "inference.max_loaded_models"
    )
    if max_loaded_models != 1:
        raise RuntimeError("Legacy inference.max_loaded_models must equal 1")
    device = inference_values["device"]
    if not isinstance(device, str) or device not in {"auto", "cpu", "cuda"}:
        raise RuntimeError("Legacy inference.device is invalid")

    return {
        "global_seed": seed,
        "allow_local_filesystem_access": _required_bool(
            feature_values["allow_local_filesystem_access"],
            "features.allow_local_filesystem_access",
        ),
        "job_polling_interval": polling_value,
        "inference_hf_local_only": _required_bool(
            inference_values["hf_local_only"], "inference.hf_local_only"
        ),
        "inference_device": device,
        "inference_model_timeout": timeout,
    }


def upgrade() -> None:
    op.create_table(
        "application_settings",
        sa.Column("settings_id", sa.Integer(), nullable=False),
        sa.Column("global_seed", sa.BigInteger(), nullable=False),
        sa.Column("allow_local_filesystem_access", sa.Boolean(), nullable=False),
        sa.Column("job_polling_interval", sa.Float(), nullable=False),
        sa.Column("inference_hf_local_only", sa.Boolean(), nullable=False),
        sa.Column("inference_device", sa.String(length=16), nullable=False),
        sa.Column("inference_model_timeout", sa.BigInteger(), nullable=False),
        sa.Column(
            "updated_at",
            server.repositories.schemas.types.UTCDateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "settings_id = 1", name="ck_application_settings_singleton"
        ),
        sa.CheckConstraint(
            "global_seed >= 0 AND global_seed <= 4294967295",
            name="ck_application_settings_seed",
        ),
        sa.CheckConstraint(
            "job_polling_interval >= 0.25 AND job_polling_interval <= 60",
            name="ck_application_settings_polling_interval",
        ),
        sa.CheckConstraint(
            "inference_device IN ('auto', 'cpu', 'cuda')",
            name="ck_application_settings_device",
        ),
        sa.CheckConstraint(
            "inference_model_timeout >= 1",
            name="ck_application_settings_model_timeout",
        ),
        sa.PrimaryKeyConstraint("settings_id"),
    )
    values = _legacy_values(_legacy_configuration_path())
    op.bulk_insert(
        sa.table(
            "application_settings",
            sa.column("settings_id", sa.Integer()),
            sa.column("global_seed", sa.BigInteger()),
            sa.column("allow_local_filesystem_access", sa.Boolean()),
            sa.column("job_polling_interval", sa.Float()),
            sa.column("inference_hf_local_only", sa.Boolean()),
            sa.column("inference_device", sa.String(length=16)),
            sa.column("inference_model_timeout", sa.BigInteger()),
            sa.column("updated_at", sa.DateTime(timezone=True)),
        ),
        [{"settings_id": 1, **values, "updated_at": datetime.now(timezone.utc)}],
    )


def downgrade() -> None:
    op.drop_table("application_settings")
