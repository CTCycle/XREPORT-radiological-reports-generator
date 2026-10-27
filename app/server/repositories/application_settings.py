from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from server.configurations.settings import ApplicationSettingsValues
from server.repositories.database import Database, get_database
from server.repositories.schemas import ApplicationSettingsRecord


_SETTINGS_ID = 1
_PUBLIC_COLUMNS = frozenset(
    {
        "global_seed",
        "job_polling_interval",
        "inference_model_timeout",
        "inference_model_idle_timeout",
        "inference_max_new_tokens_deterministic",
        "inference_max_new_tokens_concise",
        "inference_max_new_tokens_detailed",
        "inference_detailed_num_beams",
        "theme_preference",
    }
)

###############################################################################
class ApplicationSettingsRepository:
    """Database authority for the singleton application-settings row."""

    # -------------------------------------------------------------------------
    def __init__(self, database: Database | None = None) -> None:
        self.database = database or get_database()

    # -------------------------------------------------------------------------
    def get_settings(self) -> ApplicationSettingsValues:
        with self.database.read_session() as session:
            record = session.execute(
                select(ApplicationSettingsRecord).where(
                    ApplicationSettingsRecord.settings_id == _SETTINGS_ID
                )
            ).scalar_one_or_none()
        if record is None:
            raise RuntimeError("Application settings row is missing")
        return self._validate_record(record)

    # -------------------------------------------------------------------------
    def update_public_settings(
        self, changes: Mapping[str, Any]
    ) -> ApplicationSettingsValues:
        unknown = set(changes).difference(_PUBLIC_COLUMNS)
        if unknown:
            raise ValueError(
                "Unsupported application setting column(s): "
                + ", ".join(sorted(unknown))
            )
        if not changes:
            raise ValueError("At least one application setting must be provided")

        with self.database.transaction() as session:
            record = session.execute(
                select(ApplicationSettingsRecord)
                .where(ApplicationSettingsRecord.settings_id == _SETTINGS_ID)
                .with_for_update()
            ).scalar_one_or_none()
            if record is None:
                raise RuntimeError("Application settings row is missing")
            candidate = {
                "global_seed": record.global_seed,
                "job_polling_interval": record.job_polling_interval,
                "inference_model_timeout": record.inference_model_timeout,
                "inference_model_idle_timeout": record.inference_model_idle_timeout,
                "inference_max_new_tokens_deterministic": record.inference_max_new_tokens_deterministic,
                "inference_max_new_tokens_concise": record.inference_max_new_tokens_concise,
                "inference_max_new_tokens_detailed": record.inference_max_new_tokens_detailed,
                "inference_detailed_num_beams": record.inference_detailed_num_beams,
                "theme_preference": record.theme_preference,
                **changes,
            }
            values = ApplicationSettingsValues.model_validate(candidate)
            for column, value in changes.items():
                setattr(record, column, value)
            record.updated_at = datetime.now(timezone.utc)
            session.flush()
        return values

    # -------------------------------------------------------------------------
    def reset_public_settings(
        self, defaults: ApplicationSettingsValues
    ) -> ApplicationSettingsValues:
        return self.update_public_settings(
            {
                "global_seed": defaults.global_seed,
                "job_polling_interval": defaults.job_polling_interval,
                "inference_model_timeout": defaults.inference_model_timeout,
                "inference_model_idle_timeout": defaults.inference_model_idle_timeout,
                "inference_max_new_tokens_deterministic": defaults.inference_max_new_tokens_deterministic,
                "inference_max_new_tokens_concise": defaults.inference_max_new_tokens_concise,
                "inference_max_new_tokens_detailed": defaults.inference_max_new_tokens_detailed,
                "inference_detailed_num_beams": defaults.inference_detailed_num_beams,
                "theme_preference": defaults.theme_preference,
            }
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_record(
        record: ApplicationSettingsRecord,
    ) -> ApplicationSettingsValues:
        return ApplicationSettingsValues.model_validate(
            {
                "global_seed": record.global_seed,
                "job_polling_interval": record.job_polling_interval,
                "inference_model_timeout": record.inference_model_timeout,
                "inference_model_idle_timeout": record.inference_model_idle_timeout,
                "inference_max_new_tokens_deterministic": record.inference_max_new_tokens_deterministic,
                "inference_max_new_tokens_concise": record.inference_max_new_tokens_concise,
                "inference_max_new_tokens_detailed": record.inference_max_new_tokens_detailed,
                "inference_detailed_num_beams": record.inference_detailed_num_beams,
                "theme_preference": record.theme_preference,
            }
        )


__all__ = ["ApplicationSettingsRepository"]
