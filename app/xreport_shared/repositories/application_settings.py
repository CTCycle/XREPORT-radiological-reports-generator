from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from xreport_shared.configurations.settings import (
    ApplicationSettingsValues,
    InferenceSettingsValues,
)
from xreport_shared.repositories.database import Database, get_database
from xreport_shared.repositories.database.schema import application_schema


_SETTINGS_ID = 1


###############################################################################
class ApplicationSettingsRepository:
    """Database authority for the active application's singleton settings row."""

    # -------------------------------------------------------------------------
    def __init__(self, database: Database | None = None) -> None:
        self.database = database or get_database()
        schema = application_schema()
        self.record_model = schema.settings_model
        self.values_model: type[ApplicationSettingsValues] = (
            ApplicationSettingsValues
            if schema.profile == "studio"
            else InferenceSettingsValues
        )
        self.columns = frozenset(self.values_model.model_fields)

    # -------------------------------------------------------------------------
    def get_settings(self) -> ApplicationSettingsValues:
        with self.database.read_session() as session:
            record = session.execute(
                select(self.record_model).where(
                    self.record_model.settings_id == _SETTINGS_ID
                )
            ).scalar_one_or_none()
        if record is None:
            raise RuntimeError("Application settings row is missing")
        return self.values_model.model_validate(self._record_values(record))

    # -------------------------------------------------------------------------
    def update_public_settings(
        self, changes: Mapping[str, Any]
    ) -> ApplicationSettingsValues:
        unknown = set(changes).difference(self.columns)
        if unknown:
            raise ValueError(
                "Unsupported application setting column(s): "
                + ", ".join(sorted(unknown))
            )
        if not changes:
            raise ValueError("At least one application setting must be provided")

        with self.database.transaction() as session:
            record = session.execute(
                select(self.record_model)
                .where(self.record_model.settings_id == _SETTINGS_ID)
                .with_for_update()
            ).scalar_one_or_none()
            if record is None:
                raise RuntimeError("Application settings row is missing")
            values = self.values_model.model_validate(
                {**self._record_values(record), **changes}
            )
            for column, value in changes.items():
                setattr(record, column, value)
            record.updated_at = datetime.now(timezone.utc)
            session.flush()
        return values

    # -------------------------------------------------------------------------
    def reset_public_settings(
        self, defaults: ApplicationSettingsValues
    ) -> ApplicationSettingsValues:
        return self.update_public_settings(defaults.model_dump())

    # -------------------------------------------------------------------------
    def _record_values(self, record: Any) -> dict[str, Any]:
        return {column: getattr(record, column) for column in self.columns}


__all__ = ["ApplicationSettingsRepository"]
