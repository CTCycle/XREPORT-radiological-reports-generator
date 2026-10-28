"""XREPORT Studio: dataset preparation, validation, training and checkpoint export."""

from __future__ import annotations

import os
from pathlib import Path

# The data layout is resolved when the shared package is first imported, so the
# application announces its profile before importing it (an explicit value from
# the launcher or the environment wins).
os.environ.setdefault("XREPORT_APP", "studio")

from xreport_shared.repositories.database.schema import (  # noqa: E402
    SchemaRegistration,
    register_application_schema,
)

register_application_schema(
    "studio",
    SchemaRegistration(
        module_name="xreport_studio.repositories.schemas.studio",
        base="StudioBase",
        settings="StudioSettingsRecord",
        history=("CheckpointEvaluation",),
        migrations_dir=Path(__file__).resolve().parent / "migrations" / "studio",
    ),
)
