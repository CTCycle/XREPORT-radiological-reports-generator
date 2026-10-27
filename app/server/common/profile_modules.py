"""Modules that belong only to XREPORT Studio (dataset preparation and training).

The packaged XREPORT backend must not contain these. They are excluded from the
PyInstaller build and a unit test asserts the XREPORT application never imports
them, even lazily.
"""

from __future__ import annotations

STUDIO_ONLY_MODULES: tuple[str, ...] = (
    "server.api.upload",
    "server.api.preparation",
    "server.api.training",
    "server.api.validation",
    "server.services.upload",
    "server.services.preparation",
    "server.services.dataset_processing",
    "server.services.training",
    "server.services.training_worker",
    "server.services.validation",
    "server.services.validation_runs",
    "server.services.evaluation",
    "server.repositories.preparation",
    "server.repositories.serialization.dataset",
    "server.repositories.serialization.validation",
    "server.repositories.schemas.studio",
    "server.models.callbacks",
    "server.models.training.model",
    "server.models.training.trainer",
)
