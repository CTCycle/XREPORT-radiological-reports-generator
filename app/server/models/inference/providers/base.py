from __future__ import annotations

from typing import Protocol

from server.domain.inference import ModelAvailability

###############################################################################
class InferenceProvider(Protocol):
    """Stable boundary between the inference service and a model runtime."""

    # -------------------------------------------------------------------------
    def list_available_models(self) -> list[ModelAvailability]: ...

    # -------------------------------------------------------------------------
    def unload(self) -> None: ...
