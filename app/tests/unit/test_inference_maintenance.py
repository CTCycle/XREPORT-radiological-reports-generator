from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from server.domain.inference import InferenceModelsResponse
from server.services.errors import NotFoundError
from server.services.inference import InferenceImageStore, InferenceService

###############################################################################
def test_unknown_model_reference_is_not_a_maintenance_target(monkeypatch: pytest.MonkeyPatch) -> None:
    service = InferenceService(MagicMock(), InferenceImageStore())
    monkeypatch.setattr(
        service,
        "get_models",
        lambda: InferenceModelsResponse(models=[], providers={}),
    )

    with pytest.raises(NotFoundError, match="not in the local inference catalog"):
        service.start_model_maintenance(
            model_ref="huggingface:example/unknown-model",
            action="delete_local",
            revision=None,
        )
