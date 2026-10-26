from __future__ import annotations

import pytest

from server.services import huggingface_access as access_module
from server.services.errors import BadRequestError, ConflictError
from server.services.huggingface_access import HuggingFaceAccessService


###############################################################################
def test_status_reports_source_without_exposing_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = HuggingFaceAccessService()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(access_module, "get_token", lambda: None)
    assert service.get_status().model_dump() == {"configured": False, "source": None}

    monkeypatch.setattr(access_module, "get_token", lambda: "hf_stored")
    assert service.get_status().source == "stored"

    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert service.get_status().source == "environment"
    assert "hf_" not in service.get_status().model_dump_json()

###############################################################################
def test_save_token_rejects_invalid_token_and_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = HuggingFaceAccessService()
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def reject(**kwargs: object) -> None:
        raise ValueError("Invalid token passed!")

    monkeypatch.setattr(access_module, "login", reject)
    with pytest.raises(BadRequestError, match="rejected this token"):
        service.save_token("hf_bad")

    monkeypatch.setenv("HF_TOKEN", "hf_env")
    with pytest.raises(ConflictError, match="takes precedence"):
        service.save_token("hf_good")
    with pytest.raises(ConflictError):
        service.clear_token()
