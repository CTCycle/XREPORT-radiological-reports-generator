from __future__ import annotations

import os

from huggingface_hub import get_token, login, logout

from server.domain.huggingface_access import HuggingFaceAccessStatus
from server.services.errors import BadRequestError, ConflictError

ENVIRONMENT_VARIABLE = "HF_TOKEN"


###############################################################################
def resolve_token() -> str | None:
    """Return the token used for Hub requests; the environment wins over the store."""
    return os.environ.get(ENVIRONMENT_VARIABLE) or get_token()


###############################################################################
class HuggingFaceAccessService:
    """Manages the local Hugging Face credential used for gated model downloads."""

    # -------------------------------------------------------------------------
    def get_status(self) -> HuggingFaceAccessStatus:
        if os.environ.get(ENVIRONMENT_VARIABLE):
            return HuggingFaceAccessStatus(configured=True, source="environment")
        if get_token():
            return HuggingFaceAccessStatus(configured=True, source="stored")
        return HuggingFaceAccessStatus(configured=False)

    # -------------------------------------------------------------------------
    def save_token(self, token: str) -> HuggingFaceAccessStatus:
        self._require_manageable()
        try:
            login(token=token.strip(), add_to_git_credential=False)
        except Exception as exc:  # noqa: BLE001
            raise BadRequestError(
                detail="Hugging Face rejected this token. Check it and try again."
            ) from exc
        return self.get_status()

    # -------------------------------------------------------------------------
    def clear_token(self) -> HuggingFaceAccessStatus:
        self._require_manageable()
        logout()
        return self.get_status()

    # -------------------------------------------------------------------------
    @staticmethod
    def _require_manageable() -> None:
        if os.environ.get(ENVIRONMENT_VARIABLE):
            raise ConflictError(
                detail=(
                    f"{ENVIRONMENT_VARIABLE} is set in the backend environment and "
                    "takes precedence; remove it there to manage the token here."
                )
            )
