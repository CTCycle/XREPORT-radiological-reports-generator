from __future__ import annotations

from pathlib import Path

from server.common.path import CHECKPOINTS_DIR
from server.common.utils.security import (
    resolve_checkpoint_path,
    validate_checkpoint_name,
)
from server.services.upload import UploadService, UploadState


###############################################################################
def test_validate_checkpoint_name_rejects_path_separators_cross_platform() -> None:
    for value in ("nested/name", "nested\\name"):
        try:
            validate_checkpoint_name(value)
        except ValueError as exc:
            assert "path separators" in str(exc)
        else:  # pragma: no cover - defensive assertion
            raise AssertionError(f"Expected path separator rejection for {value!r}")


###############################################################################
def test_resolve_checkpoint_path_returns_checkpoint_child_path() -> None:
    checkpoint_name = "checkpoint_123"

    resolved = Path(resolve_checkpoint_path(checkpoint_name))

    assert resolved == (CHECKPOINTS_DIR / checkpoint_name).resolve()
    assert CHECKPOINTS_DIR.resolve() in resolved.parents


###############################################################################
def test_upload_service_sanitizes_windows_style_filename() -> None:
    service = UploadService(UploadState())

    response = service.upload_dataset(
        filename="nested\\dataset.csv",
        contents=b"image,text\nscan_1.png,report\n",
    )

    assert response.filename == "dataset.csv"
    assert response.dataset_name == "dataset"
