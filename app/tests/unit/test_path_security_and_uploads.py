from __future__ import annotations

from server.common.utils.security import validate_checkpoint_name
from server.services.upload import UploadService, UploadState

###############################################################################
def test_checkpoint_names_reject_path_separators_cross_platform() -> None:
    for value in ("nested/name", "nested\\name"):
        try:
            validate_checkpoint_name(value)
        except ValueError as exc:
            assert "path separators" in str(exc)
        else:
            raise AssertionError(f"Expected path separator rejection for {value!r}")

###############################################################################
def test_upload_sanitizes_windows_style_filename() -> None:
    response = UploadService(UploadState()).upload_dataset(
        filename="nested\\dataset.csv",
        contents=b"image,text\nscan_1.png,report\n",
    )

    assert response.filename == "dataset.csv"
    assert response.dataset_name == "dataset"
