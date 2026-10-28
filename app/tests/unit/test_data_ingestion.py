from __future__ import annotations

from xreport_shared.common.utils.security import validate_checkpoint_name
from xreport_studio.services.upload import UploadService, UploadState

MAX_DATASET_UPLOAD_BYTES = 1024 * 1024
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
from xreport_studio.domain.training import LoadDatasetRequest
from xreport_shared.services.errors import BadRequestError, PayloadTooLargeError
from xreport_studio.services.preparation import PreparationService, count_image_files, scan_image_folder
from xreport_studio.repositories.serialization.dataset import DatasetIntegrityError, DatasetRepository

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

###############################################################################
def test_image_scanning_counts_recursively_and_filters_required_stems(
    tmp_path: Path,
) -> None:
    image_folder = tmp_path / "images"
    nested_folder = image_folder / "nested"
    nested_folder.mkdir(parents=True)
    (image_folder / "one.jpg").write_bytes(b"image")
    (image_folder / "ignore.txt").write_text("not an image", encoding="utf-8")
    (nested_folder / "two.PNG").write_bytes(b"image")
    (nested_folder / "three.bmp").write_bytes(b"image")

    assert count_image_files(str(image_folder)) == 3

    matched = scan_image_folder(
        str(image_folder),
        required_stems={"two"},
    )

    assert [Path(path).name for path in matched] == ["two.PNG"]

###############################################################################
def _preparation_service(upload_state: UploadState, repository) -> PreparationService:
    service = PreparationService.__new__(PreparationService)
    service.server_settings = SimpleNamespace(
        global_settings=SimpleNamespace(seed=42),
    )
    service.upload_state = upload_state
    service.dataset_repository = repository
    return service

###############################################################################
def test_load_dataset_reports_unmatched_rows_before_persisting(tmp_path: Path) -> None:
    image_folder = tmp_path
    (image_folder / "present.jpg").write_bytes(b"image")
    upload_state = UploadState()
    upload_id = "upload-demo"
    upload_state.store(
        upload_id,
        {
            "dataset_name": "demo",
            "dataframe": pd.DataFrame(
                {
                    "image": ["present.jpg", "missing.jpg"],
                    "text": ["present report", "missing report"],
                }
            ),
        },
    )
    saved: list[pd.DataFrame] = []
    repository = SimpleNamespace(upsert_source_dataset=saved.append)
    service = _preparation_service(upload_state, repository)

    preview = service.load_dataset(
        LoadDatasetRequest(
            upload_id=upload_id,
            image_folder_path=str(image_folder),
            sample_size=1.0,
            confirm_unmatched=False,
        )
    )

    assert preview.success is False
    assert preview.requires_confirmation is True
    assert preview.matched_records == 1
    assert preview.unmatched_records == 1
    assert saved == []
    assert upload_state.contains(upload_id) is True

###############################################################################
def test_load_dataset_confirmation_persists_explicit_partial_import(
    tmp_path: Path,
) -> None:
    image_folder = tmp_path
    (image_folder / "present.jpg").write_bytes(b"image")
    upload_state = UploadState()
    upload_id = "upload-demo"
    upload_state.store(
        upload_id,
        {
            "dataset_name": "demo",
            "dataframe": pd.DataFrame(
                {
                    "image": ["present.jpg", "missing.jpg"],
                    "text": ["present report", "missing report"],
                }
            ),
        },
    )
    saved: list[pd.DataFrame] = []
    repository = SimpleNamespace(upsert_source_dataset=saved.append)
    service = _preparation_service(upload_state, repository)

    result = service.load_dataset(
        LoadDatasetRequest(
            upload_id=upload_id,
            image_folder_path=str(image_folder),
            sample_size=1.0,
            confirm_unmatched=True,
        )
    )

    assert result.success is True
    assert result.partial_import is True
    assert result.matched_records == 1
    assert result.unmatched_records == 1
    assert len(saved) == 1
    assert len(saved[0]) == 1
    assert upload_state.contains(upload_id) is False

###############################################################################
def test_load_dataset_matches_image_stems_case_insensitively(tmp_path: Path) -> None:
    image_path = tmp_path / "Present.PNG"
    image_path.write_bytes(b"image")
    upload_state = UploadState()
    upload_id = "upload-case-fold"
    upload_state.store(
        upload_id,
        {
            "dataset_name": "case-fold",
            "dataframe": pd.DataFrame(
                {"image": ["present.jpg"], "text": ["present report"]}
            ),
        },
    )
    saved: list[pd.DataFrame] = []
    service = _preparation_service(
        upload_state,
        SimpleNamespace(upsert_source_dataset=saved.append),
    )

    result = service.load_dataset(
        LoadDatasetRequest(
            upload_id=upload_id,
            image_folder_path=str(tmp_path),
            sample_size=1.0,
            confirm_unmatched=False,
        )
    )

    assert result.success is True
    assert result.matched_records == 1
    assert result.unmatched_records == 0
    assert len(saved) == 1
    assert saved[0].iloc[0]["image_path"] == str(image_path)

###############################################################################
def test_load_dataset_rejects_no_matching_rows_without_persisting(
    tmp_path: Path,
) -> None:
    (tmp_path / "present.jpg").write_bytes(b"image")
    upload_state = UploadState()
    upload_id = "upload-no-match"
    upload_state.store(
        upload_id,
        {
            "dataset_name": "no-match",
            "dataframe": pd.DataFrame(
                {"image": ["missing.jpg"], "text": ["missing report"]}
            ),
        },
    )
    saved: list[pd.DataFrame] = []
    service = _preparation_service(
        upload_state,
        SimpleNamespace(upsert_source_dataset=saved.append),
    )

    with pytest.raises(BadRequestError, match="No dataset rows matched"):
        service.load_dataset(
            LoadDatasetRequest(
                upload_id=upload_id,
                image_folder_path=str(tmp_path),
                sample_size=1.0,
                confirm_unmatched=False,
            )
        )

    assert saved == []
    assert upload_state.contains(upload_id) is True

###############################################################################
def test_load_dataset_rejects_missing_image_column_without_clearing_upload(
    tmp_path: Path,
) -> None:
    (tmp_path / "present.jpg").write_bytes(b"image")
    upload_state = UploadState()
    upload_id = "upload-no-image-column"
    upload_state.store(
        upload_id,
        {
            "dataset_name": "no-image-column",
            "dataframe": pd.DataFrame(
                {"text": ["present report"], "other": ["present.jpg"]}
            ),
        },
    )
    service = _preparation_service(upload_state, SimpleNamespace())

    with pytest.raises(BadRequestError, match="image column"):
        service.load_dataset(
            LoadDatasetRequest(
                upload_id=upload_id,
                image_folder_path=str(tmp_path),
                sample_size=1.0,
                confirm_unmatched=False,
            )
        )

    assert upload_state.contains(upload_id) is True

###############################################################################
def test_load_dataset_rejects_missing_text_column_without_clearing_upload(
    tmp_path: Path,
) -> None:
    image_folder = tmp_path
    (image_folder / "present.jpg").write_bytes(b"image")
    upload_state = UploadState()
    upload_id = "upload-demo"
    upload_state.store(
        upload_id,
        {
            "dataset_name": "demo",
            "dataframe": pd.DataFrame({"image": ["present.jpg"]}),
        },
    )
    service = _preparation_service(upload_state, SimpleNamespace())

    with pytest.raises(BadRequestError, match="text"):
        service.load_dataset(
            LoadDatasetRequest(
                upload_id=upload_id,
                image_folder_path=str(image_folder),
                sample_size=1.0,
                confirm_unmatched=False,
            )
        )

    assert upload_state.contains(upload_id) is True

###############################################################################
def test_load_dataset_requires_the_explicit_upload_id(tmp_path: Path) -> None:
    image_folder = tmp_path
    (image_folder / "present.jpg").write_bytes(b"image")
    upload_state = UploadState()
    upload_state.store(
        "upload-one",
        {
            "dataset_name": "one",
            "dataframe": pd.DataFrame(
                {"image": ["present.jpg"], "text": ["report"]}
            ),
        },
    )
    service = _preparation_service(upload_state, SimpleNamespace())

    with pytest.raises(BadRequestError, match="upload_id"):
        service.load_dataset(
            LoadDatasetRequest(
                upload_id="upload-two",
                image_folder_path=str(image_folder),
                sample_size=1.0,
                confirm_unmatched=False,
            )
        )

###############################################################################
def test_validate_img_paths_fails_closed_for_deleted_images(tmp_path: Path) -> None:
    repository = DatasetRepository.__new__(DatasetRepository)
    with pytest.raises(DatasetIntegrityError, match="missing or invalid"):
        repository.validate_img_paths(
            pd.DataFrame({"path": [str(tmp_path / "does-not-exist.png")]})
        )

###############################################################################
def test_upload_accepts_exact_payload_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATASET_UPLOAD_MAX_MB", "1")
    state = UploadState()
    service = UploadService(state)
    parser_called = False

    def fake_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal parser_called
        parser_called = True
        return pd.DataFrame({"text": ["report"]})

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    response = service.upload_dataset(
        "at_limit.csv", b"x" * MAX_DATASET_UPLOAD_BYTES
    )

    assert parser_called is True
    assert response.success is True
    assert response.upload_id

###############################################################################
def test_upload_rejects_payload_over_limit_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATASET_UPLOAD_MAX_MB", "1")
    parser_called = False

    def fake_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal parser_called
        parser_called = True
        return pd.DataFrame({"text": ["report"]})

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    with pytest.raises(PayloadTooLargeError):
        UploadService(UploadState()).upload_dataset(
            "over_limit.csv", b"x" * (MAX_DATASET_UPLOAD_BYTES + 1)
        )

    assert parser_called is False

###############################################################################
def test_upload_ids_keep_parsed_contents_independent() -> None:
    state = UploadState()
    service = UploadService(state)

    first = service.upload_dataset(
        "first.csv", b"image,text\nfirst.png,First report\n"
    )
    second = service.upload_dataset(
        "second.csv", b"image,text\nsecond.png,Second report\n"
    )

    assert first.upload_id != second.upload_id
    first_data = state.get(first.upload_id)
    second_data = state.get(second.upload_id)
    assert first_data is not None
    assert second_data is not None
    assert first_data["filename"] == "first.csv"
    assert first_data["dataframe"].to_dict("records") == [
        {"image": "first.png", "text": "First report"}
    ]
    assert second_data["filename"] == "second.csv"
    assert second_data["dataframe"].to_dict("records") == [
        {"image": "second.png", "text": "Second report"}
    ]
