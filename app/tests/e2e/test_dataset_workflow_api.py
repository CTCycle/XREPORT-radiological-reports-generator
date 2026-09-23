from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from playwright.sync_api import APIRequestContext


_ONE_PIXEL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x0bIDATx\x9cc\xf8\xcf\xc0\xf0\x1f\x00"
    b"\x05\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _set_filesystem_access(api_context: APIRequestContext, enabled: bool) -> None:
    response = api_context.patch(
        "/api/settings",
        data={"features": {"allow_local_filesystem_access": enabled}},
    )
    assert response.ok, f"Could not set filesystem access: {response.status}"


def _dataset_names(api_context: APIRequestContext) -> list[dict[str, object]]:
    response = api_context.get("/api/preparation/dataset/names")
    assert response.ok, response.text()
    return response.json()["datasets"]


def test_s21_partial_import_requires_confirmation_and_persists_matched_row(
    api_context: APIRequestContext,
    tmp_path: Path,
) -> None:
    settings = api_context.get("/api/settings")
    assert settings.ok
    original_access = settings.json()["values"]["features"][
        "allow_local_filesystem_access"
    ]

    fixture_root = tmp_path / "s21-partial-fixture"
    image_folder = fixture_root / "images"
    image_folder.mkdir(parents=True)
    image_path = image_folder / "MixedCase.PNG"
    image_path.write_bytes(_ONE_PIXEL_PNG)
    dataset_name = f"s21-partial-{uuid4().hex}"
    imported = False

    try:
        _set_filesystem_access(api_context, True)
        uploaded = api_context.post(
            "/api/upload/dataset",
            multipart={
                "file": {
                    "name": f"{dataset_name}.csv",
                    "mimeType": "text/csv",
                    "buffer": (
                        "image,text\n"
                        "mixedcase.jpg,Matched report\n"
                        "missing.jpg,Unmatched report\n"
                    ).encode(),
                }
            },
        )
        assert uploaded.ok, uploaded.text()
        payload = uploaded.json()
        assert payload["dataset_name"] == dataset_name

        request = {
            "upload_id": payload["upload_id"],
            "image_folder_path": str(image_folder),
            "sample_size": 1.0,
            "confirm_unmatched": False,
        }
        preview = api_context.post(
            "/api/preparation/dataset/load",
            data=request,
        )
        assert preview.ok, preview.text()
        preview_payload = preview.json()
        assert preview_payload["success"] is False
        assert preview_payload["requires_confirmation"] is True
        assert preview_payload["matched_records"] == 1
        assert preview_payload["unmatched_records"] == 1
        assert not any(row["name"] == dataset_name for row in _dataset_names(api_context))

        request["confirm_unmatched"] = True
        confirmed = api_context.post(
            "/api/preparation/dataset/load",
            data=request,
        )
        assert confirmed.ok, confirmed.text()
        confirmed_payload = confirmed.json()
        assert confirmed_payload["success"] is True
        assert confirmed_payload["partial_import"] is True
        assert confirmed_payload["matched_records"] == 1
        assert confirmed_payload["unmatched_records"] == 1
        imported = True

        source = next(
            row for row in _dataset_names(api_context) if row["name"] == dataset_name
        )
        assert source["row_count"] == 1
        image = api_context.get(
            f"/api/preparation/dataset/{dataset_name}/images/1"
        )
        assert image.ok, image.text()
        image_payload = image.json()
        assert image_payload["image_name"] == "mixedcase.jpg"
        assert image_payload["caption"] == "Matched report"
        assert image_payload["valid_path"] is True
        assert Path(image_payload["path"]) == image_path
    finally:
        if imported:
            api_context.delete(f"/api/preparation/dataset/{dataset_name}")
        _set_filesystem_access(api_context, original_access)


def test_s21_no_matching_rows_fail_without_persisting_source(
    api_context: APIRequestContext,
    tmp_path: Path,
) -> None:
    settings = api_context.get("/api/settings")
    assert settings.ok
    original_access = settings.json()["values"]["features"][
        "allow_local_filesystem_access"
    ]

    image_folder = tmp_path / "s21-no-match-images"
    image_folder.mkdir()
    (image_folder / "available.png").write_bytes(_ONE_PIXEL_PNG)
    dataset_name = f"s21-no-match-{uuid4().hex}"

    try:
        _set_filesystem_access(api_context, True)
        uploaded = api_context.post(
            "/api/upload/dataset",
            multipart={
                "file": {
                    "name": f"{dataset_name}.csv",
                    "mimeType": "text/csv",
                    "buffer": b"image,text\nmissing.jpg,No matching report\n",
                }
            },
        )
        assert uploaded.ok, uploaded.text()

        response = api_context.post(
            "/api/preparation/dataset/load",
            data={
                "upload_id": uploaded.json()["upload_id"],
                "image_folder_path": str(image_folder),
                "sample_size": 1.0,
                "confirm_unmatched": True,
            },
        )
        assert response.status == 400, response.text()
        assert "No dataset rows matched" in response.json()["detail"]
        assert not any(row["name"] == dataset_name for row in _dataset_names(api_context))
    finally:
        _set_filesystem_access(api_context, original_access)
