from __future__ import annotations

import pytest
from pathlib import Path
from playwright.sync_api import APIRequestContext, Page, expect
import base64
import os
import re


@pytest.fixture
def api_context(studio_context):
    """These tests exercise XREPORT Studio, not the inference application."""
    return studio_context


@pytest.fixture(scope="session")
def base_url(studio_base_url):
    return studio_base_url


_API_ONE_PIXEL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x0bIDATx\x9cc\xf8\xcf\xc0\xf0\x1f\x00"
    b"\x05\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)

###############################################################################
def test_s15_filesystem_validation(
    api_context: APIRequestContext,
    tmp_path: Path,
) -> None:
    accessible_root = tmp_path / "filesystem-fixture"
    image_folder = accessible_root / "images"
    empty_folder = accessible_root / "empty"
    image_folder.mkdir(parents=True)
    empty_folder.mkdir()
    (image_folder / "fixture.png").write_bytes(_API_ONE_PIXEL_PNG)
    file_path = accessible_root / "not-a-folder.txt"
    file_path.write_text("fixture", encoding="utf-8")
    missing_path = accessible_root / "missing"

    valid_response = api_context.post(
        "/api/preparation/images/validate",
        data={"folder_path": str(image_folder)},
    )
    assert valid_response.ok
    assert valid_response.json()["valid"] is True
    assert valid_response.json()["image_count"] == 1

    empty_response = api_context.post(
        "/api/preparation/images/validate",
        data={"folder_path": str(empty_folder)},
    )
    assert empty_response.ok
    assert empty_response.json()["valid"] is False
    assert empty_response.json()["image_count"] == 0

    for invalid_path in (missing_path, file_path):
        invalid_response = api_context.post(
            "/api/preparation/images/validate",
            data={"folder_path": str(invalid_path)},
        )
        assert invalid_response.ok
        assert invalid_response.json()["valid"] is False
        assert invalid_response.json()["image_count"] == 0


_ONE_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)

###############################################################################
def test_s15_dataset_folder_access_and_recovery(
    api_context: APIRequestContext,
    base_url: str,
    page: Page,
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "filesystem-fixture"
    image_folder = fixture_root / "images"
    empty_folder = fixture_root / "empty"
    image_folder.mkdir(parents=True)
    empty_folder.mkdir()
    (image_folder / "fixture.png").write_bytes(base64.b64decode(_ONE_PIXEL_PNG))

    evidence_dir = Path(
        os.environ.get("S15_EVIDENCE_DIR", str(tmp_path / "s15-evidence"))
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)

    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(f"{base_url.rstrip('/')}/dataset")
    folder_button = page.get_by_role(
        "button", name="Upload Image Folder", exact=False
    )
    expect(folder_button).to_be_enabled()
    selected = {"path": str(empty_folder)}
    page.route(
        "**/api/preparation/folder/select",
        lambda route: route.fulfill(json=selected),
    )
    folder_button.click()
    page.get_by_text("No valid images found", exact=False).wait_for(state="visible")
    page.screenshot(path=str(evidence_dir / "s15-empty-folder.png"), full_page=True)

    selected["path"] = str(image_folder)
    folder_button.click()
    page.get_by_role(
        "button", name=re.compile(r"Upload Image Folder.*1 images", re.S)
    ).wait_for(state="visible")
    page.screenshot(path=str(evidence_dir / "s15-image-selected.png"), full_page=True)
