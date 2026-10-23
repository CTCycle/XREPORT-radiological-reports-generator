from __future__ import annotations

from pathlib import Path
from playwright.sync_api import APIRequestContext, Page, expect
import base64
import os
import re






_API_ONE_PIXEL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x0bIDATx\x9cc\xf8\xcf\xc0\xf0\x1f\x00"
    b"\x05\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)

###############################################################################
def _set_api_filesystem_access(api_context: APIRequestContext, enabled: bool) -> None:
    response = api_context.patch(
        "/api/settings",
        data={"features": {"allow_local_filesystem_access": enabled}},
    )
    assert response.ok, f"Could not set filesystem access: {response.status}"
    assert (
        response.json()["values"]["features"]["allow_local_filesystem_access"]
        is enabled
    )

###############################################################################
def test_s15_filesystem_browse_validation_and_feature_gate(
    api_context: APIRequestContext,
    tmp_path: Path,
) -> None:
    settings_response = api_context.get("/api/settings")
    assert settings_response.ok
    original_access = settings_response.json()["values"]["features"][
        "allow_local_filesystem_access"
    ]

    accessible_root = tmp_path / "filesystem-fixture"
    image_folder = accessible_root / "images"
    empty_folder = accessible_root / "empty"
    image_folder.mkdir(parents=True)
    empty_folder.mkdir()
    (image_folder / "fixture.png").write_bytes(_API_ONE_PIXEL_PNG)
    file_path = accessible_root / "not-a-folder.txt"
    file_path.write_text("fixture", encoding="utf-8")
    missing_path = accessible_root / "missing"

    try:
        _set_api_filesystem_access(api_context, True)

        browse_response = api_context.get(
            "/api/preparation/browse", params={"path": str(accessible_root)}
        )
        assert browse_response.ok
        items = {item["name"]: item for item in browse_response.json()["items"]}
        assert items["images"]["image_count"] == 1
        assert items["empty"]["image_count"] == 0

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

        missing_browse = api_context.get(
            "/api/preparation/browse", params={"path": str(missing_path)}
        )
        assert missing_browse.status == 404
        file_browse = api_context.get(
            "/api/preparation/browse", params={"path": str(file_path)}
        )
        assert file_browse.status == 404

        for invalid_path in (missing_path, file_path):
            invalid_response = api_context.post(
                "/api/preparation/images/validate",
                data={"folder_path": str(invalid_path)},
            )
            assert invalid_response.ok
            assert invalid_response.json()["valid"] is False
            assert invalid_response.json()["image_count"] == 0

        _set_api_filesystem_access(api_context, False)
        disabled_browse = api_context.get(
            "/api/preparation/browse", params={"path": str(accessible_root)}
        )
        assert disabled_browse.status == 403
        disabled_validation = api_context.post(
            "/api/preparation/images/validate",
            data={"folder_path": str(image_folder)},
        )
        assert disabled_validation.status == 403
    finally:
        _set_api_filesystem_access(api_context, original_access)









_ONE_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)

###############################################################################
def _set_filesystem_access(api_context: APIRequestContext, enabled: bool) -> None:
    response = api_context.patch(
        "/api/settings",
        data={"features": {"allow_local_filesystem_access": enabled}},
    )
    assert response.ok, f"Could not set filesystem access: {response.status}"

###############################################################################
def test_s15_dataset_folder_access_and_recovery(
    api_context: APIRequestContext,
    base_url: str,
    page: Page,
    tmp_path: Path,
) -> None:
    settings_response = api_context.get("/api/settings")
    assert settings_response.ok
    original_access = settings_response.json()["values"]["features"][
        "allow_local_filesystem_access"
    ]

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

    try:
        _set_filesystem_access(api_context, False)
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.goto(f"{base_url.rstrip('/')}/dataset")
        folder_button = page.get_by_role(
            "button", name="Upload Image Folder", exact=False
        )
        expect(folder_button).to_be_disabled()
        assert "Disabled by server configuration" in folder_button.inner_text()
        page.screenshot(path=str(evidence_dir / "s15-disabled-browse.png"), full_page=True)

        _set_filesystem_access(api_context, True)
        page.reload()
        folder_button = page.get_by_role(
            "button", name="Upload Image Folder", exact=False
        )
        expect(folder_button).to_be_enabled()
        folder_button.click()
        dialog = page.get_by_role("dialog", name="Select image folder")
        path_input = dialog.get_by_placeholder("Server folder path")

        missing_path = str(fixture_root / "missing")
        path_input.fill(missing_path)
        dialog.get_by_role("button", name="Browse", exact=True).click()
        page.get_by_text("Path not found", exact=False).wait_for(state="visible")
        page.screenshot(path=str(evidence_dir / "s15-invalid-path.png"), full_page=True)

        path_input.fill(str(fixture_root))
        dialog.get_by_role("button", name="Browse", exact=True).click()
        page.get_by_text("Path not found", exact=False).wait_for(state="hidden")
        empty_item = dialog.locator(".folder-item").filter(has_text="empty")
        empty_item.click()
        dialog.get_by_role("button", name="Use this folder").click()
        page.get_by_text("No valid images found", exact=False).wait_for(state="visible")
        assert dialog.is_visible()
        page.screenshot(path=str(evidence_dir / "s15-empty-folder.png"), full_page=True)

        path_input.fill(str(image_folder))
        dialog.get_by_role("button", name="Browse", exact=True).click()
        dialog.get_by_role("button", name="Use this folder").click()
        dialog.wait_for(state="hidden")
        page.get_by_role(
            "button", name=re.compile(r"Upload Image Folder.*1 images", re.S)
        ).wait_for(state="visible")
        page.screenshot(path=str(evidence_dir / "s15-image-selected.png"), full_page=True)
    finally:
        _set_filesystem_access(api_context, original_access)
