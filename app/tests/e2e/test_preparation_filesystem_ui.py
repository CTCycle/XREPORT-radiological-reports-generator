from __future__ import annotations

import base64
import os
from pathlib import Path
import re

from playwright.sync_api import APIRequestContext, Page


_ONE_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def _set_filesystem_access(api_context: APIRequestContext, enabled: bool) -> None:
    response = api_context.patch(
        "/api/settings",
        data={"features": {"allow_local_filesystem_access": enabled}},
    )
    assert response.ok, f"Could not set filesystem access: {response.status}"


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
        assert folder_button.is_disabled()
        assert "Disabled by server configuration" in folder_button.inner_text()
        page.screenshot(path=str(evidence_dir / "s15-disabled-browse.png"), full_page=True)

        _set_filesystem_access(api_context, True)
        page.reload()
        folder_button = page.get_by_role(
            "button", name="Upload Image Folder", exact=False
        )
        assert folder_button.is_enabled()
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
