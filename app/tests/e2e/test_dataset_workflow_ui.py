from __future__ import annotations

import base64
import os
from pathlib import Path
from uuid import uuid4

from playwright.sync_api import APIRequestContext, Page, expect


_ONE_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def _set_filesystem_access(api_context: APIRequestContext, enabled: bool) -> None:
    response = api_context.patch(
        "/api/settings",
        data={"features": {"allow_local_filesystem_access": enabled}},
    )
    assert response.ok, f"Could not set filesystem access: {response.status}"


def test_s21_dataset_page_shows_and_confirms_partial_import(
    api_context: APIRequestContext,
    base_url: str,
    page: Page,
    tmp_path: Path,
) -> None:
    settings = api_context.get("/api/settings")
    assert settings.ok
    original_access = settings.json()["values"]["features"][
        "allow_local_filesystem_access"
    ]

    fixture_root = tmp_path / "s21-browser-fixture"
    image_folder = fixture_root / "images"
    image_folder.mkdir(parents=True)
    (image_folder / "Matched.PNG").write_bytes(base64.b64decode(_ONE_PIXEL_PNG))
    dataset_name = f"s21-browser-{uuid4().hex}"
    csv_path = fixture_root / f"{dataset_name}.csv"
    csv_path.write_text(
        "image,text\nmatched.jpg,Matched report\nmissing.jpg,Unmatched report\n",
        encoding="utf-8",
    )
    evidence_dir = Path(
        os.environ.get("S21_EVIDENCE_DIR", str(tmp_path / "s21-evidence"))
    ).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    imported = False

    try:
        _set_filesystem_access(api_context, True)
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.goto(f"{base_url.rstrip('/')}/dataset")

        page.get_by_role("button", name="Upload Image Folder").click()
        dialog = page.get_by_role("dialog", name="Select image folder")
        dialog.get_by_placeholder("Server folder path").fill(str(image_folder))
        dialog.get_by_role("button", name="Browse", exact=True).click()
        dialog.get_by_role("button", name="Use this folder").click()
        dialog.wait_for(state="hidden")

        page.locator("input[type='file']").set_input_files(str(csv_path))
        page.get_by_text("2 rows, 2 cols", exact=True).wait_for(state="visible")
        page.get_by_role("button", name="Load Dataset", exact=True).click()

        confirmation = page.get_by_role("alert").filter(
            has_text="Found 1 matched and 1 unmatched records"
        )
        expect(confirmation).to_be_visible()
        expect(
            confirmation.get_by_role("button", name="Import matched rows")
        ).to_be_visible()
        names_before = api_context.get("/api/preparation/dataset/names")
        assert names_before.ok
        assert not any(
            row["name"] == dataset_name
            for row in names_before.json()["datasets"]
        )
        page.screenshot(
            path=str(evidence_dir / "s21-before-partial-confirmation.png"),
            full_page=True,
        )

        confirmation.get_by_role("button", name="Import matched rows").click()
        expect(
            page.get_by_text(
                "Partial import confirmed: 1 matched records imported; "
                "1 unmatched records were not imported."
            )
        ).to_be_visible()
        imported = True
        page.screenshot(
            path=str(evidence_dir / "s21-partial-import-confirmed.png"),
            full_page=True,
        )

        names_after = api_context.get("/api/preparation/dataset/names")
        assert names_after.ok
        source = next(
            row
            for row in names_after.json()["datasets"]
            if row["name"] == dataset_name
        )
        assert source["row_count"] == 1
    finally:
        if imported:
            api_context.delete(f"/api/preparation/dataset/{dataset_name}")
        _set_filesystem_access(api_context, original_access)
