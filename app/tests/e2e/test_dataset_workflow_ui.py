from __future__ import annotations

import base64
import os
import shutil
from pathlib import Path
from uuid import uuid4

from playwright.sync_api import APIRequestContext, Page, expect
from server.common.path import RESOURCES_DIR


_ONE_PIXEL_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"


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
            row["name"] == dataset_name for row in names_before.json()["datasets"]
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
            row for row in names_after.json()["datasets"] if row["name"] == dataset_name
        )
        assert source["row_count"] == 1
    finally:
        if imported:
            api_context.delete(f"/api/preparation/dataset/{dataset_name}")
        _set_filesystem_access(api_context, original_access)


def test_s23_viewer_navigation_and_long_source_path_row_containment(
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

    dataset_name = f"s23-row-{uuid4().hex}"
    fixture_root = Path(RESOURCES_DIR) / f"s23-{uuid4().hex[:8]}"
    fixture_root.mkdir(parents=True)
    repeats = max(
        1,
        min(
            8,
            (230 - len(str(fixture_root)) - len("\\images") - 13) // 13,
        ),
    )
    long_path = "source-path-" + "long-segment-" * repeats
    image_folder = fixture_root / long_path / "images"
    image_folder.mkdir(parents=True)
    for index in range(1, 9):
        (image_folder / f"Image-{index:02d}.PNG").write_bytes(
            base64.b64decode(_ONE_PIXEL_PNG)
        )
    csv_path = fixture_root / f"{dataset_name}.csv"
    csv_path.write_text(
        "image,text\n"
        + "".join(
            f"image-{index:02d}.jpg,Synthetic report row {index}\n"
            for index in range(1, 9)
        ),
        encoding="utf-8",
    )
    evidence_dir = Path(
        os.environ.get("S23_S26_EVIDENCE_DIR", str(tmp_path / "s23-evidence"))
    ).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    imported = False

    try:
        _set_filesystem_access(api_context, True)
        with csv_path.open("rb") as dataset_file:
            uploaded = api_context.post(
                "/api/upload/dataset",
                multipart={
                    "file": {
                        "name": csv_path.name,
                        "mimeType": "text/csv",
                        "buffer": dataset_file.read(),
                    }
                },
            )
        assert uploaded.ok, uploaded.text()
        upload_id = uploaded.json()["upload_id"]
        loaded = api_context.post(
            "/api/preparation/dataset/load",
            data={
                "upload_id": upload_id,
                "image_folder_path": str(image_folder),
                "sample_size": 1.0,
                "confirm_unmatched": False,
            },
        )
        assert loaded.ok, loaded.text()
        imported = loaded.json()["success"]
        assert loaded.json()["matched_records"] == 8

        page.set_viewport_size({"width": 1920, "height": 1080})
        page.goto(f"{base_url.rstrip('/')}/dataset")
        row = page.locator(".dataset-table-row").filter(has_text=dataset_name)
        expect(row).to_be_visible()
        source_cell = row.locator(".dataset-path")
        expect(source_cell).to_be_visible()
        row_box = row.bounding_box()
        source_box = source_cell.bounding_box()
        assert row_box is not None and source_box is not None
        assert source_box["y"] >= row_box["y"]
        assert (
            source_box["y"] + source_box["height"] <= row_box["y"] + row_box["height"]
        )
        source_style = source_cell.evaluate(
            "element => { const style = getComputedStyle(element); return {"
            "whiteSpace: style.whiteSpace, overflow: style.overflow, "
            "textOverflow: style.textOverflow, minWidth: style.minWidth, "
            "hasOverflow: element.scrollWidth > element.clientWidth }; }"
        )
        assert source_style == {
            "whiteSpace": "nowrap",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
            "minWidth": "0px",
            "hasOverflow": True,
        }
        assert source_cell.get_attribute("title") == str(image_folder)
        page.screenshot(
            path=str(evidence_dir / "s23-long-source-path-row.png"),
            full_page=True,
        )

        row.get_by_role("button", name="View images").click()
        expect(page.get_by_role("heading", name="Image Viewer")).to_be_visible()
        expect(page.get_by_text("1 / 8")).to_be_visible()
        expect(page.get_by_role("img", name="image-01.jpg")).to_be_visible()
        page.screenshot(
            path=str(evidence_dir / "s23-image-viewer-first.png"),
            full_page=True,
        )
        page.get_by_role("button", name="Next image").click()
        expect(page.get_by_text("2 / 8")).to_be_visible()
        expect(page.get_by_role("img", name="image-02.jpg")).to_be_visible()
        page.screenshot(
            path=str(evidence_dir / "s23-image-viewer-next.png"),
            full_page=True,
        )
        page.get_by_role("button", name="Previous image").click()
        expect(page.get_by_text("1 / 8")).to_be_visible()
    finally:
        try:
            if imported:
                cleanup = api_context.delete(f"/api/preparation/dataset/{dataset_name}")
                assert cleanup.ok, cleanup.text()
        finally:
            try:
                _set_filesystem_access(api_context, original_access)
            finally:
                shutil.rmtree(fixture_root, ignore_errors=True)
