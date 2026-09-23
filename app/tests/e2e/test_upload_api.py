"""Cross-layer checks for dataset upload behavior that matters to users."""

from io import BytesIO

from openpyxl import Workbook
from playwright.sync_api import APIRequestContext
import pytest

###############################################################################
def test_upload_parses_semicolon_csv_and_preserves_utf8_text(
    api_context: APIRequestContext,
) -> None:
    csv_content = (
        "id;image;text\n"
        "1;img001.png;Normal findings\n"
        "2;img002.png;No acute findings – normal"
    ).encode("utf-8")

    response = api_context.post(
        "/api/upload/dataset",
        multipart={
            "file": {
                "name": "clinical_dataset.csv",
                "mimeType": "text/csv",
                "buffer": csv_content,
            }
        },
    )

    assert response.ok, f"Expected 200, got {response.status}: {response.text()}"
    payload = response.json()
    assert isinstance(payload["upload_id"], str) and payload["upload_id"]
    assert payload["dataset_name"] == "clinical_dataset"
    assert payload["row_count"] == 2
    assert payload["columns"] == ["id", "image", "text"]

###############################################################################
def test_upload_parses_comma_csv(api_context: APIRequestContext) -> None:
    response = api_context.post(
        "/api/upload/dataset",
        multipart={
            "file": {
                "name": "comma_dataset.csv",
                "mimeType": "text/csv",
                "buffer": b"id,image,text\n1,img001.png,Normal findings\n",
            }
        },
    )

    assert response.status == 200, response.text()
    payload = response.json()
    assert payload["dataset_name"] == "comma_dataset"
    assert payload["row_count"] == 1
    assert payload["columns"] == ["id", "image", "text"]
    assert payload["upload_id"]

###############################################################################
def test_upload_parses_xlsx_workbook(api_context: APIRequestContext) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(["id", "image", "text"])
    worksheet.append([1, "img001.png", "Normal findings"])
    worksheet.append([2, "img002.png", "No acute findings – normal"])
    contents = BytesIO()
    workbook.save(contents)

    response = api_context.post(
        "/api/upload/dataset",
        multipart={
            "file": {
                "name": "clinical_dataset.xlsx",
                "mimeType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "buffer": contents.getvalue(),
            }
        },
    )

    assert response.status == 200, response.text()
    payload = response.json()
    assert payload["dataset_name"] == "clinical_dataset"
    assert payload["row_count"] == 2
    assert payload["column_count"] == 3
    assert payload["columns"] == ["id", "image", "text"]
    assert payload["upload_id"]

###############################################################################
@pytest.mark.parametrize(
    ("filename", "contents"),
    [("empty.csv", b""), ("corrupt.xlsx", b"not an XLSX workbook")],
)
def test_upload_rejects_empty_or_corrupt_workbooks(
    api_context: APIRequestContext,
    filename: str,
    contents: bytes,
) -> None:
    response = api_context.post(
        "/api/upload/dataset",
        multipart={
            "file": {
                "name": filename,
                "mimeType": "application/octet-stream",
                "buffer": contents,
            }
        },
    )

    assert response.status == 400

###############################################################################
def test_upload_rejects_unsupported_file_types(api_context: APIRequestContext) -> None:
    response = api_context.post(
        "/api/upload/dataset",
        multipart={
            "file": {
                "name": "dataset.txt",
                "mimeType": "text/plain",
                "buffer": b"not a dataset",
            }
        },
    )

    assert response.status == 400

###############################################################################
def test_upload_rejects_payload_over_16_mib(api_context: APIRequestContext) -> None:
    contents = b"x" * (16 * 1024 * 1024 + 1)
    response = api_context.post(
        "/api/upload/dataset",
        multipart={
            "file": {
                "name": "oversized.csv",
                "mimeType": "text/csv",
                "buffer": contents,
            }
        },
    )

    assert response.status == 413

###############################################################################
def test_upload_assigns_independent_ids(api_context: APIRequestContext) -> None:
    upload_ids: list[str] = []
    for name, text in (("first", "First report"), ("second", "Second report")):
        response = api_context.post(
            "/api/upload/dataset",
            multipart={
                "file": {
                    "name": f"{name}.csv",
                    "mimeType": "text/csv",
                    "buffer": f"image,text\n{name}.png,{text}\n".encode(),
                }
            },
        )
        assert response.status == 200, response.text()
        upload_ids.append(response.json()["upload_id"])

    assert upload_ids[0]
    assert upload_ids[1]
    assert upload_ids[0] != upload_ids[1]
