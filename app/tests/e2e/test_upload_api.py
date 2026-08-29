"""Cross-layer checks for dataset upload behavior that matters to users."""

from playwright.sync_api import APIRequestContext

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
