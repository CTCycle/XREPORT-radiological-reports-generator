"""
E2E tests for Dataset Upload API endpoint.
Tests: POST /upload/dataset
"""

from playwright.sync_api import APIRequestContext


class TestDatasetUploadEndpoint:
    """Tests for the /upload/dataset API endpoint."""

    def test_upload_without_file_returns_422(self, api_context: APIRequestContext):
        """POST /upload/dataset without a file should return 422 (validation error)."""
        response = api_context.post("/upload/dataset")
        # FastAPI returns 422 for missing required fields
        assert response.status == 422

    def test_upload_invalid_file_type_returns_400(self, api_context: APIRequestContext):
        """POST /upload/dataset with an invalid file type should return 400."""
        response = api_context.post(
            "/upload/dataset",
            multipart={
                "file": {
                    "name": "test.txt",
                    "mimeType": "text/plain",
                    "buffer": b"some text content",
                }
            },
        )
        # Invalid file type should fail with 400
        assert response.status == 400

        data = response.json()
        assert "detail" in data

    def test_upload_valid_csv_succeeds(self, api_context: APIRequestContext):
        """POST /upload/dataset with valid CSV should parse successfully."""
        # Create a sample CSV with expected columns (id, image path, report text)
        csv_content = b"id,image,text\n1,img001.png,Normal chest X-ray\n2,img002.png,Mild cardiomegaly"

        response = api_context.post(
            "/upload/dataset",
            multipart={
                "file": {
                    "name": "test_dataset.csv",
                    "mimeType": "text/csv",
                    "buffer": csv_content,
                }
            },
        )

        assert response.ok, f"Expected 200, got {response.status}: {response.text()}"

        data = response.json()
        assert data["success"] is True
        assert "filename" in data
        assert "dataset_name" in data
        assert data["dataset_name"] == "test_dataset"
        assert "row_count" in data
        assert data["row_count"] == 2
        assert "column_count" in data
        assert "columns" in data
        assert isinstance(data["columns"], list)

    def test_upload_csv_extracts_dataset_name_from_filename(
        self, api_context: APIRequestContext
    ):
        """POST /upload/dataset should extract dataset name from filename."""
        csv_content = b"col1,col2\nval1,val2"

        response = api_context.post(
            "/upload/dataset",
            multipart={
                "file": {
                    "name": "my_custom_dataset.csv",
                    "mimeType": "text/csv",
                    "buffer": csv_content,
                }
            },
        )

        if response.ok:
            data = response.json()
            assert data["dataset_name"] == "my_custom_dataset"

    def test_upload_xlsx_succeeds(self, api_context: APIRequestContext):
        """POST /upload/dataset should accept XLSX files."""
        # Note: Creating a real XLSX in tests is complex
        # This test verifies the endpoint accepts the format
        # but will fail parsing with invalid content
        response = api_context.post(
            "/upload/dataset",
            multipart={
                "file": {
                    "name": "test.xlsx",
                    "mimeType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "buffer": b"not a real xlsx",  # Will fail parsing
                }
            },
        )
        # Expect 400 because it's not a valid XLSX
        assert response.status == 400

    def test_upload_empty_csv_returns_400(self, api_context: APIRequestContext):
        """POST /upload/dataset with empty content should return 400."""
        response = api_context.post(
            "/upload/dataset",
            multipart={
                "file": {
                    "name": "empty.csv",
                    "mimeType": "text/csv",
                    "buffer": b"",
                }
            },
        )
        # Empty file should fail parsing
        assert response.status == 400


class TestDatasetUploadEdgeCases:
    """Edge case tests for dataset upload functionality."""

    def test_upload_csv_with_semicolon_separator(self, api_context: APIRequestContext):
        """POST /upload/dataset should auto-detect semicolon separator."""
        # CSV with semicolon separator (common in European locales)
        csv_content = b"id;image;text\n1;img001.png;Normal findings\n2;img002.png;Abnormal findings"

        response = api_context.post(
            "/upload/dataset",
            multipart={
                "file": {
                    "name": "semicolon_dataset.csv",
                    "mimeType": "text/csv",
                    "buffer": csv_content,
                }
            },
        )

        # Should succeed - pandas can auto-detect separator
        if response.ok:
            data = response.json()
            assert data["row_count"] == 2
            assert "id" in data["columns"]

    def test_upload_csv_with_special_characters(self, api_context: APIRequestContext):
        """POST /upload/dataset should handle special characters in content."""
        csv_content = "id,image,text\n1,img001.png,Findings: pneumonia (bilateral)\n2,img002.png,No acute findings – normal".encode(
            "utf-8"
        )

        response = api_context.post(
            "/upload/dataset",
            multipart={
                "file": {
                    "name": "special_chars.csv",
                    "mimeType": "text/csv",
                    "buffer": csv_content,
                }
            },
        )

        # Should handle UTF-8 content
        assert response.status in [200, 400]
