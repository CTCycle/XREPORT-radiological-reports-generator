"""
E2E tests for active preparation and dataset API endpoints.
"""

from playwright.sync_api import APIRequestContext


class TestPreparationEndpoints:
    """Tests for the /preparation/* API endpoints."""

    def test_get_dataset_status_returns_shape(self, api_context: APIRequestContext):
        response = api_context.get("/preparation/dataset/status")
        assert response.ok, f"Expected 200, got {response.status}"

        data = response.json()
        assert isinstance(data.get("has_data"), bool)
        assert isinstance(data.get("row_count"), int)
        assert isinstance(data.get("message"), str)

    def test_get_dataset_names_returns_shape(self, api_context: APIRequestContext):
        response = api_context.get("/preparation/dataset/names")
        assert response.ok, f"Expected 200, got {response.status}"

        data = response.json()
        assert isinstance(data.get("datasets"), list)
        assert isinstance(data.get("count"), int)
        assert data["count"] == len(data["datasets"])

    def test_get_processed_dataset_names_returns_shape(
        self, api_context: APIRequestContext
    ):
        response = api_context.get("/preparation/dataset/processed/names")
        assert response.ok, f"Expected 200, got {response.status}"

        data = response.json()
        assert isinstance(data.get("datasets"), list)
        assert isinstance(data.get("count"), int)
        assert data["count"] == len(data["datasets"])

    def test_browse_root_returns_drives(self, api_context: APIRequestContext):
        response = api_context.get("/preparation/browse")
        assert response.ok, f"Expected 200, got {response.status}"

        data = response.json()
        assert data.get("current_path") == ""
        assert data.get("parent_path") is None
        assert isinstance(data.get("items"), list)
        assert isinstance(data.get("drives"), list)

    def test_get_preparation_job_status_invalid_job_returns_404(
        self, api_context: APIRequestContext
    ):
        response = api_context.get("/preparation/jobs/non_existent_job")
        assert response.status == 404
        assert "detail" in response.json()

    def test_cancel_preparation_job_invalid_job_returns_404(
        self, api_context: APIRequestContext
    ):
        response = api_context.delete("/preparation/jobs/non_existent_job")
        assert response.status == 404
        assert "detail" in response.json()
