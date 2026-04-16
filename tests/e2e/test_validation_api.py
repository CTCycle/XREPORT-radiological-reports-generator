"""
E2E tests for Validation API endpoints.
Tests: /validation/checkpoint/reports/{checkpoint}, /validation/jobs/{job_id}
"""

import uuid

from playwright.sync_api import APIRequestContext


class TestValidationEndpoints:
    """Tests for active /validation endpoints."""

    def test_get_checkpoint_evaluation_report_missing_checkpoint_returns_404(
        self, api_context: APIRequestContext
    ):
        missing_checkpoint = f"missing_{uuid.uuid4().hex}"
        response = api_context.get(
            f"/api/validation/checkpoint/reports/{missing_checkpoint}"
        )
        assert response.status == 404
        data = response.json()
        assert "detail" in data

    def test_get_validation_job_status_invalid_job_returns_404(
        self, api_context: APIRequestContext
    ):
        response = api_context.get("/api/validation/jobs/non_existent_job")
        assert response.status == 404
        assert "detail" in response.json()

    def test_cancel_validation_job_invalid_job_returns_404(
        self, api_context: APIRequestContext
    ):
        response = api_context.delete("/api/validation/jobs/non_existent_job")
        assert response.status == 404
        assert "detail" in response.json()

