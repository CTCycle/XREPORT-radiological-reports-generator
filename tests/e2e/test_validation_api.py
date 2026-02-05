"""
E2E tests for Validation API endpoints.
Tests: /validation/checkpoint/reports/{checkpoint}
"""

from playwright.sync_api import APIRequestContext


class TestCheckpointEvaluationReports:
    """Tests for checkpoint evaluation report endpoints."""

    def test_get_checkpoint_evaluation_report(self, api_context: APIRequestContext):
        """GET /validation/checkpoint/reports/{checkpoint} returns 200 or 404."""
        checkpoints_response = api_context.get("/training/checkpoints")
        assert checkpoints_response.ok, (
            f"Expected 200, got {checkpoints_response.status}"
        )

        checkpoints = checkpoints_response.json().get("checkpoints", [])
        if not checkpoints:
            return

        checkpoint_name = checkpoints[0]["name"]
        response = api_context.get(
            f"/validation/checkpoint/reports/{checkpoint_name}"
        )

        assert response.status in [200, 404]

        if response.status == 200:
            data = response.json()
            assert data.get("checkpoint") == checkpoint_name
            assert "metrics" in data
            assert "metric_configs" in data
            assert "results" in data
