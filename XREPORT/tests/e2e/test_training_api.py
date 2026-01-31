"""
E2E tests for Training API endpoints.
Tests: /training/checkpoints, /training/status, /training/start, /training/stop
"""
from playwright.sync_api import APIRequestContext


class TestTrainingEndpoints:
    """Tests for the /training/* API endpoints."""

    def test_get_training_status(self, api_context: APIRequestContext):
        """GET /training/status should return current training state."""
        response = api_context.get("/training/status")
        assert response.ok, f"Expected 200, got {response.status}"
        
        data = response.json()
        assert "is_training" in data
        assert "current_epoch" in data
        assert "total_epochs" in data
        assert "loss" in data
        assert "val_loss" in data
        assert "accuracy" in data
        assert "val_accuracy" in data
        assert "progress_percent" in data
        assert "elapsed_seconds" in data
        assert isinstance(data["is_training"], bool)

    def test_get_training_status_includes_chart_data(self, api_context: APIRequestContext):
        """GET /training/status should return numeric fields for dashboard display."""
        response = api_context.get("/training/status")
        assert response.ok
        
        data = response.json()
        assert isinstance(data.get("current_epoch"), int)
        assert isinstance(data.get("total_epochs"), int)

    def test_get_checkpoints_list(self, api_context: APIRequestContext):
        """GET /training/checkpoints should return a list of checkpoint info."""
        response = api_context.get("/training/checkpoints")
        assert response.ok, f"Expected 200, got {response.status}"
        
        data = response.json()
        assert isinstance(data, dict)
        assert "checkpoints" in data

        for checkpoint in data["checkpoints"]:
            assert "name" in checkpoint

    def test_stop_training_when_not_running(self, api_context: APIRequestContext):
        """POST /training/stop should return error if no training is active."""
        # First verify no training is running
        status_response = api_context.get("/training/status")
        if status_response.ok and status_response.json().get("is_training"):
            return  # Skip if training is actually running
        
        response = api_context.post("/training/stop")
        # When no training is running, expect 400 or 409
        assert response.status in [400, 409]
        
        data = response.json()
        assert "detail" in data


class TestTrainingStartValidation:
    """Tests for training start request validation."""

    def test_start_training_requires_valid_request(self, api_context: APIRequestContext):
        """POST /training/start should validate request body."""
        status_response = api_context.get("/training/status")
        if status_response.ok and status_response.json().get("is_training"):
            return  # Skip if training is already running

        # Invalid request should fail validation (epochs below minimum)
        response = api_context.post("/training/start", data={"epochs": 0})
        assert response.status == 422

    def test_start_training_while_already_running_returns_409(self, api_context: APIRequestContext):
        """POST /training/start should return 409 if training is already in progress."""
        # Check if training is already running
        status_response = api_context.get("/training/status")
        status_data = status_response.json()
        
        if status_data.get("is_training"):
            # Attempt to start again
            response = api_context.post("/training/start", data={
                "dataset_name": "test",
                "epochs": 1,
            })
            assert response.status == 409


class TestTrainingResumeEndpoint:
    """Tests for training resume functionality."""

    def test_resume_training_requires_checkpoint(self, api_context: APIRequestContext):
        """POST /training/resume should require a checkpoint name."""
        # Empty request should fail
        response = api_context.post("/training/resume", data={})
        
        # Should fail validation
        assert response.status in [400, 422]

    def test_resume_with_invalid_checkpoint_returns_error(self, api_context: APIRequestContext):
        """POST /training/resume with invalid checkpoint should fail."""
        status_response = api_context.get("/training/status")
        if status_response.ok and status_response.json().get("is_training"):
            return  # Skip if training is already running

        response = api_context.post("/training/resume", data={
            "checkpoint": "non_existent_checkpoint_xyz",
            "additional_epochs": 1
        })
        
        # Should return 404 or 400
        assert response.status in [400, 404, 422]
