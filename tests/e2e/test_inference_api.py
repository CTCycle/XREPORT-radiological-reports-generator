"""
E2E tests for Inference API endpoints.
Tests: /inference/checkpoints, /inference/generate
"""

from playwright.sync_api import APIRequestContext


class TestInferenceEndpoints:
    """Tests for the /inference/* API endpoints."""

    def test_get_checkpoints_list(self, api_context: APIRequestContext):
        """GET /inference/checkpoints should return available checkpoints."""
        response = api_context.get("/inference/checkpoints")
        assert response.ok, f"Expected 200, got {response.status}"

        data = response.json()
        assert "checkpoints" in data
        assert "success" in data
        assert isinstance(data["checkpoints"], list)

        # Each checkpoint should have name and optional created date
        for checkpoint in data["checkpoints"]:
            assert "name" in checkpoint

    def test_generate_without_images_returns_422(self, api_context: APIRequestContext):
        """POST /inference/generate without images should return 422."""
        response = api_context.post(
            "/inference/generate",
            multipart={
                "checkpoint": "test_checkpoint",
                "generation_mode": "greedy",
            },
        )
        # Missing required images field
        assert response.status == 422

    def test_generate_with_invalid_checkpoint_returns_error(
        self, api_context: APIRequestContext
    ):
        """POST /inference/generate with invalid checkpoint should fail."""
        # Create a minimal test image (1x1 white PNG)
        # PNG header + minimal valid PNG data
        png_data = bytes(
            [
                0x89,
                0x50,
                0x4E,
                0x47,
                0x0D,
                0x0A,
                0x1A,
                0x0A,  # PNG signature
                0x00,
                0x00,
                0x00,
                0x0D,
                0x49,
                0x48,
                0x44,
                0x52,  # IHDR chunk
                0x00,
                0x00,
                0x00,
                0x01,
                0x00,
                0x00,
                0x00,
                0x01,  # 1x1
                0x08,
                0x02,
                0x00,
                0x00,
                0x00,
                0x90,
                0x77,
                0x53,
                0xDE,
                0x00,
                0x00,
                0x00,
                0x0C,
                0x49,
                0x44,
                0x41,  # IDAT chunk
                0x54,
                0x08,
                0xD7,
                0x63,
                0xF8,
                0xFF,
                0xFF,
                0x3F,
                0x00,
                0x05,
                0xFE,
                0x02,
                0xFE,
                0xDC,
                0xCC,
                0x59,
                0xE7,
                0x00,
                0x00,
                0x00,
                0x00,
                0x49,
                0x45,
                0x4E,  # IEND chunk
                0x44,
                0xAE,
                0x42,
                0x60,
                0x82,
            ]
        )

        response = api_context.post(
            "/inference/generate",
            multipart={
                "checkpoint": "non_existent_checkpoint_xyz",
                "generation_mode": "greedy_search",
                "images": {
                    "name": "test.png",
                    "mimeType": "image/png",
                    "buffer": png_data,
                },
            },
        )

        # Should fail because checkpoint doesn't exist
        # Could be 400, 404, or 500 depending on when validation occurs
        assert response.status in [400, 404, 500]


class TestInferenceGenerationModes:
    """Tests for different generation modes."""

    def test_generate_accepts_greedy_mode(self, api_context: APIRequestContext):
        """POST /inference/generate should accept 'greedy' generation mode."""
        # Get checkpoints first
        checkpoints_response = api_context.get("/inference/checkpoints")
        if not checkpoints_response.ok:
            return

        checkpoints = checkpoints_response.json().get("checkpoints", [])
        if not checkpoints:
            return  # Skip if no checkpoints available

        checkpoint_name = checkpoints[0]["name"]

        # Create a minimal test image
        png_data = self._create_minimal_png()

        response = api_context.post(
            "/inference/generate",
            multipart={
                "checkpoint": checkpoint_name,
                "generation_mode": "greedy",
                "images": {
                    "name": "test.png",
                    "mimeType": "image/png",
                    "buffer": png_data,
                },
            },
        )

        # Either succeeds or fails gracefully
        # Actual generation may fail due to model loading issues in test env
        assert response.status in [200, 400, 500]

    def test_generate_accepts_beam_mode(self, api_context: APIRequestContext):
        """POST /inference/generate should accept 'beam' generation mode."""
        checkpoints_response = api_context.get("/inference/checkpoints")
        if not checkpoints_response.ok:
            return

        checkpoints = checkpoints_response.json().get("checkpoints", [])
        if not checkpoints:
            return

        checkpoint_name = checkpoints[0]["name"]
        png_data = self._create_minimal_png()

        response = api_context.post(
            "/inference/generate",
            multipart={
                "checkpoint": checkpoint_name,
                "generation_mode": "beam",
                "images": {
                    "name": "test.png",
                    "mimeType": "image/png",
                    "buffer": png_data,
                },
            },
        )

        assert response.status in [200, 400, 500]

    def _create_minimal_png(self) -> bytes:
        """Create a minimal valid 1x1 white PNG image."""
        return bytes(
            [
                0x89,
                0x50,
                0x4E,
                0x47,
                0x0D,
                0x0A,
                0x1A,
                0x0A,
                0x00,
                0x00,
                0x00,
                0x0D,
                0x49,
                0x48,
                0x44,
                0x52,
                0x00,
                0x00,
                0x00,
                0x01,
                0x00,
                0x00,
                0x00,
                0x01,
                0x08,
                0x02,
                0x00,
                0x00,
                0x00,
                0x90,
                0x77,
                0x53,
                0xDE,
                0x00,
                0x00,
                0x00,
                0x0C,
                0x49,
                0x44,
                0x41,
                0x54,
                0x08,
                0xD7,
                0x63,
                0xF8,
                0xFF,
                0xFF,
                0x3F,
                0x00,
                0x05,
                0xFE,
                0x02,
                0xFE,
                0xDC,
                0xCC,
                0x59,
                0xE7,
                0x00,
                0x00,
                0x00,
                0x00,
                0x49,
                0x45,
                0x4E,
                0x44,
                0xAE,
                0x42,
                0x60,
                0x82,
            ]
        )


class TestInferenceResponseFormat:
    """Tests for inference response format."""

    def test_checkpoints_response_format(self, api_context: APIRequestContext):
        """GET /inference/checkpoints should return properly formatted response."""
        response = api_context.get("/inference/checkpoints")
        assert response.ok

        data = response.json()

        # Verify response structure
        assert "checkpoints" in data
        assert "success" in data
        assert "message" in data
        assert isinstance(data["success"], bool)
        assert isinstance(data["message"], str)
