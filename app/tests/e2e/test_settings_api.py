from __future__ import annotations

from playwright.sync_api import APIRequestContext


def test_settings_api_allowlist_patch_and_reset(api_context: APIRequestContext) -> None:
    initial_response = api_context.get("/api/settings")
    assert initial_response.ok
    initial = initial_response.json()
    values = initial["values"]
    assert set(values) == {"global", "features", "jobs", "inference"}
    assert set(values["inference"]) == {"model_timeout"}
    original_seed = values["global"]["seed"]

    try:
        patch_response = api_context.patch(
            "/api/settings",
            data={"global": {"seed": (original_seed + 1) % 4_294_967_296}},
        )
        assert patch_response.ok
        assert patch_response.json()["values"]["global"]["seed"] == (
            original_seed + 1
        ) % 4_294_967_296

        invalid_response = api_context.patch(
            "/api/settings", data={"inference": {"device": "cuda"}}
        )
        assert invalid_response.status == 422
    finally:
        reset_response = api_context.post("/api/settings/reset")
        assert reset_response.ok
