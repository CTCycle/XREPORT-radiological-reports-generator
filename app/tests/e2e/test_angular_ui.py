"""Rendered Angular route and error-flow checks for the local application."""

from playwright.sync_api import Page, expect

###############################################################################
def test_inference_route_renders_catalog_and_navigation(
    page: Page,
    base_url: str,
) -> None:
    response = page.goto(f"{base_url}/inference")

    assert response is not None and response.ok
    expect(page.get_by_role("heading", name="Turn a radiograph into a draft report")).to_be_visible()
    expect(page.get_by_text("Model catalogue", exact=True)).to_be_visible()
    expect(page.get_by_role("link", name="Dataset")).to_be_visible()
    expect(page.get_by_role("link", name="Training")).to_be_visible()

###############################################################################
def test_settings_route_persists_and_resets_runtime_values(
    page: Page,
    base_url: str,
) -> None:
    response = page.goto(f"{base_url}/inference")
    assert response is not None and response.ok
    page.get_by_role("link", name="Settings").click()
    expect(page).to_have_url(f"{base_url}/settings")
    expect(page.get_by_role("heading", name="Settings")).to_be_visible()
    expect(page.get_by_role("heading", name="General")).to_be_visible()
    expect(page.get_by_role("heading", name="Data access")).to_be_visible()
    expect(page.get_by_role("heading", name="Advanced")).to_be_visible()

    seed = page.get_by_role("spinbutton", name="Default random seed")
    try:
        seed.fill("123")
        page.get_by_role("button", name="Save changes").click()
        expect(page.get_by_text("Settings saved.", exact=True)).to_be_visible()

        page.reload()
        expect(seed).to_have_value("123")
    finally:
        page.goto(f"{base_url}/settings")
        page.get_by_role("button", name="Reset to defaults").click()
        expect(page.get_by_text("Settings reset to defaults.", exact=True)).to_be_visible()

###############################################################################
def test_dataset_and_training_routes_render_workflow_surfaces(
    page: Page,
    base_url: str,
) -> None:
    response = page.goto(f"{base_url}/dataset")

    assert response is not None and response.ok
    expect(page.get_by_text("Data Source", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Load Dataset")).to_be_visible()

    response = page.goto(f"{base_url}/training")

    assert response is not None and response.ok
    expect(page.get_by_role("heading", name="XREPORT Transformer")).to_be_visible()
    expect(page.get_by_role("heading", name="Training Dashboard")).to_be_visible()

###############################################################################
def test_validation_route_reports_missing_dataset_as_user_error(
    page: Page,
    base_url: str,
) -> None:
    dataset_name = "e2e-missing-dataset"
    response = page.goto(f"{base_url}/dataset/validate/{dataset_name}")

    assert response is not None and response.ok
    page.get_by_role("button", name="Run Validation").click()
    expect(page.get_by_text(f"No data found for dataset: {dataset_name}.")).to_be_visible(
        timeout=20_000,
    )
