"""Browser checks for the XREPORT Studio application (dataset and training)."""

from __future__ import annotations

import re

from playwright.sync_api import Page, expect
import pytest


@pytest.fixture(scope="session")
def base_url(studio_base_url: str) -> str:
    return studio_base_url


###############################################################################
def test_studio_navigation_contains_only_development_pages(
    page: Page, base_url: str
) -> None:
    response = page.goto(f"{base_url}/dataset")

    assert response is not None and response.ok
    navigation = page.get_by_role("navigation", name="Primary navigation")
    expect(navigation.get_by_role("link", name="Dataset", exact=True)).to_be_visible()
    expect(navigation.get_by_role("link", name="Training", exact=True)).to_be_visible()
    expect(navigation.get_by_role("link", name="Settings", exact=True)).to_be_visible()
    expect(navigation.get_by_role("link", name="Inference", exact=True)).to_have_count(0)
    expect(navigation.get_by_role("link", name="Reports", exact=True)).to_have_count(0)
    expect(navigation.get_by_text("XREPORT Studio")).to_be_visible()

    page.goto(f"{base_url}/inference")
    expect(page).to_have_url(f"{base_url}/dataset")


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


###############################################################################
@pytest.mark.parametrize("width,height", [(320, 640), (640, 800), (1440, 900)])
def test_studio_routes_fit_narrow_and_wide_viewports(
    page: Page, base_url: str, width: int, height: int
) -> None:
    page.set_viewport_size({"width": width, "height": height})
    for route, heading in (
        ("/dataset", "Available Datasets"),
        ("/training", "Training Dashboard"),
        ("/settings", "Settings"),
    ):
        response = page.goto(f"{base_url}{route}")
        assert response is not None and response.ok, route
        if route == "/dataset":
            expect(page.get_by_text(heading, exact=True)).to_be_visible()
        else:
            expect(page.get_by_role("heading", name=heading)).to_be_visible()
        metrics = page.evaluate(
            "() => ({ document: document.documentElement.scrollWidth, viewport: window.innerWidth })"
        )
        assert metrics["document"] <= metrics["viewport"], route


###############################################################################
def test_training_panels_collapse_with_the_keyboard(page: Page, base_url: str) -> None:
    page.set_viewport_size({"width": 1024, "height": 720})
    page.goto(f"{base_url}/training")
    expect(page.get_by_role("heading", name="XREPORT Transformer")).to_be_visible()

    toggles = page.locator("main button.panel-collapse-toggle")
    expect(toggles).to_have_count(2)
    collapse_button = toggles.first
    collapse_button.focus()
    expect(collapse_button).to_have_attribute("aria-expanded", "true")
    page.keyboard.press("Space")
    expect(collapse_button).to_have_attribute("aria-expanded", "false")
    page.keyboard.press("Enter")
    expect(collapse_button).to_have_attribute("aria-expanded", re.compile("true"))
