"""Rendered Angular route and error-flow checks for the local application."""

import re
from pathlib import Path

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
    expect(page.get_by_role("heading", name="Data access")).not_to_be_visible()
    expect(page.get_by_role("tab", name="Data access")).to_be_visible()
    expect(page.get_by_role("tab", name="Advanced")).to_be_visible()

    page.get_by_role("tab", name="Data access").click()
    expect(page.get_by_role("heading", name="Data access")).to_be_visible()
    expect(page.get_by_role("checkbox", name="Enabled")).to_be_visible()

    page.get_by_role("tab", name="Advanced").click()
    expect(page.get_by_role("heading", name="Advanced")).to_be_visible()
    expect(page.get_by_role("spinbutton", name="Job status polling interval seconds")).to_be_visible()
    page.get_by_role("tab", name="General").click()

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

###############################################################################
def test_affected_pages_render_responsive_layouts_and_capture_qa_evidence(
    page: Page,
    base_url: str,
) -> None:
    """Exercise the affected surfaces at a narrow viewport and save evidence."""
    qa_dir = Path(__file__).parents[3] / "assets" / "QA"
    qa_dir.mkdir(parents=True, exist_ok=True)
    console_errors: list[str] = []
    request_failures: list[str] = []
    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
    page.on("requestfailed", lambda request: request_failures.append(f"{request.method} {request.url}"))

    page.set_viewport_size({"width": 900, "height": 700})
    response = page.goto(f"{base_url}/inference")
    assert response is not None and response.ok
    expect(page.get_by_text("Model catalogue", exact=True)).to_be_visible()
    banner = page.locator(".model-info-banner")
    expect(banner).to_be_visible()
    page.screenshot(path=str(qa_dir / "xreport-inference-narrow-dark.png"), full_page=False)

    inference_metrics = page.evaluate(
        """
        () => {
          const catalog = document.querySelector('.catalog-panel');
          const groups = document.querySelector('.model-groups');
          const cards = [...document.querySelectorAll('.model-card')];
          return {
            catalogHeight: catalog?.getBoundingClientRect().height ?? 0,
            groupsHeight: groups?.getBoundingClientRect().height ?? 0,
            overflowY: groups ? getComputedStyle(groups).overflowY : '',
            cardHeightsByList: [...document.querySelectorAll('.model-list')].map((list) =>
              [...list.querySelectorAll('.model-card')].map((card) =>
                Math.round(card.getBoundingClientRect().height * 100) / 100,
              ),
            ),
          };
        }
        """,
    )
    assert inference_metrics["catalogHeight"] > 0
    assert inference_metrics["groupsHeight"] > 0
    assert inference_metrics["overflowY"] in {"auto", "scroll"}
    assert all(
        len(set(card_heights)) == 1
        for card_heights in inference_metrics["cardHeightsByList"]
        if card_heights
    )

    page.get_by_role("button", name="Dismiss model information").click()
    expect(banner).to_have_count(0)
    cards = page.locator(".model-card")
    if cards.count() > 1:
        cards.nth(1).click()
        expect(page.locator(".model-info-banner")).to_be_visible()

    page.set_viewport_size({"width": 640, "height": 800})
    page.goto(f"{base_url}/dataset")
    expect(page.get_by_text("Available Datasets", exact=True)).to_be_visible()
    page.screenshot(path=str(qa_dir / "xreport-dataset-narrow-dark.png"), full_page=False)
    dataset_heights = page.locator(".dataset-table-row").evaluate_all(
        "els => els.map((element) => Math.round(element.getBoundingClientRect().height * 100) / 100)",
    )
    assert dataset_heights
    assert len(set(dataset_heights)) == 1

    page.goto(f"{base_url}/settings")
    expect(page.get_by_role("tab", name="General")).to_be_visible()
    page.screenshot(path=str(qa_dir / "xreport-settings-narrow-dark.png"), full_page=False)
    for tab_name, heading_name in (("General", "General"), ("Data access", "Data access"), ("Advanced", "Advanced")):
        page.get_by_role("tab", name=tab_name).click()
        expect(page.get_by_role("heading", name=heading_name, level=2)).to_be_visible()
        assert page.locator(".settings-field").count() > 0

    settings_metrics = page.evaluate(
        """
        () => [...document.querySelectorAll('.settings-field')].map((row) => ({
          columns: getComputedStyle(row).gridTemplateColumns,
          copyLeft: row.querySelector('.settings-field-copy')?.getBoundingClientRect().left,
          controlLeft: row.querySelector('.settings-field-control')?.getBoundingClientRect().left,
        }))
        """,
    )
    assert settings_metrics
    assert all(len(metric["columns"].split()) == 1 for metric in settings_metrics)

    page.get_by_role("button", name="Use Light theme").click()
    expect(page.locator("html")).to_have_attribute("data-theme", "light")
    page.screenshot(path=str(qa_dir / "xreport-settings-narrow-light.png"), full_page=False)
    page.get_by_role("button", name="Use Dark theme").click()
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")

    assert not console_errors, console_errors
    assert not request_failures, request_failures
