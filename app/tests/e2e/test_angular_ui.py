"""Focused browser checks for the canonical Angular client.

The standard Windows test runner starts the configured backend/frontend before
invoking this module through the client ``test:e2e`` script.
"""

import os

import pytest

playwright = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, sync_playwright


BASE_URL = os.environ.get("UI_BASE_URL", "http://127.0.0.1:8003").rstrip("/")

###############################################################################
@pytest.fixture()
def page() -> Page:
    with sync_playwright() as playwright_runtime:
        try:
            browser = playwright_runtime.chromium.launch(headless=True)
        except PlaywrightError as error:
            if "Executable doesn't exist" in str(error):
                pytest.skip("Playwright Chromium is not installed in this environment")
            raise
        browser_page = browser.new_page(viewport={"width": 390, "height": 844})
        yield browser_page
        browser.close()

###############################################################################
def test_root_redirect_and_mobile_route_matrix(page: Page) -> None:
    console_errors: list[str] = []
    page.on("console", lambda message: console_errors.append(message.text) if message.type in {"error", "warning"} else None)

    page.goto(f"{BASE_URL}/", wait_until="domcontentloaded")
    page.locator("main").wait_for()
    assert page.url.rstrip("/").endswith("/inference")

    for route in ("/inference", "/dataset", "/training", "/dataset/validate/mimic-cxr-training-sample"):
        page.goto(f"{BASE_URL}{route}", wait_until="domcontentloaded")
        page.locator("main").wait_for()
        widths = page.evaluate("({ body: document.body.scrollWidth, client: document.documentElement.clientWidth })")
        assert widths["body"] <= widths["client"] + 2, (route, widths)

    page.set_viewport_size({"width": 768, "height": 1024})
    page.goto(f"{BASE_URL}/dataset", wait_until="domcontentloaded")
    page.locator(".dataset-table").wait_for()
    tablet = page.evaluate(
        """
        () => {
          const table = document.querySelector('.dataset-table');
          return {
            bodyWidth: document.body.scrollWidth,
            clientWidth: document.documentElement.clientWidth,
            tableClientWidth: table?.clientWidth ?? 0,
            tableScrollWidth: table?.scrollWidth ?? 0,
          };
        }
        """
    )
    assert tablet["bodyWidth"] <= tablet["clientWidth"] + 2
    assert tablet["tableScrollWidth"] <= tablet["tableClientWidth"] + 1, tablet

    assert not console_errors

###############################################################################
def test_inference_model_browser_layout_and_selection(page: Page) -> None:
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(f"{BASE_URL}/inference", wait_until="domcontentloaded")
    page.locator(".model-card").first.wait_for()

    desktop = page.evaluate(
        """
        () => {
            const group = document.querySelector('.model-group');
            const list = group?.querySelector('.model-list');
            const cards = [...(group?.querySelectorAll('.model-card') ?? [])];
            const firstRowY = cards[0]?.getBoundingClientRect().y ?? 0;
            const firstRow = cards.filter((card) => Math.abs(card.getBoundingClientRect().y - firstRowY) < 1);
            const bottoms = firstRow.map((card) => card.getBoundingClientRect().bottom);
            const catalogBottom = document.querySelector('.catalog-panel')?.getBoundingClientRect().bottom ?? 0;
            const detailsBottom = document.querySelector('.model-details')?.getBoundingClientRect().bottom ?? 0;
            return {
                bodyWidth: document.body.scrollWidth,
                clientWidth: document.documentElement.clientWidth,
                columns: list ? getComputedStyle(list).gridTemplateColumns : '',
                cardCount: cards.length,
                firstRowBottomDelta: bottoms.length > 1 ? Math.max(...bottoms) - Math.min(...bottoms) : 0,
                panelBottomDelta: Math.abs(catalogBottom - detailsBottom),
            };
        }
        """
    )
    assert desktop["bodyWidth"] <= desktop["clientWidth"] + 2
    assert len(desktop["columns"].split(" ")) == 2
    assert desktop["cardCount"] >= 1
    assert desktop["firstRowBottomDelta"] < 1
    assert desktop["panelBottomDelta"] < 1

    public_cards = page.locator(".model-group").first.locator(".model-card")
    selected_card = public_cards.nth(1) if public_cards.count() > 1 else public_cards.first()
    selected_name = selected_card.locator("strong").inner_text()
    selected_card.click()
    assert page.locator(".model-details h3").inner_text() == selected_name

    first_name = public_cards.first().locator("strong").inner_text()
    page.get_by_placeholder("Filter by model, anatomy, or origin").fill(first_name.split(" ")[0])
    assert page.locator(".model-card").count() >= 1
    page.get_by_placeholder("Filter by model, anatomy, or origin").fill("")

    page.set_viewport_size({"width": 390, "height": 844})
    mobile = page.evaluate(
        """
        () => ({
            bodyWidth: document.body.scrollWidth,
            clientWidth: document.documentElement.clientWidth,
            columns: getComputedStyle(document.querySelector('.model-list')).gridTemplateColumns,
        })
        """
    )
    assert mobile["bodyWidth"] <= mobile["clientWidth"] + 2
    assert len(mobile["columns"].split(" ")) == 1

###############################################################################
def test_guidance_first_use_persistence_replay_and_keyboard(page: Page) -> None:
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(f"{BASE_URL}/inference", wait_until="domcontentloaded")
    page.locator("app-inference-page").wait_for()
    page.evaluate("localStorage.clear()")
    page.reload(wait_until="domcontentloaded")

    tip = page.get_by_text("A quick way to get started")
    tip.wait_for()
    page.get_by_role("button", name="Show me").click()
    tour = page.locator(".guided-tour-dialog")
    tour.wait_for()
    assert "Step 1 of 4" in tour.inner_text()
    page.get_by_role("button", name="Next").click()
    assert "Step 2 of 4" in tour.inner_text()
    page.get_by_role("button", name="Back").click()
    assert "Step 1 of 4" in tour.inner_text()
    page.get_by_role("button", name="Close walkthrough").click()
    assert page.evaluate("JSON.parse(localStorage.getItem('xreport.guidance.v1'))['inference-tour'].status") == "skipped"

    page.get_by_role("button", name="Help & tips").click()
    tips = page.locator(".guidance-modal")
    tips.wait_for()
    page.get_by_role("button", name="Show walkthrough").click()
    tour.wait_for()
    assert page.url.rstrip("/").endswith("/inference")
    assert "Step 1 of 4" in tour.inner_text()
    assert page.get_by_role("button", name="Skip walkthrough").count() == 0
    page.get_by_role("button", name="Next").click()
    page.get_by_role("button", name="Next").click()
    page.get_by_role("button", name="Next").click()
    assert "Step 4 of 4" in tour.inner_text()
    page.get_by_role("button", name="Finish").click()
    assert page.evaluate("JSON.parse(localStorage.getItem('xreport.guidance.v1'))['inference-tour'].status") == "completed"

    page.get_by_role("button", name="About model cards").click()
    popover = page.locator(".guidance-popover-panel")
    popover.wait_for()
    page.keyboard.press("Escape")
    popover.wait_for(state="hidden")
    assert page.get_by_role("button", name="About model cards").get_attribute("aria-expanded") == "false"

    page.set_viewport_size({"width": 390, "height": 844})
    page.reload(wait_until="domcontentloaded")
    page.locator("app-inference-page").wait_for()
    assert not tip.is_visible()
    widths = page.evaluate("({ body: document.body.scrollWidth, client: document.documentElement.clientWidth })")
    assert widths["body"] <= widths["client"] + 2

###############################################################################
def test_dataset_training_workflow_journey_and_tour(page: Page) -> None:
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(f"{BASE_URL}/dataset", wait_until="domcontentloaded")
    page.locator("app-dataset-page").wait_for()

    page.get_by_role("button", name="Help and tips").click()
    journey = page.locator(".guidance-workflow-journey")
    journey.wait_for()
    assert journey.locator(".guidance-journey-step").count() == 5
    assert "Data to training" in journey.inner_text()
    assert page.get_by_role("button", name="Walk me through it").count() == 1

    page.get_by_role("button", name="Walk me through it").click()
    tour = page.locator(".guided-tour-dialog")
    tour.wait_for()
    assert "Step 1 of 5" in tour.inner_text()

    page.get_by_role("button", name="Next").click()
    assert "Step 2 of 5" in tour.inner_text()
    page.get_by_role("button", name="Next").click()
    page.wait_for_url(f"{BASE_URL}/training")
    assert "Step 3 of 5" in tour.inner_text()
    assert page.locator('[data-guidance-target="training-dataset-list"]').count() == 1

    page.get_by_role("button", name="Next").click()
    assert "Step 4 of 5" in tour.inner_text()
    assert page.locator('[data-guidance-target="training-new-action"]').count() == 1
    page.get_by_role("button", name="Next").click()
    assert "Step 5 of 5" in tour.inner_text()
    assert page.locator('[data-guidance-target="training-resume-action"]').count() == 1
    page.get_by_role("button", name="Close walkthrough").click()
    assert page.locator(".guided-tour-dialog").count() == 0

###############################################################################
def test_dataset_viewer_validation_wizard_and_escape(page: Page) -> None:
    page.goto(f"{BASE_URL}/dataset", wait_until="domcontentloaded")
    page.locator('button[title="View images"]').first.wait_for()

    page.locator('button[title="View images"]').first.click()
    viewer = page.locator('[role="dialog"].viewer-modal')
    viewer.wait_for()
    assert "Image Viewer" in viewer.inner_text()
    page.keyboard.press("Escape")
    viewer.wait_for(state="hidden")

    page.get_by_role("button", name="Run validation").first.click()
    wizard = page.locator('[role="dialog"].wizard-modal')
    wizard.wait_for()
    assert wizard.locator(".wizard-metric").count() == 3
    page.keyboard.press("Escape")
    wizard.wait_for(state="hidden")

###############################################################################
def test_training_wizard_has_five_steps(page: Page) -> None:
    page.goto(f"{BASE_URL}/training", wait_until="domcontentloaded")
    row = page.locator("button.panel-row-main-button").first
    row.wait_for()
    row.click()
    page.get_by_role("button", name="Configure Training").click()

    wizard = page.locator('[role="dialog"].training-wizard-modal')
    wizard.wait_for()
    assert wizard.locator(".training-wizard-step").count() == 5
    wizard.locator(".training-wizard-step").nth(4).click()
    assert "Training Summary" in wizard.inner_text()
    page.keyboard.press("Escape")
    wizard.wait_for(state="hidden")

###############################################################################
def test_training_dashboard_visual_structure_and_responsive_layout(page: Page) -> None:
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(f"{BASE_URL}/training", wait_until="domcontentloaded")
    dashboard = page.locator(".training-dashboard")
    dashboard.wait_for()

    assert dashboard.locator(".dashboard-metric-card").count() == 5
    assert dashboard.locator(".chart-placeholder").count() == 2
    assert dashboard.get_by_role("status").inner_text() == "Idle"
    desktop_styles = page.evaluate(
        """
        () => {
            const dashboard = document.querySelector('.training-dashboard');
            const header = document.querySelector('.dashboard-header');
            const metrics = document.querySelector('.dashboard-metrics-grid');
            const placeholder = document.querySelector('.chart-placeholder');
            return {
                dashboardBorder: getComputedStyle(dashboard).borderTopStyle,
                headerDisplay: getComputedStyle(header).display,
                metricsDisplay: getComputedStyle(metrics).display,
                metricsColumns: getComputedStyle(metrics).gridTemplateColumns,
                placeholderBorder: getComputedStyle(placeholder).borderTopStyle,
            };
        }
        """
    )
    assert desktop_styles["dashboardBorder"] == "solid"
    assert desktop_styles["headerDisplay"] == "flex"
    assert desktop_styles["metricsDisplay"] == "grid"
    assert len(desktop_styles["metricsColumns"].split(" ")) == 5
    assert desktop_styles["placeholderBorder"] == "dashed"

    page.set_viewport_size({"width": 390, "height": 844})
    page.reload(wait_until="domcontentloaded")
    dashboard.wait_for()
    mobile_widths = page.evaluate(
        "({ body: document.body.scrollWidth, client: document.documentElement.clientWidth, columns: getComputedStyle(document.querySelector('.dashboard-metrics-grid')).gridTemplateColumns })"
    )
    assert mobile_widths["body"] <= mobile_widths["client"] + 2
    assert len(mobile_widths["columns"].split(" ")) == 1
