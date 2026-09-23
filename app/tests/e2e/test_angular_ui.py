"""Rendered Angular route and error-flow checks for the local application."""

import os
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
    expect(page.get_by_role("link", name="Reports")).to_be_visible()
    expect(page.get_by_role("link", name="Dataset")).to_be_visible()
    expect(page.get_by_role("link", name="Training")).to_be_visible()


###############################################################################
def test_reports_route_renders_history_or_explicit_empty_state(
    page: Page,
    base_url: str,
) -> None:
    response = page.goto(f"{base_url}/reports")

    assert response is not None and response.ok
    expect(page.get_by_role("heading", name="Reports")).to_be_visible()
    expect(page.get_by_role("link", name="Inference")).to_be_visible()
    expect(page.locator(".report-card, .reports-empty").first).to_be_visible()
    assert page.locator(".report-card, .reports-empty").count() > 0
    overflow = page.evaluate(
        "() => document.documentElement.scrollWidth <= document.documentElement.clientWidth"
    )
    assert overflow


###############################################################################
def test_reports_detail_edit_reload_and_confirmed_delete(
    page: Page,
    base_url: str,
    seeded_history: str,
) -> None:
    response = page.goto(f"{base_url}/reports")

    assert response is not None and response.ok
    card = page.locator(f'a.report-card[href="/reports/{seeded_history}"]')
    expect(card).to_be_visible()
    expect(card).to_contain_text("e2e/reports")
    card.click()

    expect(page).to_have_url(f"{base_url}/reports/{seeded_history}")
    findings = page.get_by_role("textbox", name="Findings").first
    expect(findings).to_have_value("Clear lungs.")
    findings.fill("Clear lungs after review.")
    page.get_by_role("button", name="Save edits").click()
    expect(page.get_by_text("Edits saved.", exact=False)).to_be_visible()

    page.reload()
    expect(page.get_by_role("textbox", name="Findings").first).to_have_value(
        "Clear lungs after review."
    )

    page.once("dialog", lambda dialog: dialog.accept())
    page.get_by_role("button", name="Delete session").click()
    expect(page).to_have_url(f"{base_url}/reports")
    expect(page.locator(f'a.report-card[href="/reports/{seeded_history}"]')).to_have_count(0)


###############################################################################
def test_desktop_catalogue_matches_details_height_and_keeps_internal_scroll(
    page: Page,
    base_url: str,
) -> None:
    page.set_viewport_size({"width": 1440, "height": 900})
    response = page.goto(f"{base_url}/inference")

    assert response is not None and response.ok
    expect(page.get_by_text("Model catalogue", exact=True)).to_be_visible()
    metrics = page.evaluate(
        """
        () => {
          const selection = document.querySelector('.model-selection');
          const catalog = document.querySelector('.catalog-panel');
          const details = document.querySelector('.model-details');
          const groups = document.querySelector('.model-groups');
          return {
            selectionContainsCatalog: Boolean(selection && catalog && selection.contains(catalog)),
            catalogHeight: catalog?.getBoundingClientRect().height ?? 0,
            detailsHeight: details?.getBoundingClientRect().height ?? 0,
            overflowY: groups ? getComputedStyle(groups).overflowY : '',
          };
        }
        """,
    )
    assert metrics["selectionContainsCatalog"]
    assert abs(metrics["catalogHeight"] - metrics["detailsHeight"]) <= 1
    assert metrics["overflowY"] in {"auto", "scroll"}

    page.get_by_placeholder("Filter by model, anatomy, or origin").fill("CheXOne")
    filtered_metrics = page.evaluate(
        """
        () => ({
          catalogHeight: document.querySelector('.catalog-panel')?.getBoundingClientRect().height ?? 0,
          detailsHeight: document.querySelector('.model-details')?.getBoundingClientRect().height ?? 0,
        })
        """,
    )
    assert abs(filtered_metrics["catalogHeight"] - filtered_metrics["detailsHeight"]) <= 1

###############################################################################
def test_startup_gate_holds_inference_until_backend_health(
    page: Page,
    base_url: str,
) -> None:
    """The shell stays visible and suppresses page API work until health recovers."""
    health_calls = 0
    requested_urls: list[str] = []

    def capture_request(request) -> None:
        requested_urls.append(request.url)

    def respond_to_health(route) -> None:
        nonlocal health_calls
        health_calls += 1
        if health_calls < 3:
            route.fulfill(
                status=503,
                content_type="application/json",
                body='{"detail":"backend is still starting"}',
            )
            return
        route.fulfill(
            status=200,
            content_type="application/json",
            body='{"status":"ok"}',
        )

    page.on("request", capture_request)
    page.route("**/api/health", respond_to_health)
    page.set_viewport_size({"width": 1024, "height": 720})
    response = page.goto(f"{base_url}/")

    assert response is not None and response.ok
    expect(page.get_by_role("heading", name="Preparing XREPORT")).to_be_visible()
    startup_images = page.locator(".startup-xray-image")
    assert startup_images.count() > 0
    page.wait_for_function(
        "() => { const images = [...document.querySelectorAll('.startup-xray-image')]; "
        "return images.length > 0 && images.every(image => image instanceof HTMLImageElement "
        "&& image.getAttribute('src') === 'startup-radiograph.png' "
        "&& image.getAttribute('alt') === '' && image.getAttribute('aria-hidden') === 'true' "
        "&& image.complete && image.naturalWidth > 0); }",
        timeout=5_000,
    )
    expect(page.locator("app-inference-page")).to_have_count(0)
    expect(page.get_by_text("Model catalogue", exact=True)).to_have_count(0)
    assert not any("/api/inference/models" in url for url in requested_urls)

    qa_dir = _e2e_screenshot_dir("")
    page.screenshot(path=str(qa_dir / "xreport-startup-gate.png"), full_page=False)

    expect(page.get_by_text("Model catalogue", exact=True)).to_be_visible(timeout=10_000)
    assert health_calls == 3
    page.screenshot(path=str(qa_dir / "xreport-startup-ready.png"), full_page=False)
    page.unroute("**/api/health", respond_to_health)

###############################################################################
def test_chexone_details_render_findings_only_catalog_contract(
    page: Page,
    base_url: str,
) -> None:
    console_errors: list[str] = []
    request_failures: list[str] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text)
        if message.type == "error"
        else None,
    )
    page.on(
        "requestfailed",
        lambda request: request_failures.append(f"{request.method} {request.url}"),
    )

    response = page.goto(f"{base_url}/inference")
    assert response is not None and response.ok
    page.get_by_role("button", name=re.compile(r"CheXOne")).click()

    details = page.locator(".model-details")
    expect(details.get_by_role("heading", name="CheXOne")).to_be_visible()
    output_row = details.locator("dt").filter(has_text="Output").locator("xpath=..")
    expect(output_row.locator("dd")).to_have_text("Findings")
    expect(details.locator(".capability-list").get_by_text("Findings", exact=True)).to_be_visible()
    expect(details.get_by_text("Impression", exact=True)).to_have_count(0)
    assert not console_errors, console_errors
    assert not request_failures, request_failures

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
    qa_dir = _e2e_screenshot_dir("")
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


###############################################################################
def _capture_s10_browser_errors(page: Page) -> dict[str, list[str]]:
    errors = {"console": [], "page": [], "requests": []}
    page.on(
        "console",
        lambda message: errors["console"].append(message.text)
        if message.type == "error"
        else None,
    )
    page.on("pageerror", lambda error: errors["page"].append(str(error)))
    page.on(
        "requestfailed",
        lambda request: errors["requests"].append(
            f"{request.method} {request.url}: {request.failure}"
        ),
    )
    return errors


def _e2e_screenshot_dir(default_relative: str) -> Path:
    configured = os.environ.get("XREPORT_E2E_SCREENSHOT_DIR", "").strip()
    qa_dir = Path(configured) if configured else Path(__file__).parents[3] / "assets" / "QA" / default_relative
    qa_dir.mkdir(parents=True, exist_ok=True)
    return qa_dir


def _s10_qa_dir() -> Path:
    configured = os.environ.get("XREPORT_S10_SCREENSHOT_DIR", "").strip()
    if configured:
        qa_dir = Path(configured)
        qa_dir.mkdir(parents=True, exist_ok=True)
        return qa_dir
    return _e2e_screenshot_dir("validation_campaign/s10")


def _expect_s10_active_navigation(page: Page, label: str) -> None:
    active_links = page.locator(
        'nav[aria-label="Primary navigation"] a.app-nav-button.active'
    )
    expect(active_links).to_have_count(1)
    expect(active_links).to_have_attribute("aria-label", label)


def _expect_s10_route_surface(page: Page, path: str) -> None:
    if path == "/inference":
        expect(
            page.get_by_role("heading", name="Turn a radiograph into a draft report")
        ).to_be_visible()
    elif path == "/reports":
        expect(page.get_by_role("heading", name="Reports")).to_be_visible()
    elif path.startswith("/reports/"):
        expect(page.get_by_role("textbox", name="Findings").first).to_be_visible()
    elif path == "/dataset":
        expect(page.get_by_text("Data Source", exact=True)).to_be_visible()
    elif path == "/training":
        expect(page.get_by_role("heading", name="Training Dashboard")).to_be_visible()
    elif path.startswith("/dataset/validate/"):
        expect(page.get_by_role("heading", name="Validation Wizard")).to_be_visible()
    elif path == "/settings":
        expect(page.get_by_role("heading", name="Settings")).to_be_visible()
    else:
        raise AssertionError(f"Unexpected S10 route: {path}")


###############################################################################
def test_s10_direct_routes_refresh_navigation_and_clean_browser(
    page: Page,
    base_url: str,
    seeded_history: str,
) -> None:
    """Load and refresh every routed surface, then traverse the main navigation."""
    qa_dir = _s10_qa_dir()
    errors = _capture_s10_browser_errors(page)

    validation_dataset = "e2e-s10-validation-route-fixture"
    validation_requests: list[str] = []

    def serve_validation_fixture(route) -> None:
        validation_requests.append(route.request.url)
        route.fulfill(
            status=200,
            json={
                "dataset_name": validation_dataset,
                "date": "2026-09-22T12:00:00Z",
                "sample_size": 0.5,
                "metrics": ["image_statistics"],
                "image_statistics": {
                    "count": 1,
                    "mean_height": 1024,
                    "mean_width": 1024,
                    "mean_pixel_value": 128.0,
                    "std_pixel_value": 10.0,
                    "mean_noise_std": 0.0,
                    "mean_noise_ratio": 0.0,
                },
            },
        )

    page.route(
        f"**/api/validation/reports/{validation_dataset}",
        serve_validation_fixture,
    )

    response = page.goto(f"{base_url}/")
    assert response is not None and response.ok
    expect(page).to_have_url(f"{base_url}/inference")
    _expect_s10_route_surface(page, "/inference")
    _expect_s10_active_navigation(page, "Inference")

    routes = [
        ("/inference", "Inference"),
        ("/reports", "Reports"),
        (f"/reports/{seeded_history}", "Reports"),
        ("/dataset", "Dataset"),
        ("/training", "Training"),
        (f"/dataset/validate/{validation_dataset}", "Dataset"),
        ("/settings", "Settings"),
    ]
    for path, active_label in routes:
        response = page.goto(f"{base_url}{path}")
        assert response is not None and response.ok, path
        expect(page).to_have_url(f"{base_url}{path}")
        _expect_s10_route_surface(page, path)
        _expect_s10_active_navigation(page, active_label)
        if path == "/inference":
            page.screenshot(path=str(qa_dir / "route-inference.png"), full_page=False)
        elif path.startswith("/reports/"):
            page.screenshot(path=str(qa_dir / "route-report-detail.png"), full_page=False)

        response = page.reload()
        assert response is not None and response.ok, f"refresh {path}"
        expect(page).to_have_url(f"{base_url}{path}")
        _expect_s10_route_surface(page, path)
        _expect_s10_active_navigation(page, active_label)

    assert len(validation_requests) == 2, validation_requests

    for label, path in (
        ("Inference", "/inference"),
        ("Reports", "/reports"),
        ("Dataset", "/dataset"),
        ("Training", "/training"),
        ("Settings", "/settings"),
    ):
        page.get_by_role("link", name=label, exact=True).click()
        expect(page).to_have_url(f"{base_url}{path}")
        _expect_s10_route_surface(page, path)
        _expect_s10_active_navigation(page, label)

    assert not errors["console"], errors["console"]
    assert not errors["page"], errors["page"]
    assert not errors["requests"], errors["requests"]


###############################################################################
def test_s10_theme_preferences_follow_system_and_persist_across_routes(
    page: Page,
    base_url: str,
) -> None:
    qa_dir = _s10_qa_dir()
    errors = _capture_s10_browser_errors(page)
    page.emulate_media(color_scheme="light")
    response = page.goto(f"{base_url}/inference")
    assert response is not None and response.ok

    system = page.get_by_role("button", name="Use System theme")
    light = page.get_by_role("button", name="Use Light theme")
    dark = page.get_by_role("button", name="Use Dark theme")
    expect(system).to_have_attribute("aria-pressed", "true")
    expect(page.locator("html")).to_have_attribute("data-theme", "light")

    light.click()
    expect(light).to_have_attribute("aria-pressed", "true")
    expect(page.locator("html")).to_have_attribute("data-theme", "light")
    assert page.evaluate("() => localStorage.getItem('theme-preference')") == "light"
    page.screenshot(path=str(qa_dir / "theme-light-inference.png"), full_page=False)
    response = page.reload()
    assert response is not None and response.ok
    expect(light).to_have_attribute("aria-pressed", "true")
    expect(page.locator("html")).to_have_attribute("data-theme", "light")
    page.emulate_media(color_scheme="dark")
    expect(page.locator("html")).to_have_attribute("data-theme", "light")

    dark.click()
    expect(dark).to_have_attribute("aria-pressed", "true")
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    assert page.evaluate("() => localStorage.getItem('theme-preference')") == "dark"
    response = page.reload()
    assert response is not None and response.ok
    expect(dark).to_have_attribute("aria-pressed", "true")
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    page.emulate_media(color_scheme="light")
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")

    system.click()
    expect(system).to_have_attribute("aria-pressed", "true")
    expect(page.locator("html")).to_have_attribute("data-theme", "light")
    assert page.evaluate("() => localStorage.getItem('theme-preference')") == "system"
    page.emulate_media(color_scheme="dark")
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    expect(system).to_have_attribute("aria-pressed", "true")

    page.get_by_role("link", name="Settings", exact=True).click()
    expect(page).to_have_url(f"{base_url}/settings")
    _expect_s10_active_navigation(page, "Settings")
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    page.screenshot(path=str(qa_dir / "theme-system-dark-settings.png"), full_page=False)
    page.emulate_media(color_scheme="light")
    expect(page.locator("html")).to_have_attribute("data-theme", "light")
    expect(system).to_have_attribute("aria-pressed", "true")
    response = page.reload()
    assert response is not None and response.ok
    expect(system).to_have_attribute("aria-pressed", "true")
    expect(page.locator("html")).to_have_attribute("data-theme", "light")
    assert page.evaluate("() => localStorage.getItem('theme-preference')") == "system"
    page.emulate_media(color_scheme="dark")
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")

    assert not errors["console"], errors["console"]
    assert not errors["page"], errors["page"]
    assert not errors["requests"], errors["requests"]


###############################################################################
def test_s10_guidance_version_dismiss_skip_completion_and_manual_replay(
    page: Page,
    base_url: str,
) -> None:
    qa_dir = _s10_qa_dir()
    errors = _capture_s10_browser_errors(page)
    page.add_init_script(
        """() => {
          const key = 'xreport.guidance.v1';
          if (!localStorage.getItem(key)) {
            localStorage.setItem(key, JSON.stringify({
              'inference-first-use': { version: 0, status: 'dismissed' }
            }));
          }
        }""",
    )
    response = page.goto(f"{base_url}/inference")
    assert response is not None and response.ok

    first_use_tip = page.locator('[data-guidance-id="inference-first-use"]')
    expect(first_use_tip).to_be_visible()
    stored_guidance = page.evaluate("() => JSON.parse(localStorage.getItem('xreport.guidance.v1'))")
    assert stored_guidance["inference-first-use"] == {"version": 1, "status": "seen"}
    page.screenshot(path=str(qa_dir / "guidance-first-use-tip.png"), full_page=False)
    first_use_tip.get_by_role("button", name="Dismiss tip").click()
    expect(first_use_tip).to_have_count(0)
    stored_guidance = page.evaluate("() => JSON.parse(localStorage.getItem('xreport.guidance.v1'))")
    assert stored_guidance["inference-first-use"] == {"version": 1, "status": "dismissed"}

    def start_inference_tour() -> None:
        page.get_by_role("button", name="Help and tips").click()
        expect(page.get_by_role("heading", name="Tips & Tricks")).to_be_visible()
        page.get_by_role("button", name="Show walkthrough").click()
        expect(page.get_by_role("heading", name="Choose a model")).to_be_visible()

    start_inference_tour()
    tour_dialog = page.locator(".guided-tour-dialog")
    tour_dialog.get_by_role("button", name="Close walkthrough").click()
    expect(tour_dialog).to_have_count(0)
    stored_guidance = page.evaluate("() => JSON.parse(localStorage.getItem('xreport.guidance.v1'))")
    assert stored_guidance["inference-tour"] == {"version": 1, "status": "skipped"}

    response = page.reload()
    assert response is not None and response.ok
    expect(first_use_tip).to_have_count(0)
    expect(page.locator(".guided-tour-dialog")).to_have_count(0)
    start_inference_tour()
    tour_dialog = page.locator(".guided-tour-dialog")
    for _ in range(3):
        tour_dialog.get_by_role("button", name="Next").click()
    expect(page.get_by_role("heading", name="Review the draft")).to_be_visible()
    page.screenshot(path=str(qa_dir / "guidance-tour-completion.png"), full_page=False)
    tour_dialog.get_by_role("button", name="Finish").click()
    expect(tour_dialog).to_have_count(0)
    stored_guidance = page.evaluate("() => JSON.parse(localStorage.getItem('xreport.guidance.v1'))")
    assert stored_guidance["inference-tour"] == {"version": 1, "status": "completed"}

    response = page.reload()
    assert response is not None and response.ok
    expect(first_use_tip).to_have_count(0)
    expect(page.locator(".guided-tour-dialog")).to_have_count(0)
    start_inference_tour()
    expect(page.locator(".guided-tour-dialog")).to_be_visible()

    assert not errors["console"], errors["console"]
    assert not errors["page"], errors["page"]
    assert not errors["requests"], errors["requests"]
