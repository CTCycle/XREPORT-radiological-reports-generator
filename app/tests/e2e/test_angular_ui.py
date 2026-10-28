"""Rendered Angular route and error-flow checks for the local application."""

import os
import re
import zipfile
from pathlib import Path

from playwright.sync_api import Page, Route, TimeoutError as PlaywrightTimeoutError, expect

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
    expect(page.get_by_role("link", name="Dataset")).to_have_count(0)
    expect(page.get_by_role("link", name="Training")).to_have_count(0)

    page.goto(f"{base_url}/dataset")
    expect(page).to_have_url(f"{base_url}/inference")

###############################################################################
def test_model_update_check_shows_progress_then_result(
    page: Page,
    base_url: str,
) -> None:
    pending: list[Route] = []
    page.route("**/api/inference/models/check-update", lambda route: pending.append(route))
    page.goto(f"{base_url}/inference")
    page.get_by_role("button", name=re.compile("CXRMate Multi")).click()

    page.get_by_role("button", name="Check for updates").click()
    expect(page.get_by_role("button", name="Checking…")).to_be_disabled()
    expect(page.get_by_text("Checking Hugging Face for a newer revision…")).to_be_visible()

    while not pending:
        page.wait_for_timeout(50)
    pending[0].fulfill(
        json={
            "model_ref": "huggingface:aehrc/cxrmate-multi-tf",
            "repository_id": "aehrc/cxrmate-multi-tf",
            "installed_revision": "a" * 40,
            "latest_revision": "b" * 40,
            "update_available": True,
            "source": "https://huggingface.co/aehrc/cxrmate-multi-tf",
            "checked_at": "2026-01-01T00:00:00+00:00",
            "error": None,
        }
    )
    expect(page.get_by_text(f"Update available: {'a' * 12} → {'b' * 12}")).to_be_visible()
    expect(page.get_by_role("button", name="Check for updates")).to_be_enabled()

###############################################################################
def test_reports_route_renders_history_or_explicit_empty_state(
    page: Page,
    base_url: str,
) -> None:
    response = page.goto(f"{base_url}/reports")

    assert response is not None and response.ok
    expect(page.get_by_role("heading", name="Reports")).to_be_visible()
    expect(page.get_by_role("link", name="Inference", exact=True)).to_be_visible()
    expect(page.locator(".report-card, .reports-empty").first).to_be_visible()
    assert page.locator(".report-card, .reports-empty").count() > 0
    overflow = page.evaluate(
        "() => document.documentElement.scrollWidth <= document.documentElement.clientWidth"
    )
    assert overflow


def test_checkpoint_import_rejects_manifestless_archive_in_live_ui(
    page: Page,
    base_url: str,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "manifestless-four-core.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("saved_model.keras", b"synthetic model placeholder")
        bundle.writestr("configuration/configuration.json", b"{}")
        bundle.writestr("configuration/metadata.json", b"{}")
        bundle.writestr("configuration/session_history.json", b"[]")

    response = page.goto(f"{base_url}/inference")
    assert response is not None and response.ok
    page.get_by_role("button", name="Import checkpoint").click()
    dialog = page.get_by_role("dialog", name="Import checkpoint")
    dialog.get_by_label("Checkpoint archive").set_input_files(str(archive))
    dialog.get_by_role("button", name="Import", exact=True).click()

    expect(dialog.get_by_role("status")).to_contain_text(
        "missing required manifest.json"
    )
    expect(dialog.get_by_role("progressbar")).to_have_count(0)
    expect(dialog.get_by_role("button", name="Cancel import")).to_have_count(0)

###############################################################################
def test_reports_detail_edit_reload_and_confirmed_delete(
    page: Page,
    base_url: str,
    seeded_history: str,
) -> None:
    qa_dir = _e2e_screenshot_dir("")
    response = page.goto(f"{base_url}/reports")

    assert response is not None and response.ok
    card = page.locator(f'a.report-card[href="/reports/{seeded_history}"]')
    expect(card).to_be_visible()
    expect(card).to_contain_text("e2e/reports")
    card.click()

    expect(page).to_have_url(f"{base_url}/reports/{seeded_history}")
    findings = page.get_by_role("textbox", name="Findings").first
    expect(findings).to_have_value("Clear lungs.")

    findings.fill("Unsaved review draft.")
    unload_dialog_types: list[str] = []

    def dismiss_unload_dialog(dialog) -> None:
        unload_dialog_types.append(dialog.type)
        dialog.dismiss()

    page.once("dialog", dismiss_unload_dialog)
    try:
        page.reload(timeout=1500)
    except PlaywrightTimeoutError:
        # Dismissing beforeunload cancels the reload, so no navigation completes.
        pass
    assert unload_dialog_types == ["beforeunload"]
    expect(page).to_have_url(f"{base_url}/reports/{seeded_history}")
    findings = page.get_by_role("textbox", name="Findings").first
    expect(findings).to_have_value("Unsaved review draft.")

    page.once("dialog", lambda dialog: dialog.dismiss())
    page.get_by_role("link", name="Back to Reports").click()
    expect(page).to_have_url(f"{base_url}/reports/{seeded_history}")
    expect(findings).to_have_value("Unsaved review draft.")

    page.once("dialog", lambda dialog: dialog.dismiss())
    page.evaluate("window.history.back()")
    expect(page).to_have_url(f"{base_url}/reports/{seeded_history}")
    expect(findings).to_have_value("Unsaved review draft.")

    page.once("dialog", lambda dialog: dialog.accept())
    page.get_by_role("link", name="Back to Reports").click()
    expect(page).to_have_url(f"{base_url}/reports")

    page.locator(f'a.report-card[href="/reports/{seeded_history}"]').click()
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

    page.set_viewport_size({"width": 390, "height": 844})
    detail_metrics = page.evaluate(
        """
        () => {
          const content = document.querySelector('.main-layout-content');
          return {
            viewportWidth: window.innerWidth,
            documentWidth: document.documentElement.scrollWidth,
            contentClientHeight: content?.clientHeight ?? 0,
            contentScrollHeight: content?.scrollHeight ?? 0,
            contentOverflowY: content ? getComputedStyle(content).overflowY : '',
          };
        }
        """,
    )
    page.screenshot(
        path=str(qa_dir / "route-report-detail-mobile.png"),
        full_page=False,
    )
    assert detail_metrics["documentWidth"] <= detail_metrics["viewportWidth"]
    assert detail_metrics["contentScrollHeight"] > detail_metrics["contentClientHeight"]
    assert detail_metrics["contentOverflowY"] in {"auto", "scroll"}
    page.mouse.move(385, 500)
    page.mouse.wheel(0, 700)
    page.wait_for_function(
        "() => document.querySelector('.main-layout-content').scrollTop > 0"
    )

    findings.fill("Unsaved draft before deleting the session.")
    page.once("dialog", lambda dialog: dialog.accept())
    page.get_by_role("button", name="Delete session").click()
    expect(page).to_have_url(f"{base_url}/reports")
    expect(page.locator(f'a.report-card[href="/reports/{seeded_history}"]')).to_have_count(0)

###############################################################################
def test_inference_draft_edit_save_and_reports_recovery(
    page: Page,
    base_url: str,
    seeded_history: str,
) -> None:
    original_reports = [
        "Findings\nClear lungs.\n\nImpression\nNo acute disease.",
        "Findings\nMild bibasal opacity.\n\nImpression\nFollow-up recommended.",
    ]
    model = {
        "model_ref": "huggingface:e2e/draft-model",
        "provider": "huggingface",
        "origin": "public",
        "display_name": "E2E Draft Model",
        "description": "Deterministic UI fixture model",
        "status": "ready",
        "enabled": True,
        "validation_status": "passed",
        "validation_receipt_status": "passed",
        "category": "radiology",
        "recommended": False,
        "research_only": True,
        "gated": False,
        "access_policy": "open",
        "anatomy_coverage": "chest_xray",
        "hardware_demand": "moderate",
        "input_semantics": "multiple_current_views",
        "capabilities": {
            "clinical_context": False,
            "multiple_current_views": True,
            "findings": True,
            "impression": True,
            "grounding": False,
        },
        "trust_remote_code": False,
        "remote_code_approved": False,
        "output_sections": ["findings", "impression"],
        "max_current_images": 2,
        "installation_state": "installed",
        "integrity_status": "verified",
        "available_actions": [],
    }

    page.route(
        "**/api/inference/models",
        lambda route: route.fulfill(json={"models": [model], "providers": {}}),
    )
    page.route(
        "**/api/inference/generate",
        lambda route: route.fulfill(
            json={"job_id": "e2e-persistent-draft", "poll_interval": 0.1}
        ),
    )

    def complete_job(route: Route) -> None:
        route.fulfill(
            json={
                "job_id": "e2e-persistent-draft",
                "job_type": "inference",
                "status": "completed",
                "poll_interval": 0.1,
                "progress": 100,
                "result": {
                    "request_id": seeded_history,
                    "reports": {
                        "e2e-first.png": original_reports[0],
                        "e2e-second.png": original_reports[1],
                    },
                    "reports_ordered": original_reports,
                    "report_filenames": ["e2e-first.png", "e2e-second.png"],
                    "count": 2,
                    "display_sections": {},
                    "provenance": {"report_scope": "image"},
                },
                "error": None,
            }
        )

    page.route("**/api/jobs/e2e-persistent-draft", complete_job)
    response = page.goto(f"{base_url}/inference")
    assert response is not None and response.ok

    image_input = page.locator('input[type="file"]')
    expect(image_input).to_have_attribute("multiple", "")
    image_input.set_input_files(
        [
            {"name": "e2e-first.png", "mimeType": "image/png", "buffer": b"first"},
            {"name": "e2e-second.png", "mimeType": "image/png", "buffer": b"second"},
        ]
    )
    page.get_by_role("button", name="Generate draft", exact=True).click()
    expect(page.get_by_text("Generated report saved.", exact=True)).to_be_visible()
    findings = page.get_by_role("textbox", name="Findings").first
    expect(findings).to_have_value("Clear lungs.")
    findings.fill("Clear lungs after review.")
    page.get_by_role("button", name="Save edits").click()
    expect(page.get_by_text("Edits saved.", exact=True)).to_be_visible()

    page.get_by_role("link", name="View in Reports").click()
    expect(page).to_have_url(f"{base_url}/reports/{seeded_history}")
    expect(page.get_by_role("textbox", name="Findings").first).to_have_value(
        "Clear lungs after review."
    )
    page.locator("details.original-output summary").first.click()
    original_output = page.locator("details.original-output .original-text").first
    expect(original_output).to_contain_text("Findings")
    expect(original_output).to_contain_text("Clear lungs.")

    page.reload()
    expect(page.get_by_role("textbox", name="Findings").first).to_have_value(
        "Clear lungs after review."
    )

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
    page.screenshot(path="G:/Projects/Repositories/Active projects/XREPORT Radiological Reports/assets/QA/catalogue-rows-desktop.png")  # TEMP

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
    expect(page.locator("#startup-title")).to_be_visible()
    startup_progress = page.get_by_role("progressbar", name="Initializing XREPORT Inference")
    expect(startup_progress).to_be_visible()
    expect(startup_progress).to_have_attribute("aria-valuemin", "0")
    expect(startup_progress).to_have_attribute("aria-valuemax", "100")
    expect(page.locator("app-startup-screen .startup-status-line")).to_be_visible()
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
def test_startup_timing_retry_and_post_ready_feature_error(
    page: Page,
    base_url: str,
) -> None:
    """Slow/unavailable recovery and a later catalogue error stay in the app."""
    health_calls = 0
    health_available = False
    page.clock.install()

    def respond_to_health(route) -> None:
        nonlocal health_calls
        health_calls += 1
        if not health_available:
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

    def fail_model_catalogue(route) -> None:
        route.fulfill(
            status=503,
            content_type="application/json",
            body='{"detail":"Synthetic model catalogue outage"}',
        )

    page.route("**/api/health", respond_to_health)
    page.route("**/api/inference/models", fail_model_catalogue)
    page.set_viewport_size({"width": 1024, "height": 720})
    response = page.goto(f"{base_url}/")

    assert response is not None and response.ok
    expect(page.locator("#startup-title")).to_be_visible()
    startup_progress = page.get_by_role("progressbar", name="Initializing XREPORT Inference")
    expect(startup_progress).to_be_visible()
    expect(
        page.get_by_role("heading", name="Turn a radiograph into a draft report")
    ).to_have_count(0)

    qa_dir = _e2e_screenshot_dir("validation_campaign/tier-0/s02-20260924")
    page.clock.run_for(15_000)
    assert int(startup_progress.get_attribute("aria-valuenow") or "0") > 0
    slow_health_calls = health_calls
    page.screenshot(path=str(qa_dir / "startup-slow.png"), full_page=False)

    page.clock.run_for(45_000)
    retry = page.get_by_role("button", name="Retry connection")
    expect(retry).to_be_visible(timeout=5_000)
    unavailable_health_calls = health_calls
    page.screenshot(path=str(qa_dir / "startup-unavailable.png"), full_page=False)
    health_available = True
    retry.click()
    page.clock.run_for(1_000)

    expect(
        page.get_by_role("heading", name="Turn a radiograph into a draft report")
    ).to_be_visible(timeout=10_000)
    expect(page.locator("app-startup-screen")).to_have_count(0)
    expect(page.locator(".catalog-state.error")).to_contain_text(
        "Synthetic model catalogue outage"
    )
    assert unavailable_health_calls > slow_health_calls
    assert health_calls > unavailable_health_calls
    page.screenshot(
        path=str(qa_dir / "startup-feature-error-after-ready.png"), full_page=False
    )

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
    expect(page.get_by_role("tab", name="Data access")).to_have_count(0)
    expect(page.get_by_role("tab", name="Advanced")).to_be_visible()

    page.get_by_role("tab", name="Advanced").click()
    expect(page.get_by_role("heading", name="Advanced")).to_be_visible()
    polling = page.get_by_role("spinbutton", name="Job status polling interval seconds")
    timeout = page.get_by_role("spinbutton", name="Inference timeout seconds")
    expect(polling).to_be_visible()
    expect(timeout).to_be_visible()

    page.get_by_role("tab", name="General").click()
    seed = page.get_by_role("spinbutton", name="Default random seed")
    try:
        seed.fill("123")
        page.reload()
        expect(seed).to_have_value("123")

        page.get_by_role("tab", name="Advanced").click()
        polling.fill("2.5")
        with page.expect_response(
            lambda item: item.url.endswith("/api/settings") and item.request.method == "PATCH"
        ) as polling_save:
            timeout.fill("901")
        assert polling_save.value.ok
        with page.expect_response(
            lambda item: item.url.endswith("/api/settings") and item.request.method == "PATCH"
        ) as timeout_save:
            timeout.blur()
        assert timeout_save.value.ok

        page.reload()
        expect(seed).to_have_value("123")
        page.get_by_role("tab", name="Advanced").click()
        expect(polling).to_have_value("2.5")
        expect(timeout).to_have_value("901")
    finally:
        page.goto(f"{base_url}/settings")
        page.get_by_role("button", name="Reset to defaults").click()
        expect(page.get_by_text("Settings reset to defaults.", exact=True)).to_be_visible()

###############################################################################
def test_database_settings_are_drafts_until_confirmation_and_failed_probe_keeps_active_backend(
    page: Page,
    base_url: str,
) -> None:
    response = page.goto(f"{base_url}/settings")
    assert response is not None and response.ok
    page.get_by_role("tab", name="Database").click()
    expect(page.get_by_role("heading", name="Database")).to_be_visible()

    database_status = page.locator(".database-status")
    expect(database_status).to_contain_text("Connected to SQLite")
    active_path = page.get_by_label("Database file path").input_value()
    page.get_by_label("Database file path").fill(active_path + ".candidate")
    expect(database_status).to_contain_text(active_path)
    expect(page.get_by_role("button", name="Apply database settings")).to_be_disabled()

    page.reload()
    page.get_by_role("tab", name="Database").click()
    expect(page.get_by_label("Database file path")).to_have_value(active_path)

    page.get_by_label("Database backend").select_option("postgresql")
    page.locator("#database-host").fill("127.0.0.1")
    page.locator("#database-port").fill("1")
    page.locator("#database-name").fill("xreport_unavailable")
    page.locator("#database-username").fill("xreport")
    page.locator("#database-password").fill("candidate-password")
    page.locator("#database-timeout").fill("1")
    page.locator("#database-batch-size").fill("128")
    page.get_by_role("checkbox", name="Require SSL").check()
    page.locator("#database-ssl-ca").fill("C:\\candidate\\ca.pem")
    expect(page.locator("#database-ssl-ca")).to_have_value("C:\\candidate\\ca.pem")
    expect(database_status).to_contain_text(active_path)
    with page.expect_response(
        lambda item: item.url.endswith("/api/settings/database/test")
        and item.request.method == "POST"
    ) as test_response:
        page.get_by_role("button", name="Test connection").click()
    assert test_response.value.ok
    expect(page.locator(".database-panel .settings-feedback-error")).to_be_visible()
    expect(page.get_by_role("button", name="Apply database settings")).to_be_disabled()
    expect(database_status).to_contain_text("Connected to SQLite")

    page.reload()
    page.get_by_role("tab", name="Database").click()
    expect(page.get_by_label("Database backend")).to_have_value("sqlite")
    expect(page.get_by_label("Database file path")).to_have_value(active_path)
    expect(page.locator(".database-status")).to_contain_text("Connected to SQLite")

###############################################################################
def test_settings_autosave_failure_reverts_value_and_shows_error(
    page: Page,
    base_url: str,
) -> None:
    response = page.goto(f"{base_url}/settings")
    assert response is not None and response.ok
    seed = page.get_by_role("spinbutton", name="Default random seed")
    previous_seed = seed.input_value()

    def reject_settings_write(route: Route) -> None:
        if route.request.method == "PATCH":
            route.fulfill(
                status=503,
                content_type="application/json",
                body='{"detail":"simulated settings write failure"}',
            )
            return
        route.continue_()

    page.route("**/api/settings", reject_settings_write)
    seed.fill("123" if previous_seed != "123" else "124")
    feedback = page.locator(".settings-feedback-error[role='alert']")
    expect(feedback).to_contain_text("simulated settings write failure")
    expect(feedback).to_contain_text("failed changes were reverted")
    expect(seed).to_have_value(previous_seed)

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
          return {
            catalogHeight: catalog?.getBoundingClientRect().height ?? 0,
            groupsHeight: groups?.getBoundingClientRect().height ?? 0,
            overflowY: groups ? getComputedStyle(groups).overflowY : '',
          };
        }
        """,
    )
    assert inference_metrics["catalogHeight"] > 0
    assert inference_metrics["groupsHeight"] > 0
    assert inference_metrics["overflowY"] in {"auto", "scroll"}

    page.get_by_role("button", name="Dismiss model information").click()
    expect(banner).to_have_count(0)
    cards = page.locator(".model-card")
    if cards.count() > 1:
        cards.nth(1).click()
        expect(page.locator(".model-info-banner")).to_be_visible()

    page.set_viewport_size({"width": 640, "height": 800})

    page.goto(f"{base_url}/settings")
    expect(page.get_by_role("tab", name="General")).to_be_visible()
    page.screenshot(path=str(qa_dir / "xreport-settings-narrow-dark.png"), full_page=False)
    for tab_name, heading_name in (("General", "General"), ("Advanced", "Advanced")):
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

    page.get_by_role("tab", name="Database").click()
    expect(page.get_by_role("heading", name="Database", level=2)).to_be_visible()
    assert page.locator(".database-field").count() > 0

    with page.expect_response(lambda item: item.url.endswith("/api/settings") and item.request.method == "PATCH"):
        page.get_by_role("button", name="Use Light theme").click()
    expect(page.locator("html")).to_have_attribute("data-theme", "light")
    page.screenshot(path=str(qa_dir / "xreport-settings-narrow-light.png"), full_page=False)
    with page.expect_response(lambda item: item.url.endswith("/api/settings") and item.request.method == "PATCH"):
        page.get_by_role("button", name="Use Dark theme").click()
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    with page.expect_response(lambda item: item.url.endswith("/api/settings") and item.request.method == "PATCH"):
        page.get_by_role("button", name="Use System theme").click()
    expect(page.get_by_text("Theme saved", exact=True)).to_have_count(0)

    page.set_viewport_size({"width": 390, "height": 844})
    response = page.goto(f"{base_url}/inference")
    assert response is not None and response.ok
    expect(
        page.get_by_role("heading", name="Turn a radiograph into a draft report")
    ).to_be_visible()
    page.screenshot(
        path=str(qa_dir / "xreport-inference-mobile-dark.png"),
        full_page=False,
    )
    mobile_metrics = page.evaluate(
        """
        () => {
          const content = document.querySelector('.main-layout-content');
          const chrome = document.querySelector('.main-layout-chrome');
          return {
            viewportWidth: window.innerWidth,
            documentWidth: document.documentElement.scrollWidth,
            contentClientHeight: content?.clientHeight ?? 0,
            contentScrollHeight: content?.scrollHeight ?? 0,
            contentOverflowY: content ? getComputedStyle(content).overflowY : '',
            chromeWidth: chrome?.getBoundingClientRect().width ?? 0,
            chromeTop: chrome?.getBoundingClientRect().top ?? -1,
          };
        }
        """,
    )
    assert mobile_metrics["viewportWidth"] == 390
    assert mobile_metrics["documentWidth"] <= mobile_metrics["viewportWidth"]
    assert mobile_metrics["contentScrollHeight"] > mobile_metrics["contentClientHeight"]
    assert mobile_metrics["contentOverflowY"] in {"auto", "scroll"}
    assert mobile_metrics["chromeWidth"] == mobile_metrics["viewportWidth"]
    page.mouse.move(385, 500)
    page.mouse.wheel(0, 700)
    page.wait_for_function(
        "() => document.querySelector('.main-layout-content').scrollTop > 0"
    )
    post_scroll_shell = page.evaluate(
        """
        () => {
          const chrome = document.querySelector('.main-layout-chrome');
          return {
            scrollTop: document.querySelector('.main-layout-content').scrollTop,
            chromeWidth: chrome.getBoundingClientRect().width,
            chromeTop: chrome.getBoundingClientRect().top,
          };
        }
        """,
    )
    assert post_scroll_shell["scrollTop"] > 0
    assert post_scroll_shell["chromeWidth"] == mobile_metrics["chromeWidth"]
    assert post_scroll_shell["chromeTop"] == mobile_metrics["chromeTop"]

    assert not console_errors, console_errors
    assert not request_failures, request_failures

###############################################################################
def test_s52_responsive_route_modal_and_reduced_motion_matrix(
    page: Page,
    base_url: str,
) -> None:
    """Exercise routed surfaces, modal contracts, and reduced-motion behavior."""
    qa_dir = _e2e_screenshot_dir("")
    console_errors: list[str] = []
    page_errors: list[str] = []
    request_failures: list[str] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text)
        if message.type == "error"
        else None,
    )
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.on(
        "requestfailed",
        lambda request: request_failures.append(
            f"{request.method} {request.url}: {request.failure}"
        ),
    )
    page.emulate_media(reduced_motion="reduce")

    viewports = [
        (320, 640),
        (390, 844),
        (640, 800),
        (1024, 720),
        (1440, 900),
    ]
    routes = [
        ("inference", "/inference", "Turn a radiograph into a draft report"),
        ("reports", "/reports", "Reports"),
        ("settings", "/settings", "Settings"),
    ]

    for width, height in viewports:
        page.set_viewport_size({"width": width, "height": height})
        for route_name, route, heading in routes:
            response = page.goto(f"{base_url}{route}")
            assert response is not None and response.ok, route
            expect(page.get_by_role("heading", name=heading)).to_be_visible()
            expect(page.locator(".main-layout-content")).to_be_visible()
            _assert_s52_semantic_route_metadata(page, route_name)

            metrics = page.evaluate(
                """() => {
                  const content = document.querySelector('.main-layout-content');
                  const style = content ? getComputedStyle(content) : null;
                  const documentWidth = document.documentElement.scrollWidth;
                  const viewportWidth = window.innerWidth;
                  if (content) {
                    const maxScroll = Math.max(0, content.scrollHeight - content.clientHeight);
                    content.scrollTop = Math.min(maxScroll, 180);
                  }
                  return {
                    documentWidth,
                    viewportWidth,
                    contentClientHeight: content?.clientHeight ?? 0,
                    contentScrollHeight: content?.scrollHeight ?? 0,
                    contentScrollTop: content?.scrollTop ?? 0,
                    contentOverflowY: style?.overflowY ?? '',
                  };
                }""",
            )
            assert metrics["documentWidth"] <= metrics["viewportWidth"]
            assert metrics["contentOverflowY"] in {"auto", "scroll"}, (
                f"missing scroll container for {route_name}@{width}x{height}: {metrics}"
            )
            if metrics["contentScrollHeight"] > metrics["contentClientHeight"]:
                assert metrics["contentScrollTop"] > 0

            page.screenshot(
                path=str(qa_dir / f"s52-{route_name}-{width}x{height}.png"),
                full_page=False,
            )

        page.goto(f"{base_url}/inference")
        page.evaluate(
            """() => {
              window.__xreportScrollCalls = [];
              const recordScroll = (owner, key, name) => {
                const original = owner[key];
                if (typeof original !== 'function') return;
                owner[key] = function(...args) {
                  const options = args[0];
                  window.__xreportScrollCalls.push({
                    name,
                    behavior: options && typeof options === 'object' ? options.behavior ?? null : null,
                  });
                  return original.apply(this, args);
                };
              };
              recordScroll(Element.prototype, 'scrollIntoView', 'scrollIntoView');
              recordScroll(Element.prototype, 'scrollBy', 'scrollBy');
            }""",
        )
        help_button = page.get_by_role("button", name="Help and tips")
        expect(help_button).to_be_visible()
        help_button.click()
        tips_dialog = page.locator(".guidance-modal")
        expect(tips_dialog).to_be_visible()
        tips_box = tips_dialog.bounding_box()
        assert tips_box is not None
        assert tips_box["x"] >= 0
        assert tips_box["y"] >= 0
        assert tips_box["x"] + tips_box["width"] <= width
        assert tips_box["y"] + tips_box["height"] <= height

        for _ in range(10):
            assert page.evaluate(
                """() => document.querySelector('.guidance-modal')?.contains(document.activeElement) ?? false"""
            )
            page.keyboard.press("Tab")
        assert page.evaluate(
            """() => document.querySelector('.guidance-modal')?.contains(document.activeElement) ?? false"""
        )
        page.keyboard.press("Escape")
        expect(tips_dialog).to_have_count(0)
        expect(help_button).to_be_focused()

        help_button.click()
        expect(tips_dialog).to_be_visible()
        page.get_by_role("button", name="Show walkthrough").click()
        tour_dialog = page.locator(".guided-tour-dialog")
        expect(tour_dialog).to_be_visible()
        tour_box = tour_dialog.bounding_box()
        assert tour_box is not None
        assert tour_box["x"] >= 0
        assert tour_box["y"] >= 0
        assert tour_box["x"] + tour_box["width"] <= width
        assert tour_box["y"] + tour_box["height"] <= height

        for _ in range(10):
            assert page.evaluate(
                """() => document.querySelector('.guided-tour-dialog')?.contains(document.activeElement) ?? false"""
            )
            page.keyboard.press("Tab")
        assert page.evaluate(
            """() => document.querySelector('.guided-tour-dialog')?.contains(document.activeElement) ?? false"""
        )

        page.get_by_role("button", name="Next").click()
        page.get_by_role("button", name="Next").click()
        page.get_by_role("button", name="Next").click()
        page.wait_for_timeout(180)
        scroll_calls = page.evaluate("() => window.__xreportScrollCalls")
        assert scroll_calls
        assert all(
            call["behavior"] in {None, "auto"}
            for call in scroll_calls
        ), scroll_calls

        reduced_motion = page.evaluate(
            """() => {
              const dot = document.createElement('span');
              dot.className = 'guidance-journey-status-dot is-loading';
              const chevron = document.createElement('span');
              chevron.className = 'guidance-journey-chevron';
              document.body.append(dot, chevron);
              const dotStyle = getComputedStyle(dot);
              const chevronStyle = getComputedStyle(chevron);
              const result = {
                matches: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
                animationName: dotStyle.animationName,
                animationDurationSeconds: parseFloat(dotStyle.animationDuration) * (dotStyle.animationDuration.endsWith('ms') ? 0.001 : 1),
                transitionProperty: chevronStyle.transitionProperty,
                transitionDurationSeconds: parseFloat(chevronStyle.transitionDuration) * (chevronStyle.transitionDuration.endsWith('ms') ? 0.001 : 1),
              };
              dot.remove();
              chevron.remove();
              return result;
            }""",
        )
        assert reduced_motion["matches"] is True
        assert reduced_motion["animationName"] in {"none", ""}
        assert reduced_motion["animationDurationSeconds"] <= 0.001
        assert reduced_motion["transitionProperty"] in {"none", "all"}
        assert reduced_motion["transitionDurationSeconds"] <= 0.001

        page.keyboard.press("Escape")
        expect(tour_dialog).to_have_count(0)

    assert not console_errors, console_errors
    assert not page_errors, page_errors
    assert not request_failures, request_failures

###############################################################################
def test_s52_semantic_metadata_and_keyboard_only_smoke(
    page: Page,
    base_url: str,
) -> None:
    """Check representative semantic contracts and keyboard-only interactions."""
    page.set_viewport_size({"width": 1024, "height": 720})
    response = page.goto(f"{base_url}/inference")

    assert response is not None and response.ok
    expect(page.get_by_role("heading", name="Turn a radiograph into a draft report")).to_be_visible()
    _assert_s52_semantic_route_metadata(page, "inference")

    help_button = page.get_by_role("button", name="Help and tips")
    _focus_s52_with_tab(page, help_button)
    page.keyboard.press("Enter")

    tips_dialog = page.get_by_role("dialog", name="Tips & Tricks")
    expect(tips_dialog).to_be_visible()
    expect(help_button).to_have_attribute("aria-expanded", "true")
    _assert_s52_dialog_semantics(page, tips_dialog, require_description=False)
    assert page.evaluate(
        "() => document.querySelector('.guidance-modal')?.contains(document.activeElement) ?? false"
    )

    walkthrough_button = page.get_by_role("button", name="Show walkthrough")
    _focus_s52_with_tab(page, walkthrough_button)
    page.keyboard.press("Enter")

    tour_dialog = page.locator(".guided-tour-dialog")
    expect(tour_dialog).to_be_visible()
    _assert_s52_dialog_semantics(page, tour_dialog, require_description=True)
    first_step = tour_dialog.get_by_role("heading").inner_text()
    next_button = page.get_by_role("button", name="Next")
    _focus_s52_with_tab(page, next_button)
    page.keyboard.press("Enter")
    expect(tour_dialog.get_by_role("heading")).not_to_have_text(first_step)
    page.keyboard.press("Escape")
    expect(tour_dialog).to_have_count(0)
    expect(help_button).to_be_focused()

    settings_link = page.get_by_role("link", name="Settings", exact=True)
    _focus_s52_with_tab(page, settings_link)
    page.keyboard.press("Enter")
    expect(page).to_have_url(f"{base_url}/settings")
    _assert_s52_semantic_route_metadata(page, "settings")

    general_tab = page.get_by_role("tab", name="General")
    advanced_tab = page.get_by_role("tab", name="Advanced")
    _focus_s52_with_tab(page, general_tab)
    page.keyboard.press("ArrowRight")
    expect(advanced_tab).to_be_focused()
    page.keyboard.press("Enter")
    expect(advanced_tab).to_have_attribute("aria-selected", "true")
    advanced_panel = page.get_by_role("tabpanel")
    advanced_tab_id = advanced_tab.get_attribute("id")
    assert advanced_tab_id
    expect(advanced_panel).to_have_attribute("aria-labelledby", advanced_tab_id)
    expect(advanced_panel.get_by_role("heading", name="Advanced")).to_be_visible()

###############################################################################
def _assert_s52_semantic_route_metadata(page: Page, route_name: str) -> None:
    navigation = page.get_by_role("navigation", name="Primary navigation")
    expect(navigation).to_have_count(1)
    expect(page.get_by_role("main")).to_have_count(1)
    assert page.locator("main").evaluate("element => element.tagName") == "MAIN"

    for label in ("Inference", "Reports", "Settings"):
        expect(navigation.get_by_role("link", name=label, exact=True)).to_be_visible()

    broken_references = page.evaluate(
        """
        () => [...document.querySelectorAll('[aria-labelledby], [aria-describedby]')]
          .flatMap((element) => ['aria-labelledby', 'aria-describedby'].flatMap((attribute) =>
            (element.getAttribute(attribute) ?? '')
              .split(/\\s+/)
              .filter(Boolean)
              .filter((id) => !document.getElementById(id))
              .map((id) => `${element.tagName.toLowerCase()}[${attribute}=${id}]`)
          ))
        """,
    )
    assert not broken_references, f"broken ARIA references on {route_name}: {broken_references}"

    unnamed_controls = page.evaluate(
        """
        () => [...document.querySelectorAll('a, button')]
          .filter((element) => {
            const text = (element.textContent ?? '').replace(/\\s+/g, ' ').trim();
            return !text && !element.getAttribute('aria-label') && !element.getAttribute('aria-labelledby');
          })
          .map((element) => element.outerHTML.slice(0, 180))
        """,
    )
    assert not unnamed_controls, f"unnamed links or buttons on {route_name}: {unnamed_controls}"

    if route_name == "inference":
        expect(page.locator('section[aria-label="Inference workflow"]')).to_have_count(1)
        expect(page.locator('aside[aria-label="Model catalog"]')).to_have_count(1)
        model_region = page.get_by_role("region", name="Scrollable model catalogue")
        expect(model_region).to_have_attribute("tabindex", "0")
        expect(page.get_by_role("textbox", name="Filter models")).to_be_visible()
    elif route_name == "reports":
        expect(page.locator('main[aria-labelledby="reports-title"]')).to_have_count(1)
        expect(page.get_by_role("region", name="Report history filters")).to_have_count(1)
        expect(page.get_by_role("textbox", name="Model reference")).to_be_visible()
        expect(page.get_by_role("combobox", name="Status")).to_be_visible()
        expect(page.get_by_role("combobox", name="Sort")).to_be_visible()
    elif route_name == "settings":
        tabs = page.get_by_role("tab")
        expect(tabs).to_have_count(5)
        tab_metadata = page.evaluate(
            """
            () => [...document.querySelectorAll('[role="tab"]')].map((tab) => ({
              id: tab.id,
              selected: tab.getAttribute('aria-selected'),
              controls: tab.getAttribute('aria-controls'),
              tabindex: tab.getAttribute('tabindex'),
            }))
            """,
        )
        assert all(
            item["selected"] in {"true", "false"}
            and item["tabindex"] in {"0", "-1"}
            and item["controls"]
            for item in tab_metadata
        ), tab_metadata
        active_tab = next(
            item for item in tab_metadata if item["selected"] == "true"
        )
        assert page.locator(f'#{active_tab["controls"]}').count() == 1
        panel = page.get_by_role("tabpanel")
        expect(panel).to_have_count(1)
        labelled_by = panel.get_attribute("aria-labelledby")
        assert labelled_by and page.locator(f'#{labelled_by}').count() == 1
        expect(page.get_by_role("spinbutton", name=re.compile("random seed|polling interval|inference timeout", re.I)).first).to_be_visible()
    else:
        raise AssertionError(f"Unexpected S52 route: {route_name}")

###############################################################################
def _assert_s52_dialog_semantics(page: Page, dialog, *, require_description: bool) -> None:
    expect(dialog).to_have_attribute("role", "dialog")
    expect(dialog).to_have_attribute("aria-modal", "true")
    labelled_by = dialog.get_attribute("aria-labelledby")
    assert labelled_by
    assert page.evaluate("id => Boolean(document.getElementById(id))", labelled_by)
    described_by = dialog.get_attribute("aria-describedby")
    if require_description:
        assert described_by
        assert page.evaluate("id => Boolean(document.getElementById(id))", described_by)
    elif described_by:
        assert page.evaluate("id => Boolean(document.getElementById(id))", described_by)

###############################################################################
def _focus_s52_with_tab(page: Page, target, max_tabs: int = 64) -> None:
    for _ in range(max_tabs):
        if target.evaluate("element => element === document.activeElement"):
            return
        page.keyboard.press("Tab")
    raise AssertionError(f"Keyboard Tab navigation did not reach {target}")

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

###############################################################################
def _e2e_screenshot_dir(default_relative: str) -> Path:
    configured = os.environ.get("XREPORT_E2E_SCREENSHOT_DIR", "").strip()
    qa_dir = Path(configured) if configured else Path(__file__).parents[3] / "assets" / "QA" / default_relative
    qa_dir.mkdir(parents=True, exist_ok=True)
    return qa_dir

###############################################################################
def _s10_qa_dir() -> Path:
    configured = os.environ.get("XREPORT_S10_SCREENSHOT_DIR", "").strip()
    if configured:
        qa_dir = Path(configured)
        qa_dir.mkdir(parents=True, exist_ok=True)
        return qa_dir
    return _e2e_screenshot_dir("validation_campaign/s10")

###############################################################################
def _expect_s10_active_navigation(page: Page, label: str) -> None:
    active_links = page.locator(
        'nav[aria-label="Primary navigation"] a.app-nav-button.active'
    )
    expect(active_links).to_have_count(1)
    expect(active_links).to_have_attribute("aria-label", label)

###############################################################################
def _expect_s10_route_surface(page: Page, path: str) -> None:
    if path == "/inference":
        expect(
            page.get_by_role("heading", name="Turn a radiograph into a draft report")
        ).to_be_visible()
    elif path == "/reports":
        expect(page.get_by_role("heading", name="Reports")).to_be_visible()
    elif path.startswith("/reports/"):
        expect(page.get_by_role("textbox", name="Findings").first).to_be_visible()
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

    response = page.goto(f"{base_url}/")
    assert response is not None and response.ok
    expect(page).to_have_url(f"{base_url}/inference")
    _expect_s10_route_surface(page, "/inference")
    _expect_s10_active_navigation(page, "Inference")

    routes = [
        ("/inference", "Inference"),
        ("/reports", "Reports"),
        (f"/reports/{seeded_history}", "Reports"),
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

    for label, path in (
        ("Inference", "/inference"),
        ("Reports", "/reports"),
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

    with page.expect_response(lambda item: item.url.endswith("/api/settings") and item.request.method == "PATCH"):
        light.click()
    expect(light).to_have_attribute("aria-pressed", "true")
    expect(page.locator("html")).to_have_attribute("data-theme", "light")
    expect(page.get_by_text("Theme saved", exact=True)).to_have_count(0)
    assert page.evaluate("() => localStorage.getItem('theme-preference')") is None
    page.screenshot(path=str(qa_dir / "theme-light-inference.png"), full_page=False)
    response = page.reload()
    assert response is not None and response.ok
    expect(light).to_have_attribute("aria-pressed", "true")
    expect(page.locator("html")).to_have_attribute("data-theme", "light")
    page.emulate_media(color_scheme="dark")
    expect(page.locator("html")).to_have_attribute("data-theme", "light")

    with page.expect_response(lambda item: item.url.endswith("/api/settings") and item.request.method == "PATCH"):
        dark.click()
    expect(dark).to_have_attribute("aria-pressed", "true")
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    expect(page.get_by_text("Theme saved", exact=True)).to_have_count(0)
    assert page.evaluate("() => localStorage.getItem('theme-preference')") is None
    response = page.reload()
    assert response is not None and response.ok
    expect(dark).to_have_attribute("aria-pressed", "true")
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    page.emulate_media(color_scheme="light")
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")

    with page.expect_response(lambda item: item.url.endswith("/api/settings") and item.request.method == "PATCH"):
        system.click()
    expect(system).to_have_attribute("aria-pressed", "true")
    expect(page.locator("html")).to_have_attribute("data-theme", "light")
    assert page.evaluate("() => localStorage.getItem('theme-preference')") is None
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
    assert page.evaluate("() => localStorage.getItem('theme-preference')") is None
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
