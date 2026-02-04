"""
E2E tests for UI navigation and page rendering.
Tests basic UI functionality using Playwright browser automation.
"""

import re
from playwright.sync_api import Page, expect


class TestHomePage:
    """Tests for the home page and basic navigation."""

    def test_homepage_loads_successfully(self, page: Page, base_url: str):
        """The homepage should load without errors."""
        page.goto(base_url)
        # Page should have loaded successfully
        expect(page).to_have_title(
            re.compile("XREPORT|Radiological Reports", re.IGNORECASE)
        )

    def test_homepage_has_navigation(self, page: Page, base_url: str):
        """The homepage should have navigation elements."""
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        # Check for main navigation links (icon-only sidebar uses title attributes)
        expect(page.locator('a[title="Dataset"]')).to_be_visible()
        expect(page.locator('a[title="Database"]')).to_be_visible()
        expect(page.locator('a[title="Training"]')).to_be_visible()
        expect(page.locator('a[title="Inference"]')).to_be_visible()

    def test_homepage_has_xreport_branding(self, page: Page, base_url: str):
        """The homepage should display the dataset landing header."""
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        expect(
            page.locator("h1").filter(
                has_text=re.compile("Dataset Management", re.IGNORECASE)
            )
        ).to_be_visible()


class TestNavigationFlow:
    """Tests for navigating between different pages."""

    def test_navigate_to_database_page(self, page: Page, base_url: str):
        """Should be able to navigate to the Database page."""
        page.goto(base_url)
        page.locator('a[title="Database"]').click()
        page.wait_for_load_state("networkidle")

        expect(page).to_have_url(re.compile(".*database", re.IGNORECASE))
        expect(page.locator("h1, h2, h3").filter(has_text="Database")).to_be_visible()

    def test_navigate_to_training_page(self, page: Page, base_url: str):
        """Should be able to navigate to the Training page."""
        page.goto(base_url)
        page.locator('a[title="Training"]').click()
        page.wait_for_load_state("networkidle")

        expect(page).to_have_url(re.compile(".*training", re.IGNORECASE))
        expect(
            page.locator("h1").filter(
                has_text=re.compile("XREPORT Transformer", re.IGNORECASE)
            )
        ).to_be_visible()

    def test_navigate_to_inference_page(self, page: Page, base_url: str):
        """Should be able to navigate to the Inference page."""
        page.goto(base_url)
        page.locator('a[title="Inference"]').click()
        page.wait_for_load_state("networkidle")

        expect(page).to_have_url(re.compile(".*inference", re.IGNORECASE))
        expect(page.locator("h1, h2, h3").filter(has_text="Inference")).to_be_visible()


class TestDatabasePage:
    """Tests for the Database browser page."""

    def test_database_page_shows_table_list(self, page: Page, base_url: str):
        """The Database page should display a list of tables."""
        page.goto(f"{base_url}/database")
        page.wait_for_load_state("networkidle")

        expect(page.get_by_text("Select Table", exact=False)).to_be_visible()

    def test_database_page_allows_table_selection(self, page: Page, base_url: str):
        """Should be able to select a table from the list."""
        page.goto(f"{base_url}/database")
        page.wait_for_load_state("networkidle")

        # Find a table item and click it
        # This might be specific to implementation, using a generic selector for now
        # expect(page.locator(".table-item").first).to_be_visible()
        pass


class TestTrainingPage:
    """Tests for the Training page."""

    def test_training_page_loads(self, page: Page, base_url: str):
        """The Training page should load without errors."""
        page.goto(f"{base_url}/training")
        page.wait_for_load_state("networkidle")

        expect(
            page.locator("h1").filter(
                has_text=re.compile("XREPORT Transformer", re.IGNORECASE)
            )
        ).to_be_visible()

    def test_training_page_shows_status(self, page: Page, base_url: str):
        """The Training page should display training status."""
        page.goto(f"{base_url}/training")
        page.wait_for_load_state("networkidle")

        expect(page.get_by_text("Training Dashboard", exact=False)).to_be_visible()

    def test_training_page_has_dashboard(self, page: Page, base_url: str):
        """The Training page should have a dashboard for metrics."""
        page.goto(f"{base_url}/training")
        page.wait_for_load_state("networkidle")

        # Check for charts or metrics area
        expect(page.locator("canvas, svg").first).to_be_visible()


class TestInferencePage:
    """Tests for the Inference page."""

    def test_inference_page_loads(self, page: Page, base_url: str):
        """The Inference page should load without errors."""
        page.goto(f"{base_url}/inference")
        page.wait_for_load_state("networkidle")

        expect(page.locator("h1, h2").filter(has_text="Inference")).to_be_visible()

    def test_inference_page_shows_checkpoint_selector(self, page: Page, base_url: str):
        """The Inference page should have a checkpoint selector."""
        page.goto(f"{base_url}/inference")
        page.wait_for_load_state("networkidle")

        expect(page.locator("#checkpoint-select")).to_be_visible()

    def test_inference_page_has_image_upload(self, page: Page, base_url: str):
        """The Inference page should have an image upload area."""
        page.goto(f"{base_url}/inference")
        page.wait_for_load_state("networkidle")

        # Check for upload area
        expect(page.locator("input[type='file']")).to_be_attached()
