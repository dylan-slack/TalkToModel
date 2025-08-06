from playwright.sync_api import Page, expect

def test_app_loads(page: Page):
    """
    This test verifies that the application loads successfully.
    """
    # 1. Arrange: Go to the application's URL.
    page.goto("http://127.0.0.1:4455")

    # 2. Assert: Confirm the page has loaded by checking for a known element.
    # I'll check for the main heading.
    heading = page.locator("div.ttm-header-title")
    expect(heading).to_have_text("TalkToModel")

    # 3. Screenshot: Capture the final result for visual verification.
    page.screenshot(path="jules-scratch/verification/verification.png")
