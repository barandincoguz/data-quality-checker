"""The expert screen in a real browser: what each control writes to the store.

The page's decisions go through JavaScript (shortcuts, the revision table,
the one-submission guard), which a Flask test client never runs. Skipped where
Playwright or a Chromium build is not available.
"""

from __future__ import annotations

import glob
import json
import os
import threading
import time

import pytest
from test_hitl import TOKEN, fixture
from werkzeug.serving import make_server

from data_quality_checker.storage import Store

playwright = pytest.importorskip("playwright.sync_api")
_CHROMIUM = sorted(
    glob.glob(
        os.path.expanduser(
            "~/Library/Caches/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell"
        )
    )
)
pytestmark = pytest.mark.skipif(not _CHROMIUM, reason="no Chromium build for Playwright")


@pytest.fixture
def screen(tmp_path):
    config, app = fixture(tmp_path, count=8)
    server = make_server("127.0.0.1", 0, app, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=_CHROMIUM[-1])
        page = browser.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(base + "/login")
        page.fill("#access-token", TOKEN)
        page.click("button[type=submit]")
        page.wait_for_url("**/review/**")
        yield config, page, base
        browser.close()
        assert not errors
    server.shutdown()


def _review(config, doc_id):
    with Store(config.database_path) as store:
        return store.get_review("batch", doc_id)


def _doc_id(page) -> str:
    return page.url.rsplit("/", 1)[-1]


def _next(page, previous: str) -> None:
    page.wait_for_function(f"!location.pathname.endsWith('{previous}')")


def test_every_control_writes_what_it_says(screen) -> None:
    config, page, base = screen

    # Mouse: the annotator's list is the truth.
    first = _doc_id(page)
    page.click("button[value=accept_human]")
    _next(page, first)
    assert _review(config, first)["action"] == "accept_human"

    # Shortcuts never fire with a modifier or while the revision panel is open.
    second = _doc_id(page)
    page.keyboard.press("Meta+m")
    page.click("#revision-panel > summary")
    page.click("#add-row")
    page.keyboard.press("m")
    time.sleep(0.3)
    assert _review(config, second)["status"] == "pending"
    page.click("#revision-panel > summary")
    page.locator("body").click(position={"x": 5, "y": 5})
    page.keyboard.press("m")
    _next(page, second)
    assert _review(config, second)["action"] == "accept_model"

    # The revision table is saved as edited.
    third = _doc_id(page)
    page.click("#revision-panel > summary")
    page.click("#fill-a")
    page.locator("#ref-rows tr").nth(0).locator("input[data-field=madde]").fill("95")
    page.click("button[value=revise]")
    _next(page, third)
    final = json.loads(_review(config, third)["final_references_json"])
    assert _review(config, third)["action"] == "revise"
    assert "95" in {row["madde"] for row in final}

    # JSON typed into the technical field wins over the table.
    fourth = _doc_id(page)
    page.click("#revision-panel > summary")
    page.click("#revision-panel details.technical > summary")
    typed = [
        {
            "kanun_no": "7394",
            "kanun_ad": "",
            "madde": "3",
            "fikra": "",
            "bent": "",
            "source_text": "",
        }
    ]
    page.fill("textarea[name=references_json]", json.dumps(typed))
    page.click("button[value=revise]")
    _next(page, fourth)
    assert [
        row["kanun_no"] for row in json.loads(_review(config, fourth)["final_references_json"])
    ] == ["7394"]

    # A deferral needs a reason; the page says so instead of submitting.
    fifth = _doc_id(page)
    page.click("details.secondary:not(#revision-panel) > summary")
    page.click("button[value=defer]")
    assert page.locator("#defer-msg").is_visible()
    assert _review(config, fifth)["status"] == "pending"
    page.fill("input[name=reason]", "ikinci görüş")
    page.click("button[value=defer]")
    _next(page, fifth)
    assert _review(config, fifth)["status"] == "deferred"
    assert _review(config, fifth)["reason"] == "ikinci görüş"

    # A double click is one decision.
    sixth = _doc_id(page)
    page.dblclick("button[value=accept_human]")
    _next(page, sixth)
    assert _review(config, sixth)["row_version"] == 1

    # Every decision carried the time its page was open.
    with Store(config.database_path) as store:
        events = [
            json.loads(payload)
            for (payload,) in store.connection.execute(
                "SELECT payload_json FROM batch_events WHERE event_type='expert_review_updated'"
            )
        ]
    assert len(events) == 6
    assert all(isinstance(event["seconds_on_page"], float) for event in events)


def test_a_row_and_its_places_in_the_text_point_at_each_other(screen) -> None:
    """6 Oct 2026: a click on a comparison row moves to where its reference is
    cited; a click on a highlight marks the rows it cites; Esc clears both."""
    _, page, _ = screen
    row = page.locator("table.cmp tbody tr[data-row]").first
    index = row.get_attribute("data-row")
    row.click()
    current = page.locator("#doctext mark.current")
    assert current.count() == 1
    assert index in current.get_attribute("data-rows").split()
    assert "active" in row.get_attribute("class")

    page.keyboard.press("Escape")
    assert page.locator("#doctext mark.current, #doctext mark.active").count() == 0
    assert page.locator("table.cmp tr.active").count() == 0

    mark = page.locator("#doctext mark.ev").first
    mark.click()
    rows = mark.get_attribute("data-rows").split()
    for cited in rows:
        assert "active" in page.locator(f'table.cmp tr[data-row="{cited}"]').get_attribute("class")
