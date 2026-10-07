"""The expert screen's helpers, in a real browser, on the synthetic batch.

Documents (`review_fixture.DOCUMENTS`, in this order): d000001 a second mention, a
repeated row and a reference only the model has (3065/13); d000002 a tebliğ the
model turned into 3065/2 (cited nowhere) and a law-only row beside its article;
d000003 GREEN; d000004 mükerrer/geçici/range; d000005 a quote not verbatim;
d000006 empty; d000007 a long ruling.
"""

from __future__ import annotations

import glob
import json
import os
import threading

import pytest
from review_fixture import build
from werkzeug.serving import make_server

from data_quality_checker.storage import Store

playwright = pytest.importorskip("playwright.sync_api")
_CHROMIUM = sorted(
    glob.glob(os.path.expanduser("~/Library/Caches/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell"))
)
pytestmark = pytest.mark.skipif(not _CHROMIUM, reason="no Chromium build for Playwright")


@pytest.fixture(scope="module")
def ui(tmp_path_factory):
    config, app = build(tmp_path_factory.mktemp("review_ui"))
    server = make_server("127.0.0.1", 0, app, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=_CHROMIUM[-1])
        yield config, browser, base
        browser.close()
    server.shutdown()


@pytest.fixture
def page(ui):
    _, browser, base = ui
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.set_default_timeout(5000)
    errors: list[str] = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.base = base
    yield page
    page.close()
    assert not errors


def _row(page, kanun_no: str, madde: str):
    return page.locator("table.cmp tbody tr[data-row]").filter(has_text=kanun_no).filter(has_text=f"madde {madde}").first


def test_only_differences_can_be_shown(page) -> None:
    page.goto(page.base + "/review/d000001")
    same = page.locator("table.cmp tbody tr.same")
    assert same.count() == 2 and same.first.is_visible()
    page.click("#only-diff")
    assert not same.first.is_visible()
    assert page.locator("table.cmp tbody tr.only_b").first.is_visible()
    page.click("#only-diff")
    assert same.first.is_visible()


def test_the_legend_counts_the_rows_of_each_kind(page) -> None:
    page.goto(page.base + "/review/d000001")
    assert page.locator(".legend [data-kind=b] .count").inner_text() == "1"
    assert page.locator(".legend [data-kind=both] .count").inner_text() == "2"


def test_the_navigator_steps_through_a_reference_s_mentions(page) -> None:
    page.goto(page.base + "/review/d000001")
    _row(page, "193", "94").click()
    assert page.locator("#nav-count").inner_text() == "1 / 2"
    first = page.locator("#doctext mark.current").inner_text()
    page.click("#nav-next")
    assert page.locator("#nav-count").inner_text() == "2 / 2"
    assert page.locator("#doctext mark.current").inner_text() != first
    page.click("#nav-prev")
    assert page.locator("#nav-count").inner_text() == "1 / 2"
    assert "193" in page.locator("#nav-label").inner_text()


def test_a_reference_cited_nowhere_says_so_when_clicked(page) -> None:
    page.goto(page.base + "/review/d000002")
    _row(page, "3065", "2").click()
    assert "bulunamadı" in page.locator("#nav-count").inner_text()
    assert page.locator("#doctext mark.current").count() == 0


def test_the_scrollbar_marks_where_the_highlights_are(page) -> None:
    page.goto(page.base + "/review/d000007")
    ticks = page.locator("#minimap i")
    assert ticks.count() >= 4
    tops = [float(t.evaluate("e => parseFloat(e.style.top)")) for t in ticks.all()]
    assert all(0 <= top <= 100 for top in tops) and len(set(tops)) > 1


def test_a_revision_can_be_composed_from_the_table_and_saves_as_composed(ui, page) -> None:
    config, _, _ = ui
    page.goto(page.base + "/review/d000004")
    assert page.locator("#compose").is_disabled()
    _row(page, "213", "mükerrer 257").locator("input.pick").check()
    _row(page, "7194", "geçici 1").locator("input.pick").check()
    page.click("#compose")
    assert page.locator("#revision-panel").get_attribute("open") is not None
    cells = page.locator("#ref-rows tr")
    assert cells.count() == 2
    picked = {(cells.nth(i).locator("input[data-field=kanun_no]").input_value(),
               cells.nth(i).locator("input[data-field=madde]").input_value()) for i in range(2)}
    assert picked == {("213", "mükerrer 257"), ("7194", "geçici 1")}
    page.click("button[value=revise]")
    page.wait_for_function("!location.pathname.endsWith('d000004')")
    with Store(config.database_path) as store:
        review = store.get_review("batch", "d000004")
    assert review["action"] == "revise"
    assert {(r["kanun_no"], r["madde"]) for r in json.loads(review["final_references_json"])} == picked


def test_preparation_warnings_are_explained_not_dumped(page) -> None:
    page.goto(page.base + "/review/d000008")
    card = page.locator(".notes-card")
    assert card.is_visible()
    assert "reference_null_field" not in page.content()
    attention = page.locator(".notes-card .note-attention")
    assert attention.count() == 1 and "madde platformdan boş geldi" in attention.inner_text()
    assert "2. satırında" in attention.inner_text()
    assert "1 bilgi notu" in page.locator(".note-info > summary").inner_text()


def test_composing_a_reference_cited_nowhere_warns_before_saving(page) -> None:
    page.goto(page.base + "/review/d000004")
    assert not page.locator("#compose-warn").is_visible()
    _row(page, "213", "257").locator("input.pick").check()
    page.click("#compose")
    warn = page.locator("#compose-warn")
    assert warn.is_visible() and "213" in warn.inner_text() and "bulunamadı" in warn.inner_text()


def test_rows_without_an_article_or_a_number_are_tagged(page) -> None:
    page.goto(page.base + "/review/d000008")
    row = _row(page, "213", "—")
    assert "maddesiz" in row.inner_text()


def test_a_kind_with_no_rows_is_dimmed_in_the_legend(page) -> None:
    page.goto(page.base + "/review/d000001")
    assert "zero" in page.locator(".legend [data-kind=mixed]").get_attribute("class")
    assert "zero" not in (page.locator(".legend [data-kind=b]").get_attribute("class") or "")


def test_a_range_row_in_the_revision_table_expands_to_one_row_per_article(ui, page) -> None:
    """User (7 Oct 2026): "473 ila 475" means 473, 474 and 475, each its own row."""
    config, _, _ = ui
    page.goto(page.base + "/review/d000004")
    page.click("#revision-panel > summary")
    page.click("#fill-a")
    rows = page.locator("#ref-rows tr")
    before = rows.count()
    index = next(i for i in range(before) if rows.nth(i).locator("input[data-field=madde]").input_value() == "620 ila 645")
    assert rows.nth(index).locator("button.expand-range").count() == 1
    rows.nth(index).locator("button.expand-range").click()
    assert rows.count() == before - 1 + 26
    maddes = [rows.nth(i).locator("input[data-field=madde]").input_value() for i in range(rows.count())]
    assert [m for m in maddes if m.isdigit() and 620 <= int(m) <= 645] == [str(n) for n in range(620, 646)]
    assert all(rows.nth(i).locator("input[data-field=kanun_no]").input_value() == "6098"
               for i in range(rows.count()) if maddes[i].isdigit() and 620 <= int(maddes[i]) <= 645)
    assert page.locator("button.expand-range").count() == 0
    page.click("button[value=revise]")
    page.wait_for_function("!location.pathname.endsWith('d000004')")
    with Store(config.database_path) as store:
        final = json.loads(store.get_review("batch", "d000004")["final_references_json"])
    assert {r["madde"] for r in final if r["kanun_no"] == "6098"} == {str(n) for n in range(620, 646)}
