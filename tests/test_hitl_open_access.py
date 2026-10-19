"""The GT750 loop's expert screen opens directly: no access token, no login page.

User decision (1 Oct 2026, the risk stated first: anyone with a session on this
shared machine can reach 127.0.0.1): the expert opens the queue straight away.
Kept, because neither asks the expert for anything: the server listens on
127.0.0.1 only, and every decision still carries the page's CSRF token, so
another site open in the expert's browser cannot submit one through the tunnel.
The old (M01) screen keeps its token.
"""

from __future__ import annotations

import re

from test_hitl import SECRET, fixture

from data_quality_checker.hitl import create_hitl_app
from data_quality_checker.storage import Store


def _loop_app(tmp_path):
    config, _ = fixture(tmp_path, count=2)
    config.raw["loop"] = {}
    app = create_hitl_app(config=config, batch_id="batch", testing=True, session_secret=SECRET)
    return config, app


def test_the_loop_queue_opens_without_a_token(tmp_path) -> None:
    _, app = _loop_app(tmp_path)
    client = app.test_client()
    response = client.get("/review/next")
    assert response.status_code == 302 and "/review/" in response.headers["Location"]
    page = client.get(response.headers["Location"])
    assert page.status_code == 200 and "Belge 1 / " in page.get_data(as_text=True)
    assert client.get("/").status_code in (200, 302)


def test_the_loop_screen_has_no_login_page(tmp_path) -> None:
    _, app = _loop_app(tmp_path)
    client = app.test_client()
    assert client.get("/login").status_code == 404
    assert client.post("/login", data={"access_token": "x"}).status_code == 404


def test_a_decision_still_needs_the_pages_csrf_token(tmp_path) -> None:
    config, app = _loop_app(tmp_path)
    client = app.test_client()
    location = client.get("/review/next").headers["Location"]
    doc_id = location.rsplit("/", 1)[-1]
    html = client.get(location).get_data(as_text=True)
    csrf = re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)
    version = re.search(r'name="row_version" value="(\d+)"', html).group(1)

    forged = client.post(
        f"/review/{doc_id}", data={"row_version": version, "action": "accept_human"}
    )
    assert forged.status_code == 403
    with Store(config.database_path) as store:
        assert store.get_review("batch", doc_id)["status"] == "pending"

    real = client.post(
        f"/review/{doc_id}",
        data={
            "row_version": version,
            "action": "accept_human",
            "csrf_token": csrf,
            "opened_at": "0",
        },
    )
    assert real.status_code == 302
    with Store(config.database_path) as store:
        review = store.get_review("batch", doc_id)
    assert review["status"] == "finalized" and review["reviewer"] == "local_expert"


def test_the_old_screen_still_asks_for_its_token(tmp_path) -> None:
    _, app = fixture(tmp_path, count=1)
    client = app.test_client()
    assert client.get("/review/next").status_code == 401
    assert client.get("/login").status_code == 200


def test_every_decision_event_keeps_the_full_list_and_what_it_replaced(tmp_path) -> None:
    """The journal needs every decision, including one the expert later changes:
    the reviews row keeps only the latest, so each event carries the list, the
    reason and the state it replaced (1 Oct 2026)."""
    import json

    config, app = _loop_app(tmp_path)
    client = app.test_client()
    location = client.get("/review/next").headers["Location"]
    doc_id = location.rsplit("/", 1)[-1]

    def submit(action, **extra):
        html = client.get(f"/review/{doc_id}").get_data(as_text=True)
        csrf = re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)
        version = re.search(r'name="row_version" value="(\d+)"', html).group(1)
        response = client.post(
            f"/review/{doc_id}",
            data={
                "row_version": version,
                "action": action,
                "csrf_token": csrf,
                "opened_at": "0",
                **extra,
            },
        )
        assert response.status_code == 302, response.get_data(as_text=True)[:300]

    def stored():
        with Store(config.database_path) as store:
            return json.loads(store.get_review("batch", doc_id)["final_references_json"])

    submit("accept_human")
    first_list = stored()
    submit("accept_model", reason="ikinci bakış")
    second_list = stored()
    with Store(config.database_path) as store:
        events = [
            json.loads(p)
            for (p,) in store.connection.execute(
                "SELECT payload_json FROM batch_events WHERE event_type='expert_review_updated' ORDER BY event_id"
            )
        ]
    first, second = events
    assert first["final_references"] == first_list and first["previous"]["status"] == "pending"
    assert second["previous"] == {
        "status": "finalized",
        "action": "accept_human",
        "final_references": first_list,
        "reason": None,
    }
    assert second["final_references"] == second_list and second["action"] == "accept_model"
    assert second["reason"] == "ikinci bakış" and second["reviewer"] == "local_expert"
