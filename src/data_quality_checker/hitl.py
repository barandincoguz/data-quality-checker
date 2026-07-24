"""Authenticated, CSRF-protected local HITL review application."""

from __future__ import annotations

import hmac
import json
import os
import secrets
from typing import Any

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template_string,
    request,
    session,
    url_for,
)

from .atomic import write_json_atomic
from .config import AppConfig
from .constants import EXPERT_ACTIONS
from .contracts import validate_reference_list
from .errors import ConfigurationError, ContractError, GateBlocked, VersionConflict
from .judges import blind_candidates, ensure_green_audit_plan
from .normalization import (
    compact_references,
    conflicting_law_identity,
    core_identity,
    full_identity,
)
from .storage import Store
from .text import evidence_match_mode
from .web import create_app


LOGIN_TEMPLATE = """
<!doctype html><meta charset="utf-8"><title>DQCheck Login</title>
<h1>DQCheck HITL</h1>
{% if error %}<p role="alert">{{ error }}</p>{% endif %}
<form method="post">
  <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
  <label>Access token <input name="access_token" type="password" required></label>
  <button type="submit">Giriş</button>
</form>
"""

REVIEW_TEMPLATE = """
<!doctype html><meta charset="utf-8"><title>DQCheck Review</title>
{% macro refcell(refs) %}
{%- for r in refs -%}
<div class="ref">m.{{ r.madde or "—" }}{% if r.fikra %}/f.{{ r.fikra }}{% endif %}{% if r.bent %}/b.{{ r.bent }}{% endif %}
<div class="src">{{ r.source_text }}</div></div>
{%- else -%}<span class="muted">—</span>{%- endfor -%}
{% endmacro %}
<style>
body{font-family:system-ui;max-width:1200px;margin:auto;padding:1rem}
pre{white-space:pre-wrap;border:1px solid #bbb;padding:1rem}
.warn{color:#8a3b00}.muted{color:#999}
table{border-collapse:collapse;width:100%;margin:1rem 0}
th,td{border:1px solid #ccc;padding:.4rem;vertical-align:top;text-align:left;font-size:.9rem}
th{background:#f3f3f3}
tr.same td{background:#f4fbf4}tr.only_a td{background:#e8f2ff}tr.only_b td{background:#fff1e6}tr.differs td{background:#fff3cd}
.badge{font-size:.72rem;padding:.05rem .4rem;border-radius:.6rem;border:1px solid #999}
.diffmark{font-weight:700;color:#8a3b00}
.ref{margin-bottom:.3rem}.src{color:#555;font-size:.8rem}
.editor input{margin:0}.hint{color:#666;font-size:.8rem}
.actions button{margin-right:.4rem;padding:.4rem .7rem}
</style>
<h1>{{ doc.internal_doc_id }} — {{ doc.router_bucket }}
{% if position %}<small class="muted">({{ position.index }}/{{ position.total }})</small>{% endif %}</h1>
{% if doc.warnings %}<p class="warn">⚠ {{ doc.warnings|tojson }}</p>{% endif %}
<pre>{{ doc.text }}</pre>

<h2>A / B karşılaştırması</h2>
<table>
<thead><tr><th>Kanun / Madde</th><th>A</th><th>B</th><th>Durum</th></tr></thead>
<tbody>
{% for row in diff %}
<tr class="{{ row.status }}">
  <td>{{ row.core.kanun_no }} · {{ row.core.kanun_ad }}<br><small>madde {{ row.core.madde or "—" }}</small></td>
  <td>{{ refcell(row.a) }}</td>
  <td>{{ refcell(row.b) }}</td>
  <td><span class="badge">{{ row.status }}</span>
  {% if row.field_diffs %}<div class="diffmark">≠ {{ row.field_diffs|join(", ") }}</div>{% endif %}</td>
</tr>
{% else %}
<tr><td colspan="4" class="muted">Aday referans yok</td></tr>
{% endfor %}
</tbody>
</table>
<details><summary>Ham A/B JSON</summary>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">
<pre>{{ doc.candidate_a|tojson(indent=2) }}</pre><pre>{{ doc.candidate_b|tojson(indent=2) }}</pre></div></details>
{% if doc.judge_suggestion %}<details><summary>Judge önerisi</summary><pre>{{ doc.judge_suggestion|tojson(indent=2) }}</pre></details>{% endif %}

<form method="post" action="{{ url_for('submit_review', internal_doc_id=doc.internal_doc_id) }}">
  <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
  <input type="hidden" name="row_version" value="{{ doc.row_version }}">
  <div class="actions">
    <button name="action" value="accept_candidate_a" accesskey="a">A'yı kabul et <span class="hint">(A)</span></button>
    <button name="action" value="accept_candidate_b" accesskey="b">B'yi kabul et <span class="hint">(B)</span></button>
    <button name="action" value="defer">Ertele</button>
    <label>Gerekçe <input name="reason"></label>
  </div>

  <h2>Revize editörü</h2>
  <p class="hint">Aşağıdaki tablodan referansları düzenleyin; kaydederken JSON'a çevrilir.
  Doldur: <button type="button" id="fill-a">A'dan</button>
  <button type="button" id="fill-b">B'den</button>
  {% if doc.judge_suggestion %}<button type="button" id="fill-judge">Judge'dan</button>{% endif %}
  <button type="button" id="add-row">+ satır</button></p>
  <table class="editor"><thead><tr>
    <th>kanun_no</th><th>kanun_ad</th><th>madde</th><th>fikra</th><th>bent</th><th>source_text</th><th></th>
  </tr></thead><tbody id="ref-rows"></tbody></table>
  <details><summary>Ham JSON (gelişmiş / JS kapalı)</summary>
  <textarea name="references_json" rows="8" cols="80"></textarea></details>
  <p><button name="action" value="revise">Revizyonu kaydet</button></p>
</form>

<script id="cand" type="application/json">{{ {"a": doc.candidate_a, "b": doc.candidate_b, "judge": doc.judge_suggestion}|tojson }}</script>
<script>
(function(){
  var DATA = JSON.parse(document.getElementById('cand').textContent);
  var FIELDS = ['kanun_no','kanun_ad','madde','fikra','bent','source_text'];
  var rows = document.getElementById('ref-rows');
  var textarea = document.querySelector('textarea[name=references_json]');
  function collect(){
    var out=[]; rows.querySelectorAll('tr').forEach(function(tr){
      var o={}, any=false;
      tr.querySelectorAll('input').forEach(function(i){o[i.dataset.field]=i.value; if(i.value.trim())any=true;});
      if(any) out.push(o);
    }); return out;
  }
  function sync(){ textarea.value = JSON.stringify(collect()); }
  function addRow(ref){
    ref = ref||{}; var tr=document.createElement('tr');
    FIELDS.forEach(function(f){
      var td=document.createElement('td'), inp=document.createElement('input');
      inp.dataset.field=f; inp.value=ref[f]||''; inp.size=(f==='source_text')?40:8;
      inp.addEventListener('input', sync); td.appendChild(inp); tr.appendChild(td);
    });
    var td=document.createElement('td'), del=document.createElement('button');
    del.type='button'; del.textContent='sil';
    del.addEventListener('click', function(){ tr.remove(); sync(); });
    td.appendChild(del); tr.appendChild(td); rows.appendChild(tr);
  }
  function fill(list){ rows.innerHTML=''; (list||[]).forEach(addRow); if(!rows.children.length) addRow(); sync(); }
  document.getElementById('fill-a').addEventListener('click', function(){ fill(DATA.a); });
  document.getElementById('fill-b').addEventListener('click', function(){ fill(DATA.b); });
  var jb=document.getElementById('fill-judge');
  if(jb && DATA.judge){ jb.addEventListener('click', function(){ fill(DATA.judge.final_references||[]); }); }
  document.getElementById('add-row').addEventListener('click', function(){ addRow(); });
  document.querySelector('button[value=revise]').addEventListener('click', sync);
  fill(DATA.a);
  document.addEventListener('keydown', function(e){
    var t=(e.target.tagName||''); if(t==='INPUT'||t==='TEXTAREA'||t==='SELECT') return;
    var k=e.key.toLowerCase();
    if(k==='a'||k==='b'){
      var val = k==='a' ? 'accept_candidate_a' : 'accept_candidate_b';
      var btn=document.querySelector('button[value='+val+']');
      if(btn){ e.preventDefault(); btn.click(); }
    } else if(k==='d'){
      var reason=document.querySelector('input[name=reason]'); if(reason){ e.preventDefault(); reason.focus(); }
    }
  });
})();
</script>
"""


_DIFF_FIELDS = ("fikra", "bent", "source_text")


def ab_diff(
    candidate_a: list[dict[str, Any]], candidate_b: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Align two blind candidate reference sets by core law-article identity.

    Pure and blind: the result never records which side is human vs model.
    References are grouped by ``core_identity`` in first-appearance order across
    A then B. Each group reports a ``status`` and, when a matched core differs
    one-to-one, the subset of ``fikra``/``bent``/``source_text`` that disagree.
    """

    order: list[tuple[str, str, str, str]] = []
    groups: dict[tuple[str, str, str, str], dict[str, list[dict[str, Any]]]] = {}
    for label, references in (("a", candidate_a), ("b", candidate_b)):
        for reference in references:
            key = core_identity(reference)
            if key not in groups:
                groups[key] = {"a": [], "b": []}
                order.append(key)
            groups[key][label].append(reference)

    rows: list[dict[str, Any]] = []
    for key in order:
        a_refs = groups[key]["a"]
        b_refs = groups[key]["b"]
        if a_refs and not b_refs:
            status = "only_a"
        elif b_refs and not a_refs:
            status = "only_b"
        elif sorted(full_identity(r) for r in a_refs) == sorted(
            full_identity(r) for r in b_refs
        ):
            status = "same"
        else:
            status = "differs"
        field_diffs: list[str] = []
        if status == "differs" and len(a_refs) == 1 and len(b_refs) == 1:
            field_diffs = [
                field
                for field in _DIFF_FIELDS
                if a_refs[0].get(field, "") != b_refs[0].get(field, "")
            ]
        sample = (a_refs or b_refs)[0]
        rows.append(
            {
                "core": {
                    "kanun_no": sample["kanun_no"],
                    "kanun_ad": sample["kanun_ad"],
                    "madde": sample["madde"],
                },
                "status": status,
                "a": a_refs,
                "b": b_refs,
                "field_diffs": field_diffs,
            }
        )
    return rows


def review_position(
    queue: list[dict[str, Any]], internal_doc_id: str
) -> dict[str, int] | None:
    """1-based position of a document within the ordered review queue."""

    for index, row in enumerate(queue, start=1):
        if row["internal_doc_id"] == internal_doc_id:
            return {"index": index, "total": len(queue)}
    return None


def _secure_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def _review_requirements(
    *, config: AppConfig, store: Store, batch_id: str
) -> tuple[set[str], dict[str, int]]:
    documents = store.list_documents(batch_id)
    audit = ensure_green_audit_plan(config=config, store=store, batch_id=batch_id)
    audit_ids = {str(value) for value in audit["sample_internal_doc_ids"]}
    escalation_path = config.public_root / "batches" / batch_id / "green_escalation.json"
    escalated = escalation_path.exists()
    required: set[str] = set(audit_ids)
    for document in documents:
        if document["router_bucket"] in {"RED", "YELLOW"}:
            required.add(document["internal_doc_id"])
        elif escalated and document["router_bucket"] == "GREEN":
            required.add(document["internal_doc_id"])
    priority: dict[str, int] = {}
    for document in documents:
        doc_id = document["internal_doc_id"]
        if document["router_bucket"] == "RED":
            priority[doc_id] = 0
        elif document["router_bucket"] == "YELLOW":
            priority[doc_id] = 1
        elif doc_id in audit_ids:
            priority[doc_id] = 2
        elif escalated and document["router_bucket"] == "GREEN":
            priority[doc_id] = 3
    return required, priority


def review_queue(*, config: AppConfig, store: Store, batch_id: str) -> list[dict[str, Any]]:
    required, priority = _review_requirements(config=config, store=store, batch_id=batch_id)
    documents = {row["internal_doc_id"]: row for row in store.list_documents(batch_id)}
    reviews = {row["internal_doc_id"]: row for row in store.list_reviews(batch_id)}
    queue = []
    for doc_id in required:
        review = reviews[doc_id]
        if review["status"] == "finalized":
            continue
        queue.append(
            {
                "internal_doc_id": doc_id,
                "public_doc_id": documents[doc_id]["public_doc_id"],
                "router_bucket": documents[doc_id]["router_bucket"],
                "review_status": review["status"],
                "row_version": review["row_version"],
                "priority": priority[doc_id],
            }
        )
    queue.sort(
        key=lambda row: (
            row["priority"],
            1 if row["review_status"] == "deferred" else 0,
            row["internal_doc_id"],
        )
    )
    return queue


def _locked_judge(config: AppConfig, batch_id: str) -> str | None:
    path = config.public_root / "batches" / batch_id / "judge_lock.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["model"])


def _document_for_review(
    *, config: AppConfig, store: Store, batch_id: str, internal_doc_id: str
) -> dict[str, Any]:
    document = store.get_document(batch_id, internal_doc_id)
    review = store.get_review(batch_id, internal_doc_id)
    prediction = store.get_prediction(batch_id, internal_doc_id, "G0")
    if document is None or review is None or prediction is None:
        raise KeyError(internal_doc_id)
    human = json.loads(document["human_references_json"])
    model = json.loads(prediction["references_json"])
    mapping, candidate_a, candidate_b = blind_candidates(
        batch_id=batch_id,
        internal_doc_id=internal_doc_id,
        model="hitl",
        human_references=human,
        model_references=model,
    )
    judge_model = _locked_judge(config, batch_id)
    judge_suggestion = None
    if judge_model:
        judge = store.get_judge_result(batch_id, internal_doc_id, judge_model)
        if judge and judge["status"] == "valid":
            judge_suggestion = json.loads(judge["result_json"])
    result = {
        "internal_doc_id": internal_doc_id,
        "public_doc_id": document["public_doc_id"],
        "router_bucket": document["router_bucket"],
        "text": document["text"],
        "warnings": json.loads(document["warnings_json"]),
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "judge_suggestion": judge_suggestion,
        "review_status": review["status"],
        "row_version": review["row_version"],
        "evidence_spans": _evidence_spans(document["text"], candidate_a, candidate_b),
    }
    if review["status"] == "finalized":
        result["blind_mapping_revealed"] = mapping
        result["final_references"] = json.loads(review["final_references_json"] or "[]")
        result["action"] = review["action"]
    return result


def _evidence_spans(
    text: str,
    candidate_a: list[dict[str, str]],
    candidate_b: list[dict[str, str]],
) -> list[dict[str, Any]]:
    spans = []
    for label, references in (("A", candidate_a), ("B", candidate_b)):
        for reference in references:
            evidence = reference.get("source_text", "")
            if not evidence:
                continue
            start = text.find(evidence)
            if start >= 0:
                spans.append({"start": start, "end": start + len(evidence), "candidate": label})
    return sorted(spans, key=lambda item: (item["start"], item["end"], item["candidate"]))


def validate_final_references(
    references: Any, *, document_text: str
) -> list[dict[str, str]]:
    validated = validate_reference_list(references)
    normalized = compact_references(validated)
    if len(normalized) != len(validated):
        raise ContractError("final references contain duplicate or generic-shadowed rows")
    if conflicting_law_identity(normalized):
        raise ContractError("final references contain conflicting law identity")
    for index, reference in enumerate(validated):
        if not reference["kanun_no"] and not reference["kanun_ad"]:
            raise ContractError(f"final reference[{index}] has no law identity")
        if not reference["source_text"] or evidence_match_mode(
            reference["source_text"], document_text
        ) is None:
            raise ContractError(f"final reference[{index}] evidence is absent from document")
    return validated


def _trigger_green_escalation(
    *, config: AppConfig, store: Store, batch_id: str, document: dict[str, Any], final: list[dict[str, str]]
) -> bool:
    if document["router_bucket"] != "GREEN":
        return False
    audit_path = config.public_root / "batches" / batch_id / "green_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if document["internal_doc_id"] not in set(audit["sample_internal_doc_ids"]):
        return False
    original = json.loads(document["human_references_json"])
    original_set = {full_identity(row) for row in compact_references(original)}
    final_set = {full_identity(row) for row in compact_references(final)}
    if original_set == final_set:
        return False
    path = config.public_root / "batches" / batch_id / "green_escalation.json"
    payload = {
        "schema_version": 1,
        "batch_id": batch_id,
        "trigger_internal_doc_id": document["internal_doc_id"],
        "trigger_public_doc_id": document["public_doc_id"],
        "reason": "audited_GREEN_legal_membership_changed",
        "remaining_green_requires_judge_and_expert": True,
    }
    if not path.exists():
        write_json_atomic(path, payload, mode=0o644)
        store.add_event(batch_id, "green_escalated", payload)
    return True


def _json_request_payload() -> dict[str, Any]:
    if request.is_json:
        payload = request.get_json(silent=False)
        if not isinstance(payload, dict):
            raise ContractError("request JSON root must be an object")
        return payload
    payload = request.form.to_dict()
    if payload.get("references_json"):
        try:
            payload["references"] = json.loads(payload["references_json"])
        except json.JSONDecodeError as exc:
            raise ContractError(f"references_json is invalid: {exc}") from exc
    return payload


def create_hitl_app(
    *,
    config: AppConfig,
    batch_id: str,
    testing: bool = False,
    session_secret: str | None = None,
    access_token: str | None = None,
) -> Flask:
    secret = session_secret or os.environ.get("DQCHECK_SESSION_SECRET")
    token = access_token or os.environ.get("DQCHECK_ACCESS_TOKEN")
    if not secret or len(secret) < 32:
        raise ConfigurationError("DQCHECK_SESSION_SECRET must contain at least 32 characters")
    if not token or len(token) < 32:
        raise ConfigurationError("DQCHECK_ACCESS_TOKEN must contain at least 32 characters")
    with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
        batch = store.get_batch(batch_id)
        if batch is None or batch["status"] != "processed":
            raise GateBlocked("HITL requires a processed batch")
        ensure_green_audit_plan(config=config, store=store, batch_id=batch_id)

    app = create_app(
        config,
        batch_id=batch_id,
        testing=testing,
        session_secret=secret,
    )

    @app.before_request
    def protect() -> Any:
        if request.endpoint in {"healthz", "login"}:
            return None
        authorization = request.headers.get("Authorization", "")
        if authorization.startswith("Bearer ") and _secure_equal(authorization[7:], token):
            session["authenticated"] = True
            session.setdefault("csrf_token", secrets.token_urlsafe(32))
        if not session.get("authenticated"):
            return jsonify({"error": "authentication_required"}), 401
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            submitted = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token", "")
            expected = str(session.get("csrf_token") or "")
            if not submitted or not expected or not _secure_equal(submitted, expected):
                return jsonify({"error": "csrf_failed"}), 403
        return None

    @app.route("/login", methods=["GET", "POST"])
    def login() -> Any:
        session.setdefault("login_csrf", secrets.token_urlsafe(32))
        error = None
        if request.method == "POST":
            submitted_csrf = request.form.get("csrf_token", "")
            submitted_token = request.form.get("access_token", "")
            if not _secure_equal(submitted_csrf, str(session["login_csrf"])):
                error = "CSRF doğrulaması başarısız"
            elif not _secure_equal(submitted_token, token):
                error = "Erişim tokenı geçersiz"
            else:
                session["authenticated"] = True
                session["csrf_token"] = secrets.token_urlsafe(32)
                return redirect(url_for("next_review"))
        return render_template_string(
            LOGIN_TEMPLATE, csrf_token=session["login_csrf"], error=error
        )

    @app.get("/api/session")
    def api_session() -> Any:
        return jsonify({"csrf_token": session["csrf_token"], "batch_id": batch_id})

    @app.get("/api/queue")
    def api_queue() -> Any:
        with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
            return jsonify({"queue": review_queue(config=config, store=store, batch_id=batch_id)})

    @app.get("/")
    def index() -> Any:
        return redirect(url_for("next_review"))

    @app.get("/review/next")
    def next_review() -> Any:
        with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
            queue = review_queue(config=config, store=store, batch_id=batch_id)
        if not queue:
            return jsonify({"status": "review_complete", "batch_id": batch_id})
        return redirect(url_for("review_document", internal_doc_id=queue[0]["internal_doc_id"]))

    @app.get("/api/documents/<internal_doc_id>")
    def api_document(internal_doc_id: str) -> Any:
        with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
            return jsonify(
                _document_for_review(
                    config=config,
                    store=store,
                    batch_id=batch_id,
                    internal_doc_id=internal_doc_id,
                )
            )

    @app.get("/review/<internal_doc_id>")
    def review_document(internal_doc_id: str) -> Any:
        with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
            document = _document_for_review(
                config=config,
                store=store,
                batch_id=batch_id,
                internal_doc_id=internal_doc_id,
            )
            queue = review_queue(config=config, store=store, batch_id=batch_id)
        return render_template_string(
            REVIEW_TEMPLATE,
            doc=document,
            diff=ab_diff(document["candidate_a"], document["candidate_b"]),
            position=review_position(queue, internal_doc_id),
            csrf_token=session["csrf_token"],
        )

    @app.post("/api/reviews/<internal_doc_id>")
    @app.post("/review/<internal_doc_id>")
    def submit_review(internal_doc_id: str) -> Any:
        try:
            payload = _json_request_payload()
            expected_version = int(payload.get("row_version"))
            requested_action = str(payload.get("action") or "")
            reason = str(payload.get("reason") or "").strip() or None
            reviewer = str(payload.get("reviewer") or "local_expert")
            with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
                document = store.get_document(batch_id, internal_doc_id)
                prediction = store.get_prediction(batch_id, internal_doc_id, "G0")
                if document is None or prediction is None:
                    raise KeyError(internal_doc_id)
                human = json.loads(document["human_references_json"])
                model = json.loads(prediction["references_json"])
                mapping, candidate_a, candidate_b = blind_candidates(
                    batch_id=batch_id,
                    internal_doc_id=internal_doc_id,
                    model="hitl",
                    human_references=human,
                    model_references=model,
                )
                if requested_action in {"accept_candidate_a", "accept_candidate_b"}:
                    selected_label = "A" if requested_action.endswith("_a") else "B"
                    selected_is_human = f"{selected_label}=human" in mapping
                    action = "accept_human" if selected_is_human else "accept_model"
                    final = candidate_a if selected_label == "A" else candidate_b
                elif requested_action == "accept_human":
                    action, final = "accept_human", human
                elif requested_action == "accept_model":
                    action, final = "accept_model", model
                elif requested_action == "revise":
                    action, final = "revise", payload.get("references")
                elif requested_action == "defer":
                    action, final = "defer", None
                    if not reason:
                        raise ContractError("defer requires a reason")
                elif requested_action == "judge_override":
                    action = "judge_override"
                    if not reason:
                        raise ContractError("judge_override requires a reason")
                    judge_model = _locked_judge(config, batch_id)
                    if not judge_model:
                        raise GateBlocked("no production judge is locked")
                    judge = store.get_judge_result(batch_id, internal_doc_id, judge_model)
                    if judge is None or judge["status"] != "valid":
                        raise GateBlocked("locked judge has no valid result for this document")
                    final = json.loads(judge["result_json"])["final_references"]
                else:
                    raise ContractError(f"unsupported expert action: {requested_action}")
                if action not in EXPERT_ACTIONS:
                    raise ContractError(f"unsupported canonical action: {action}")
                if action == "defer":
                    validated_final = None
                    status = "deferred"
                else:
                    validated_final = validate_final_references(final, document_text=document["text"])
                    status = "finalized"
                review = store.update_review(
                    batch_id=batch_id,
                    internal_doc_id=internal_doc_id,
                    expected_version=expected_version,
                    status=status,
                    action=action,
                    final_references=validated_final,
                    reason=reason,
                    reviewer=reviewer,
                )
                escalated = bool(
                    validated_final is not None
                    and _trigger_green_escalation(
                        config=config,
                        store=store,
                        batch_id=batch_id,
                        document=document,
                        final=validated_final,
                    )
                )
                store.add_event(
                    batch_id,
                    "expert_review_updated",
                    {
                        "internal_doc_id": internal_doc_id,
                        "status": status,
                        "action": action,
                        "row_version": review["row_version"],
                        "green_escalated": escalated,
                    },
                )
            response = {
                "status": status,
                "action": action,
                "row_version": review["row_version"],
                "blind_mapping_revealed": mapping if status == "finalized" else None,
                "green_escalated": escalated,
            }
            if request.is_json:
                return jsonify(response)
            return redirect(url_for("next_review"))
        except VersionConflict as exc:
            return jsonify({"error": "version_conflict", "detail": str(exc)}), 409
        except (ContractError, GateBlocked, KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": "invalid_review", "detail": str(exc)}), 400

    return app
