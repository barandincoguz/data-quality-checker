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
from .judges import ensure_green_audit_plan
from .normalization import (
    compact_references,
    conflicting_law_identity,
    core_identity,
    full_identity,
)
from .preparation import annotation_attribution_path
from .reference_policy import (
    DEFAULT_REFERENCE_POLICY_ID,
    apply_reference_policy,
    reference_policy_spec,
)
from .review_backup import (
    create_review_backup,
    review_backup_lock,
    review_backup_status,
)
from .storage import Store
from .text import evidence_match_mode
from .web import create_app

LOGIN_TEMPLATE = """
<!doctype html><meta charset="utf-8"><title>DQCheck Giriş</title>
<style>
body{font-family:system-ui;background:#f5f6f8;margin:0;color:#20242a}
main{max-width:440px;margin:12vh auto;background:white;padding:2rem;border:1px solid #d9dde3;border-radius:.7rem}
label{display:block;font-weight:650;margin:1.2rem 0 .4rem}input{box-sizing:border-box;width:100%;padding:.75rem;border:1px solid #9aa3ad;border-radius:.4rem}
button{width:100%;margin-top:1rem;padding:.8rem;border:0;border-radius:.4rem;background:#155eef;color:white;font-weight:700}
.hint{color:#59636e;line-height:1.45}.error{color:#a12622;background:#fff1f0;padding:.7rem}
</style>
<main><h1>DQCheck belge inceleme</h1>
<p class="hint">Model çıktısı ile insan anotasyonunu belge metnine göre karşılaştırın.</p>
{% if error %}<p class="error" role="alert">{{ error }}</p>{% endif %}
<form method="post">
  <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
  <label for="access-token">Erişim anahtarı</label>
  <input id="access-token" name="access_token" type="password" autocomplete="current-password" required autofocus>
  <button type="submit">İncelemeye başla</button>
</form></main>
"""

DURABILITY_PENDING_TEMPLATE = """
<!doctype html><meta charset="utf-8"><title>DQCheck kayıt uyarısı</title>
<style>
body{font-family:system-ui;background:#f5f6f8;margin:0;color:#20242a}
main{max-width:620px;margin:12vh auto;background:white;padding:2rem;border:1px solid #d9dde3;border-radius:.7rem}
.warn{border-left:5px solid #b54708;background:#fff7e6;padding:1rem;line-height:1.5}
a{display:inline-block;margin-top:1rem;color:#155eef;font-weight:700}
</style>
<main><h1>Karar kaydedildi, yedek bekliyor</h1>
<div class="warn"><strong>Karar ana veritabanına kaydedildi.</strong>
Korumalı snapshot henüz doğrulanamadı. Kararı tekrar göndermeyin;
uygulama bir sonraki açılışta yedeği otomatik tamamlayacaktır.</div>
<a href="{{ url_for('next_review') }}">İnceleme kuyruğuna dön</a></main>
"""

REVIEW_TEMPLATE = """
<!doctype html><meta charset="utf-8"><title>DQCheck Review</title>
{% macro refcell(refs) %}
{%- for r in refs -%}
<div class="ref"><strong>{{ r.kanun_no or r.kanun_ad or "Kanun belirtilmemiş" }}</strong>
· madde {{ r.madde or "—" }}{% if r.fikra %} · fıkra {{ r.fikra }}{% endif %}{% if r.bent %} · bent {{ r.bent }}{% endif %}
<div class="src">{{ r.source_text }}</div></div>
{%- else -%}<span class="muted">—</span>{%- endfor -%}
{% endmacro %}
<style>
*{box-sizing:border-box}body{font-family:system-ui;margin:0;background:#f5f6f8;color:#20242a;line-height:1.45}
main{max-width:1120px;margin:auto;padding:1rem}.topbar{background:white;border-bottom:1px solid #d9dde3;position:sticky;top:0;z-index:10}
.topinner{max-width:1120px;margin:auto;padding:.75rem 1rem}.topline{display:flex;align-items:center;gap:.7rem;flex-wrap:wrap}.topline h1{font-size:1.15rem;margin:0}
.docid{color:#6a737d;font-size:.78rem}.durability{color:#146c3a;font-size:.8rem;font-weight:650;margin-top:.45rem}.progress{height:.45rem;background:#e5e8ec;border-radius:1rem;margin-top:.55rem;overflow:hidden}.progress span{display:block;height:100%;background:#155eef}
.bucket{font-size:.78rem;font-weight:750;padding:.2rem .55rem;border-radius:1rem}.bucket-RED{background:#ffe8e6;color:#9f1d18}.bucket-YELLOW{background:#fff1c2;color:#6b4b00}.bucket-GREEN{background:#dff6e8;color:#146c3a}
section,details.secondary,details.technical{background:white;border:1px solid #d9dde3;border-radius:.55rem;padding:1rem;margin:1rem 0}h2{font-size:1.05rem;margin:.1rem 0 .7rem}
.guide{border-left:5px solid #155eef}.guide p{margin:.35rem 0}.steps{margin:.6rem 0 0;padding-left:1.3rem}.steps li{margin:.25rem 0}.hint{color:#59636e;font-size:.86rem}.warn{color:#8a3b00;background:#fff7e6;padding:.65rem}.muted{color:#7a838d}
.doctext{white-space:pre-wrap;max-height:360px;overflow:auto;border:1px solid #ccd2d9;background:#fbfcfd;padding:1rem;border-radius:.4rem}
table{border-collapse:collapse;width:100%;margin:.8rem 0}th,td{border:1px solid #d4d8dd;padding:.55rem;vertical-align:top;text-align:left;font-size:.88rem}th{background:#f2f4f7}
tr.same td{background:#f7faf8}tr.only_a td{background:#eef5ff}tr.only_b td{background:#fff5e9}tr.differs td{background:#fff7d6}.badge{display:inline-block;font-size:.75rem;font-weight:700;padding:.12rem .45rem;border-radius:.6rem;border:1px solid #9aa3ad}
.diffmark{font-weight:650;color:#8a3b00;margin-top:.3rem}.ref{margin-bottom:.55rem}.src{color:#59636e;font-size:.8rem;margin-top:.15rem}.summary{font-weight:650}
.decision{border:2px solid #155eef}.decision-grid{display:grid;grid-template-columns:1fr 1fr;gap:.7rem;margin-top:.8rem}.decision button{min-height:68px;border:1px solid #155eef;border-radius:.5rem;background:#edf4ff;color:#123a78;font-size:1rem;font-weight:750;padding:.7rem;cursor:pointer}.decision button:hover{background:#dceaff}.decision button span{display:block;font-size:.78rem;font-weight:450;margin-top:.2rem}
summary{cursor:pointer;font-weight:700}.secondary[open]>summary,.technical[open]>summary{margin-bottom:.8rem}.defer-row{display:flex;gap:.6rem;align-items:end;flex-wrap:wrap}.defer-row label{flex:1;min-width:260px}.defer-row input{width:100%;padding:.55rem;margin-top:.25rem}.defer-row button{padding:.58rem .8rem}
.editor-wrap{overflow:auto}.editor input{width:100%;min-width:78px;padding:.35rem}.editor td:last-child{width:1%}.editor .source-input{min-width:260px}.toolbar{display:flex;gap:.4rem;flex-wrap:wrap;margin:.6rem 0}.toolbar button{padding:.4rem .6rem}.save-revision{background:#155eef;color:white;border:0;border-radius:.4rem;font-weight:700;padding:.65rem .9rem}
pre{white-space:pre-wrap;border:1px solid #bbb;padding:1rem}mark.ev{padding:0 .05rem;border-radius:.15rem;color:inherit}mark.ev-A{background:#cfe4ff}mark.ev-B{background:#ffe0c2}mark.ev-A.ev-B{background:#e2d6f5}.legend{font-size:.8rem;color:#555;margin:.4rem 0}.legend .sw{display:inline-block;padding:0 .4rem;border-radius:.2rem;margin-right:.5rem}
@media(max-width:720px){.decision-grid{grid-template-columns:1fr}th,td{font-size:.8rem;padding:.35rem}.topbar{position:static}}
</style>
<header class="topbar"><div class="topinner">
  <div class="topline"><h1>{% if position %}Belge {{ position.index }} / {{ position.total }}{% else %}Belge inceleme{% endif %}</h1>
    <span class="bucket bucket-{{ doc.router_bucket }}">{{ doc.router_bucket }}</span>
    <span class="docid">Kayıt: {{ doc.internal_doc_id }}</span></div>
  {% if position %}<div class="progress" aria-label="İnceleme ilerlemesi"><span style="width:{{ (100 * position.index / position.total)|round(1) }}%"></span></div>{% endif %}
  <div class="durability">Kayıt koruması: Güncel · {{ durability.review_count }} karar yedeklendi</div>
</div></header>
<main>
<section class="guide"><h2>{{ doc.guidance.title }}</h2><p>{{ doc.guidance.explanation }}</p>
  <ol class="steps"><li>Belge metnindeki kanun ve madde atıflarını kontrol edin.</li><li>İnsan anotasyonu ile model çıktısını metinle karşılaştırın.</li><li>Doğru olanı seçin; ikisi de yanlışsa düzeltin.</li></ol>
</section>
{% if doc.warnings %}<p class="warn">Dikkat: Bu belgede hazırlama uyarısı var: {{ doc.warnings|tojson }}</p>{% endif %}

<section><h2>1. Belge metnini kontrol et</h2>
<p class="hint">Renkler yalnız adayların gösterdiği kanıt parçalarını belirtir; doğruluk anlamına gelmez.</p>
<p class="legend"><span class="sw" style="background:#cfe4ff">İnsan anotasyonu kanıtı</span><span class="sw" style="background:#ffe0c2">Model kanıtı</span><span class="sw" style="background:#e2d6f5">İkisinde de</span></p>
<div class="doctext">{% for seg in segments %}{% if seg.candidates %}<mark class="ev{% for c in seg.candidates %} ev-{{ c }}{% endfor %}" title="{{ seg.candidates|join('+') }}">{{ seg.text }}</mark>{% else %}{{ seg.text }}{% endif %}{% endfor %}</div></section>

<section><h2>2. İnsan anotasyonu ile modeli karşılaştır</h2>
<p class="hint">Sol sütun insan anotasyonudur; sağ sütun G0 model çıktısıdır.</p>
<p class="summary">{{ diff_summary.difference_count }} farklı · {{ diff_summary.same_count }} aynı referans grubu</p>
<table><thead><tr><th>Referans</th><th>İnsan anotasyonu<br><small>Tamamlayan: {{ doc.human_attribution.display_name }}</small>{% if doc.human_attribution.last_editor_name and doc.human_attribution.last_editor_name != doc.human_attribution.display_name %}<br><small>Son düzenleyen: {{ doc.human_attribution.last_editor_name }}</small>{% endif %}</th><th>Model çıktısı<br><small>G0 · {{ doc.model_name }}</small></th><th>Sonuç</th></tr></thead><tbody>
{% for row in diff %}<tr class="{{ row.status }}">
  <td>{{ row.core.kanun_no }} · {{ row.core.kanun_ad }}<br><small>madde {{ row.core.madde or "—" }}</small></td>
  <td>{{ refcell(row.a) }}</td><td>{{ refcell(row.b) }}</td>
  <td><span class="badge">{{ row.status_label }}</span>{% if row.field_diffs %}<div class="diffmark">Farklı alan: {{ row.field_diffs|join(", ") }}</div>{% endif %}</td>
</tr>{% else %}<tr><td colspan="4" class="muted">İki adayda da referans yok.</td></tr>{% endfor %}
</tbody></table></section>

<form method="post" action="{{ url_for('submit_review', internal_doc_id=doc.internal_doc_id) }}">
  <input type="hidden" name="csrf_token" value="{{ csrf_token }}"><input type="hidden" name="row_version" value="{{ doc.row_version }}">
  <section class="decision"><h2>3. Karar ver</h2><p>Belgedeki tüm hedef kanun/madde referanslarını doğru veren adayı seçin.</p>
    <div class="decision-grid"><button name="action" value="accept_human" accesskey="h">İnsan anotasyonu doğru<span>{{ doc.human_attribution.display_name }} · Kısayol: H</span></button>
    <button name="action" value="accept_model" accesskey="m">Model çıktısı doğru<span>G0 · Kısayol: M</span></button></div>
  </section>
  <details class="secondary"><summary>Karar veremiyorum — bu belgeyi ertele</summary><div class="defer-row">
    <label>Erteleme gerekçesi<input name="reason" placeholder="Örn. metin eksik veya hukuki yorum gerekli"></label><button name="action" value="defer">Gerekçeyle ertele</button></div>
  </details>
  <details class="secondary" id="revision-panel"><summary>İkisi de tam doğru değil — referansları düzelt</summary>
    <p class="hint">Başlangıç olarak insan anotasyonunu veya modeli seçin; yalnız yanlış alanları değiştirin.</p>
    <div class="toolbar"><button type="button" id="fill-a">İnsan anotasyonundan başla</button><button type="button" id="fill-b">Model çıktısından başla</button>
    {% if doc.judge_suggestion %}<button type="button" id="fill-judge">Yardımcı görüşten başla</button>{% endif %}<button type="button" id="add-row">+ Referans ekle</button></div>
    <div class="editor-wrap"><table class="editor"><thead><tr><th>Kanun no</th><th>Kanun adı</th><th>Madde</th><th>Fıkra</th><th>Bent</th><th>Metindeki ifade</th><th></th></tr></thead><tbody id="ref-rows"></tbody></table></div>
    <details class="technical"><summary>Teknik JSON alanı (yalnız gerekirse)</summary><textarea name="references_json" rows="8" cols="80"></textarea></details>
    <p><button class="save-revision" name="action" value="revise">Düzeltmeyi kaydet ve devam et</button></p>
  </details>
</form>

<details class="technical"><summary>Teknik ayrıntılar</summary>
  <p class="hint">Aktif policy: {{ doc.reference_policy.policy_id }} · İki adaydan çıkarılan boilerplate: {{ doc.reference_policy.removed_reference_count }}</p>
  <details><summary>Ham insan/model JSON</summary><div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem"><div><strong>İnsan anotasyonu</strong><pre>{{ doc.candidate_a|tojson(indent=2) }}</pre></div><div><strong>G0 model çıktısı</strong><pre>{{ doc.candidate_b|tojson(indent=2) }}</pre></div></div></details>
  {% if doc.judge_suggestion %}<details><summary>Otomatik yardımcı görüş</summary><pre>{{ doc.judge_suggestion|tojson(indent=2) }}</pre></details>{% endif %}
</details>

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
      if(f==='source_text') inp.className='source-input';
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
    if(k==='h'||k==='m'){
      var val = k==='h' ? 'accept_human' : 'accept_model';
      var btn=document.querySelector('button[value='+val+']');
      if(btn){ e.preventDefault(); btn.click(); }
    } else if(k==='d'){
      var reason=document.querySelector('input[name=reason]'); if(reason){ e.preventDefault(); reason.focus(); }
    }
  });
})();
</script>
</main>
"""


_DIFF_FIELDS = ("fikra", "bent", "source_text")
_DIFF_STATUS_LABELS = {
    "same": "Aynı",
    "only_a": "Yalnız insan anotasyonunda",
    "only_b": "Yalnız model çıktısında",
    "differs": "Alan farkı",
}


def ab_diff(
    candidate_a: list[dict[str, Any]], candidate_b: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Align human (A) and model (B) references by core law-article identity.

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
        elif sorted(full_identity(r) for r in a_refs) == sorted(full_identity(r) for r in b_refs):
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
                "status_label": _DIFF_STATUS_LABELS[status],
                "a": a_refs,
                "b": b_refs,
                "field_diffs": field_diffs,
            }
        )
    return rows


def review_position(queue: list[dict[str, Any]], internal_doc_id: str) -> dict[str, int] | None:
    """1-based position of a document within the ordered review queue."""

    for index, row in enumerate(queue, start=1):
        if row["internal_doc_id"] == internal_doc_id:
            return {"index": index, "total": len(queue)}
    return None


def summarize_diff(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Return a compact, UI-facing summary of human/model diff rows."""

    same_count = sum(row["status"] == "same" for row in rows)
    return {
        "same_count": same_count,
        "difference_count": len(rows) - same_count,
        "total_count": len(rows),
    }


def review_guidance(bucket: str) -> dict[str, str]:
    """Explain why a document is in review without revealing candidate origin."""

    guidance = {
        "RED": {
            "title": "Kanun veya madde listesi uyuşmuyor",
            "explanation": (
                "Adaylardan biri hedef referansı kaçırmış ya da fazladan bir "
                "referans üretmiş olabilir. Belge metnindeki tüm kanun ve maddeleri "
                "özellikle kontrol edin."
            ),
        },
        "YELLOW": {
            "title": "Referans ayrıntıları uyuşmuyor",
            "explanation": (
                "Ana kanun ve madde büyük ölçüde uyumlu görünüyor; fıkra, bent veya "
                "metindeki kanıt parçasında fark olabilir."
            ),
        },
        "GREEN": {
            "title": "Hızlı kalite kontrol örneği",
            "explanation": (
                "Adaylar uyumlu görünüyor. Bunun bir denetim örneği olduğunu göz "
                "önünde bulundurarak belge metniyle kısa bir doğrulama yapın."
            ),
        },
        "QUARANTINE": {
            "title": "Teknik inceleme gerekiyor",
            "explanation": (
                "Bu belge otomatik akışta güvenle değerlendirilemedi. Metin ve "
                "referansları baştan sona kontrol edin."
            ),
        },
    }
    return guidance.get(
        bucket,
        {
            "title": "Belgeyi kontrol edin",
            "explanation": "A ve B adaylarını belge metnine göre karşılaştırın.",
        },
    )


def human_attribution(
    document_metadata: dict[str, Any],
    imported_attribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize attribution for display, with an explicit missing-data label."""

    attribution = document_metadata.get("annotation_attribution")
    if not isinstance(attribution, dict) or not attribution:
        attribution = imported_attribution or {}
    completed_by = attribution.get("completed_by")
    last_editor = attribution.get("last_editor")
    preferred = completed_by if isinstance(completed_by, dict) else last_editor
    display_name = "Anotatör bilgisi kayıtlı değil"
    if isinstance(preferred, dict):
        display_name = str(preferred.get("username") or display_name)
    last_editor_name = None
    if isinstance(last_editor, dict) and last_editor.get("username"):
        last_editor_name = str(last_editor["username"])
    return {
        **attribution,
        "display_name": display_name,
        "last_editor_name": last_editor_name,
    }


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
    *,
    config: AppConfig,
    store: Store,
    batch_id: str,
    internal_doc_id: str,
    imported_attribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    document = store.get_document(batch_id, internal_doc_id)
    review = store.get_review(batch_id, internal_doc_id)
    prediction = store.get_prediction(batch_id, internal_doc_id, "G0")
    if document is None or review is None or prediction is None:
        raise KeyError(internal_doc_id)
    human, human_audit = apply_reference_policy(json.loads(document["human_references_json"]))
    model, model_audit = apply_reference_policy(json.loads(prediction["references_json"]))
    mapping = "A=human,B=model"
    candidate_a, candidate_b = human, model
    judge_model = _locked_judge(config, batch_id)
    judge_suggestion = None
    if judge_model:
        judge = store.get_judge_result(batch_id, internal_doc_id, judge_model)
        if judge and judge["status"] == "valid":
            judge_suggestion = json.loads(judge["result_json"])
    metadata = json.loads(document["metadata_json"])
    result = {
        "internal_doc_id": internal_doc_id,
        "public_doc_id": document["public_doc_id"],
        "router_bucket": document["router_bucket"],
        "guidance": review_guidance(str(document["router_bucket"])),
        "text": document["text"],
        "warnings": json.loads(document["warnings_json"]),
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "candidate_mapping": mapping,
        "human_attribution": human_attribution(metadata, imported_attribution),
        "model_name": config.model.model_id.rsplit("/", 1)[-1],
        "judge_suggestion": judge_suggestion,
        "review_status": review["status"],
        "row_version": review["row_version"],
        "evidence_spans": _evidence_spans(document["text"], candidate_a, candidate_b),
        "reference_policy": {
            "policy_id": DEFAULT_REFERENCE_POLICY_ID,
            "policy_fingerprint": reference_policy_spec()["fingerprint"],
            "removed_reference_count": (
                human_audit["removed_reference_count"] + model_audit["removed_reference_count"]
            ),
        },
    }
    if review["status"] == "finalized":
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


def evidence_segments(text: str, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Slice ``text`` into contiguous runs annotated with covering candidates.

    Pure and blind. Given evidence spans (``_evidence_spans`` output), return
    ordered segments ``{"text", "candidates"}`` whose concatenated text equals
    the input, where ``candidates`` is the sorted set of candidate labels whose
    span covers that run (``[]`` for plain text). Overlapping A/B spans split
    into their own run carrying both labels; adjacent runs with the same label
    set are merged. Never records which side is human vs model.
    """

    if not text:
        return []
    length = len(text)
    bounds = {0, length}
    for span in spans:
        start = max(0, min(length, span["start"]))
        end = max(0, min(length, span["end"]))
        if end > start:
            bounds.add(start)
            bounds.add(end)
    points = sorted(bounds)
    segments: list[dict[str, Any]] = []
    for lo, hi in zip(points, points[1:]):
        candidates = sorted(
            {span["candidate"] for span in spans if span["start"] <= lo and span["end"] >= hi}
        )
        if segments and segments[-1]["candidates"] == candidates:
            segments[-1]["text"] += text[lo:hi]
        else:
            segments.append({"text": text[lo:hi], "candidates": candidates})
    return segments


def validate_final_references(references: Any, *, document_text: str) -> list[dict[str, str]]:
    validated = validate_reference_list(references)
    validated, _ = apply_reference_policy(validated)
    normalized = compact_references(validated)
    if len(normalized) != len(validated):
        raise ContractError("final references contain duplicate or generic-shadowed rows")
    if conflicting_law_identity(normalized):
        raise ContractError("final references contain conflicting law identity")
    for index, reference in enumerate(validated):
        if not reference["kanun_no"] and not reference["kanun_ad"]:
            raise ContractError(f"final reference[{index}] has no law identity")
        if (
            not reference["source_text"]
            or evidence_match_mode(reference["source_text"], document_text) is None
        ):
            raise ContractError(f"final reference[{index}] evidence is absent from document")
    return validated


def _trigger_green_escalation(
    *,
    config: AppConfig,
    store: Store,
    batch_id: str,
    document: dict[str, Any],
    final: list[dict[str, str]],
) -> bool:
    if document["router_bucket"] != "GREEN":
        return False
    audit_path = config.public_root / "batches" / batch_id / "green_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if document["internal_doc_id"] not in set(audit["sample_internal_doc_ids"]):
        return False
    original, _ = apply_reference_policy(json.loads(document["human_references_json"]))
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
    with (
        review_backup_lock(config=config, batch_id=batch_id),
        Store(
            config.database_path,
            busy_timeout_ms=config.runtime.busy_timeout_ms,
        ) as store,
    ):
        batch = store.get_batch(batch_id)
        if batch is None or batch["status"] != "processed":
            raise GateBlocked("HITL requires a processed batch")
        ensure_green_audit_plan(config=config, store=store, batch_id=batch_id)
        create_review_backup(config=config, store=store, batch_id=batch_id)

    imported_attributions: dict[str, dict[str, Any]] = {}
    attribution_file = annotation_attribution_path(config, batch_id)
    if attribution_file.exists():
        attribution_payload = json.loads(attribution_file.read_text(encoding="utf-8"))
        if attribution_payload.get("batch_id") != batch_id:
            raise ConfigurationError("annotation attribution batch_id mismatch")
        raw_attributions = attribution_payload.get("attributions")
        if not isinstance(raw_attributions, dict):
            raise ConfigurationError("annotation attribution sidecar is invalid")
        imported_attributions = {
            str(doc_id): attribution
            for doc_id, attribution in raw_attributions.items()
            if isinstance(attribution, dict)
        }

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
        return render_template_string(LOGIN_TEMPLATE, csrf_token=session["login_csrf"], error=error)

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
                    imported_attribution=imported_attributions.get(internal_doc_id),
                )
            )

    @app.get("/review/<internal_doc_id>")
    def review_document(internal_doc_id: str) -> Any:
        with (
            review_backup_lock(config=config, batch_id=batch_id),
            Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store,
        ):
            document = _document_for_review(
                config=config,
                store=store,
                batch_id=batch_id,
                internal_doc_id=internal_doc_id,
                imported_attribution=imported_attributions.get(internal_doc_id),
            )
            queue = review_queue(config=config, store=store, batch_id=batch_id)
            durability = review_backup_status(
                config=config,
                store=store,
                batch_id=batch_id,
            )
        diff = ab_diff(document["candidate_a"], document["candidate_b"])
        return render_template_string(
            REVIEW_TEMPLATE,
            doc=document,
            diff=diff,
            diff_summary=summarize_diff(diff),
            position=review_position(queue, internal_doc_id),
            durability=durability,
            segments=evidence_segments(document["text"], document["evidence_spans"]),
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
            with (
                review_backup_lock(config=config, batch_id=batch_id),
                Store(
                    config.database_path,
                    busy_timeout_ms=config.runtime.busy_timeout_ms,
                ) as store,
            ):
                document = store.get_document(batch_id, internal_doc_id)
                prediction = store.get_prediction(batch_id, internal_doc_id, "G0")
                if document is None or prediction is None:
                    raise KeyError(internal_doc_id)
                human, _ = apply_reference_policy(json.loads(document["human_references_json"]))
                model, _ = apply_reference_policy(json.loads(prediction["references_json"]))
                mapping = "A=human,B=model"
                candidate_a, candidate_b = human, model
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
                    validated_final = validate_final_references(
                        final, document_text=document["text"]
                    )
                    status = "finalized"
                review, review_event_id = store.update_review_with_event(
                    batch_id=batch_id,
                    internal_doc_id=internal_doc_id,
                    expected_version=expected_version,
                    status=status,
                    action=action,
                    final_references=validated_final,
                    reason=reason,
                    reviewer=reviewer,
                    event_payload={
                        "internal_doc_id": internal_doc_id,
                        "status": status,
                        "action": action,
                        "row_version": expected_version + 1,
                    },
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
                try:
                    durability = create_review_backup(
                        config=config,
                        store=store,
                        batch_id=batch_id,
                    )
                except Exception as exc:
                    pending = {
                        "error": "durability_pending",
                        "review_saved": True,
                        "status": status,
                        "action": action,
                        "row_version": review["row_version"],
                        "review_event_id": review_event_id,
                        "durability": {
                            "status": "pending",
                            "error_type": type(exc).__name__,
                        },
                    }
                    if request.is_json:
                        return jsonify(pending), 503
                    return render_template_string(DURABILITY_PENDING_TEMPLATE), 503
            response = {
                "status": status,
                "action": action,
                "row_version": review["row_version"],
                "review_event_id": review_event_id,
                "candidate_mapping": mapping,
                "green_escalated": escalated,
                "durability": {
                    "status": durability["status"],
                    "snapshot_sha256": durability["snapshot_sha256"],
                    "review_count": durability["review_count"],
                },
            }
            if request.is_json:
                return jsonify(response)
            return redirect(url_for("next_review"))
        except VersionConflict as exc:
            return jsonify({"error": "version_conflict", "detail": str(exc)}), 409
        except (ContractError, GateBlocked, KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": "invalid_review", "detail": str(exc)}), 400

    return app
