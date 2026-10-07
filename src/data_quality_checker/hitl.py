"""Authenticated, CSRF-protected local HITL review application."""

from __future__ import annotations

import hmac
import json
import os
import secrets
import time
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
from .contracts import REFERENCE_FIELDS, validate_reference_list
from .errors import ConfigurationError, ContractError, GateBlocked, IntegrityError, VersionConflict
from .judges import ensure_green_audit_plan, judge_lock_payload, judge_visible_to_expert
from .loop_sources import has_loop_settings
from .preparation import annotation_attribution_path
from .reference_locator import mention_offsets
from .reference_matching import DocumentContext, canonical_reference, compare_reference_lists
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
from .web import create_app, validate_runtime_secret

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

BACKUP_BLOCKED_TEMPLATE = """
<!doctype html><meta charset="utf-8"><title>DQCheck kayıt uyarısı</title>
<style>
body{font-family:system-ui;background:#f5f6f8;margin:0;color:#20242a}
main{max-width:620px;margin:12vh auto;background:white;padding:2rem;border:1px solid #d9dde3;border-radius:.7rem}
.warn{border-left:5px solid #b42318;background:#fef3f2;padding:1rem;line-height:1.5}
</style>
<main><h1>Kayıt yedeği alınamıyor</h1>
<div class="warn" role="alert"><strong>Verdiğiniz kararlar ana veritabanında duruyor.</strong>
Korumalı kayıt yedeği şu an alınamıyor; yeni karar vermeden önce sorumluya haber verin.
Sorun giderilince bu sayfayı yenilemeniz yeterli.</div></main>
"""

QUEUE_DONE_TEMPLATE = """
<!doctype html><meta charset="utf-8"><title>DQCheck — kuyruk tamamlandı</title>
<style>
body{font-family:system-ui;background:#f5f6f8;margin:0;color:#20242a}
main{max-width:620px;margin:12vh auto;background:white;padding:2rem;border:1px solid #d9dde3;border-radius:.7rem}
.ok{border-left:5px solid #146c3a;background:#ecfdf3;padding:1rem;line-height:1.5}
</style>
<main><h1>İnceleme kuyruğu tamamlandı</h1>
<div class="ok">{{ batch_id }} partisinde incelenmesi gereken belge kalmadı. Teşekkürler.</div></main>
"""


def _verified_backup(*, config: AppConfig, store: Store, batch_id: str) -> dict[str, Any]:
    """The backup status for the page, caught up first if a decision outran it.

    A decision whose backup failed is already committed; startup would take the
    missing backup, and so does the next page instead of failing on it until a
    restart. `create_review_backup` snapshots the live store and verifies it,
    so a backup that still cannot be made raises here rather than being skipped.
    """
    try:
        return review_backup_status(config=config, store=store, batch_id=batch_id)
    except IntegrityError:
        create_review_backup(config=config, store=store, batch_id=batch_id)
        return review_backup_status(config=config, store=store, batch_id=batch_id)


FORM_ERROR_TEMPLATE = """
<!doctype html><meta charset="utf-8"><title>DQCheck — karar kaydedilmedi</title>
<style>
body{font-family:system-ui;background:#f5f6f8;margin:0;color:#20242a}
main{max-width:620px;margin:12vh auto;background:white;padding:2rem;border:1px solid #d9dde3;border-radius:.7rem}
.warn{border-left:5px solid #b54708;background:#fff7e6;padding:1rem;line-height:1.5}
a{display:inline-block;margin-top:1rem;color:#155eef;font-weight:700}.hint{color:#59636e;font-size:.85rem}
</style>
<main><h1>Karar kaydedilmedi</h1>
<div class="warn" role="alert">{{ message }}</div>
<p class="hint">Teknik ayrıntı: {{ detail }}</p>
<a href="{{ url_for('review_document', internal_doc_id=internal_doc_id) }}">Belgeye dön</a></main>
"""

# What the expert reads when a form submission is refused. The JSON API keeps
# the English `detail`; a person at the form gets a sentence and a way back.
_FORM_ERROR_MESSAGES = {
    "defer requires a reason": "Ertelemek için bir gerekçe yazın.",
    "references_json is invalid": (
        "Teknik JSON alanındaki metin geçerli bir liste değil. Düzeltip yeniden kaydedin "
        "ya da tabloyu kullanın."
    ),
    "no production judge is locked": "Hakem henüz kilitlenmedi; hakemin önerisi seçilemez.",
    "the judge is hidden from the expert in this loop": (
        "Bu döngüde hakem uzmana gösterilmez; kararınızı insan anotasyonu, model çıktısı "
        "ya da düzeltme ile verin."
    ),
    "locked judge has no valid result": "Hakemin bu belge için geçerli bir önerisi yok.",
}


def _form_error_message(detail: str) -> str:
    for prefix, message in _FORM_ERROR_MESSAGES.items():
        if detail.startswith(prefix):
            return message
    return "Gönderilen karar geçersiz."


REVIEW_TEMPLATE = """
<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{% if position %}Belge {{ position.index }}/{{ position.total }} · {% endif %}DQCheck inceleme</title>
{% macro refcell(refs) %}
{%- for r in refs -%}
<div class="ref"><span class="law">{{ r.kanun_no or "—" }}</span>{% if r.kanun_ad %} <span class="lawname">{{ r.kanun_ad }}</span>{% endif %}
<span class="art">madde {{ r.madde or "—" }}</span>{% if not r.madde %} <span class="tag" title="Satırda madde yok: kanun düzeyinde bir referans sayılır">maddesiz</span>{% endif %}{% if not r.kanun_no %} <span class="tag" title="Satırda kanun numarası yok: yalnız kanun adıyla eşleştirilebilir">numarasız</span>{% endif %}{% if r.source_text %}
<div class="quote" title="Bursiyerin metinden aldığı alıntı; karar vermez, yalnız vurgulanır">“{{ r.source_text }}”</div>{% endif %}</div>
{%- else -%}<span class="muted">—</span>{%- endfor -%}
{% endmacro %}
<style>
:root{--bg:#f3f4f6;--panel:#fff;--line:#dde1e6;--ink:#1d2127;--muted:#646d78;--brand:#1a56db;--brand-soft:#e8effd;
--a:#1d4ed8;--a-bg:#d9e6ff;--b:#b45309;--b-bg:#ffe2c6;--both:#6d28d9;--both-bg:#e6dcfb;--mix:#b42318;--mix-bg:#fde0de;
--ok:#137333;--ok-bg:#dff3e5;--warn:#9a3412;--warn-bg:#fff1e5;--head:64px}
*{box-sizing:border-box}html,body{margin:0}body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--ink);line-height:1.45;font-size:15px}
button{font:inherit}.muted{color:var(--muted)}small{font-size:.78rem}
:focus-visible{outline:3px solid #93b4f5;outline-offset:2px}
/* top bar */
.topbar{position:sticky;top:0;z-index:20;background:var(--panel);border-bottom:1px solid var(--line);height:var(--head)}
.topinner{max-width:1600px;margin:auto;padding:.55rem 1.1rem 0;display:flex;align-items:center;gap:.8rem;flex-wrap:wrap}
.topinner h1{font-size:1.05rem;margin:0;white-space:nowrap}.docid{color:var(--muted);font-size:.78rem}
.bucket{font-size:.74rem;font-weight:700;padding:.18rem .6rem;border-radius:1rem;letter-spacing:.01em}
.bucket-RED{background:#ffe4e2;color:#9f1c17}.bucket-YELLOW{background:#fff1c2;color:#6b4b00}.bucket-GREEN{background:var(--ok-bg);color:var(--ok)}.bucket-QUARANTINE{background:#ece6ff;color:#4b2aa8}
.spacer{flex:1}.durability{color:var(--ok);font-size:.78rem;font-weight:600}.keys{font-size:.76rem;color:var(--muted)}
kbd{font:600 .72rem ui-monospace,monospace;background:#f1f3f5;border:1px solid #cfd5dc;border-bottom-width:2px;border-radius:.3rem;padding:0 .3rem;color:#333}
.progress{position:absolute;left:0;right:0;bottom:-1px;height:3px;background:#e5e8ec}.progress span{display:block;height:100%;background:var(--brand)}
/* layout */
.layout{max-width:1600px;margin:auto;padding:1rem 1.1rem 3rem;display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);gap:1.1rem;align-items:start}
.card{background:var(--panel);border:1px solid var(--line);border-radius:.6rem;padding:1rem 1.1rem;margin:0 0 1rem}
h2{font-size:1rem;margin:0 0 .6rem}.hint{color:var(--muted);font-size:.83rem;margin:.2rem 0 .6rem}
.textpane{position:sticky;top:calc(var(--head) + 1rem);max-height:calc(100vh - var(--head) - 2rem);display:flex;flex-direction:column;margin:0}
.docwrap{position:relative;flex:1;min-height:0;display:flex}.docwrap .doctext{flex:1;min-height:0;padding-right:1.5rem}
.minimap{position:absolute;top:.35rem;bottom:.35rem;right:.3rem;width:.5rem;border-radius:.3rem;background:#eef0f3}.minimap i{position:absolute;left:0;right:0;height:4px;margin-top:-2px;border-radius:2px;cursor:pointer}
.minimap i.k-a{background:var(--a)}.minimap i.k-b{background:var(--b)}.minimap i.k-both{background:var(--both)}.minimap i.k-mixed{background:var(--mix)}.minimap i.k-neutral{background:#9aa1ab}
.nav{display:flex;align-items:center;gap:.4rem;background:#fffbe6;border:1px solid #f1e3a1;border-radius:.4rem;padding:.3rem .5rem;margin:0 0 .5rem;font-size:.84rem}
.nav[hidden]{display:none}.nav button{border:1px solid #d8cf9f;background:#fff;border-radius:.3rem;padding:.1rem .45rem;cursor:pointer}.nav button:disabled{opacity:.4;cursor:default}.nav-label{font-weight:650}.nav-count{min-width:4.5rem;text-align:center;font-variant-numeric:tabular-nums}
.notes-card{border-left:4px solid #d97706}.note{margin:.35rem 0;font-size:.88rem}.note-attention{background:var(--warn-bg);color:#7c2d12;padding:.5rem .7rem;border-radius:.4rem}
.note-info,.note-why{margin-top:.4rem}.note-info summary,.note-why summary{font-size:.84rem;font-weight:600;color:#4b5563}
.summary .spacer,.cmp-tools .spacer{flex:1}.summary{align-items:center}.cmp-tools{display:flex;align-items:center;gap:.35rem;flex-wrap:wrap;font-size:.82rem;margin:-.2rem 0 .6rem}
.tool{border:1px solid #c5ccd4;background:#fff;border-radius:.4rem;padding:.3rem .65rem;font-size:.8rem;cursor:pointer}.tool[aria-pressed=true]{background:#1f2937;color:#fff;border-color:#1f2937}
.tool.primary{background:var(--brand);color:#fff;border-color:var(--brand);font-weight:650}.tool:disabled{opacity:.45;cursor:default}
.link{border:0;background:none;color:var(--brand);padding:0;cursor:pointer;font-size:.82rem}.link:hover{text-decoration:underline}
.tag{display:inline-block;font-size:.68rem;font-weight:650;color:#92400e;background:#fef3c7;border-radius:.25rem;padding:0 .3rem;vertical-align:1px}
.cmp tbody tr:has(input.pick:checked) td{background:#f0faf3}.pickcell{text-align:center}.pick{width:1rem;height:1rem;cursor:pointer}
table.cmp.only-diff tr.same,table.cmp.only-diff tr.excused{display:none}
.guide{border-left:4px solid var(--brand)}.guide h2{margin-bottom:.3rem}.guide p{margin:.2rem 0;font-size:.9rem}
.guide details summary{font-size:.84rem;color:var(--brand);cursor:pointer;margin-top:.4rem}.steps{margin:.4rem 0 0;padding-left:1.2rem;font-size:.88rem}
.warnbox{color:var(--warn);background:var(--warn-bg);padding:.6rem .8rem;border-radius:.4rem;font-size:.88rem}
/* document text and highlights */
.legend{display:flex;flex-wrap:wrap;gap:.35rem .8rem;font-size:.78rem;color:var(--muted);margin:0 0 .5rem}
.legend>span{display:inline-flex;align-items:center;gap:.3rem}.legend>span.zero{opacity:.45}.legend .count{font-weight:700;color:#374151}.legend i{display:inline-block;width:.9rem;height:.9rem;border-radius:.2rem}
.doctext{white-space:pre-wrap;overflow:auto;border:1px solid var(--line);background:#fbfcfd;padding:1rem 1.1rem;border-radius:.45rem;font-size:.95rem;line-height:1.7}
mark.ev{color:inherit;border-radius:.2rem;padding:.05rem .1rem;cursor:pointer;box-decoration-break:clone;-webkit-box-decoration-break:clone}
mark.ev-a{background:var(--a-bg)}mark.ev-b{background:var(--b-bg)}mark.ev-both{background:var(--both-bg)}mark.ev-mixed{background:var(--mix-bg);box-shadow:inset 0 -2px 0 var(--mix)}mark.ev-neutral{background:#eceef1}
mark.ev.active{outline:2px solid var(--ink);outline-offset:1px}mark.ev.current{outline:3px solid var(--brand);background:#fff3a6}
/* comparison */
.summary{display:flex;gap:.5rem;flex-wrap:wrap;margin:.2rem 0 .7rem}.pill{display:inline-block;font-size:.76rem;font-weight:650;padding:.15rem .55rem;border-radius:1rem;border:1px solid transparent;white-space:nowrap}
.pill-diff{background:var(--mix-bg);color:var(--mix)}.pill-same{background:var(--ok-bg);color:var(--ok)}
.pill-same-row{background:var(--both-bg);color:var(--both)}.pill-only_a{background:var(--a-bg);color:var(--a)}.pill-only_b{background:var(--b-bg);color:var(--b)}
.pill-excused{background:#eef0f3;color:#4b5563}.pill-missing{background:#fff;color:var(--mix);border-color:#f3b4ae}.pill-found{background:#fff;color:var(--muted);border-color:var(--line)}
table.cmp{border-collapse:separate;border-spacing:0;width:100%;font-size:.86rem;table-layout:fixed}.cmp col.c-p{width:5%}.cmp col.c-a{width:35%}.cmp col.c-b{width:29%}.cmp col.c-s{width:31%}
.cmp th{position:sticky;top:var(--head);background:#f6f7f9;z-index:1;text-align:left;font-weight:650;font-size:.78rem;color:#3d444d;padding:.5rem .55rem;border-bottom:1px solid var(--line)}
.cmp td{padding:.5rem .55rem;border-bottom:1px solid #edf0f3;vertical-align:top;overflow-wrap:anywhere}
.cmp tbody tr{cursor:pointer}.cmp tbody tr:hover td{background:#f8fafc}
.cmp tr.excused td:first-child{box-shadow:inset 4px 0 0 #b8bec7}.cmp tr.only_a td:first-child{box-shadow:inset 4px 0 0 var(--a)}.cmp tr.only_b td:first-child{box-shadow:inset 4px 0 0 var(--b)}.cmp tr.same td:first-child{box-shadow:inset 4px 0 0 var(--both)}
.cmp tr.active td{background:#fffbe6}
.ref{margin-bottom:.4rem}.ref:last-child{margin-bottom:0}.law{font-weight:700}.lawname{color:#3d444d}.art{display:inline-block;margin-left:.15rem;font-weight:600}
.quote{color:var(--muted);font-size:.76rem;margin-top:.1rem;max-height:3.1em;overflow:hidden}
.notes{font-size:.74rem;color:var(--muted);margin-top:.3rem}.rowlabel{font-weight:650}
.statuscell .pill{margin:0 .2rem .25rem 0}
/* decision */
.decision{border:2px solid var(--brand)}.decision p{margin:.1rem 0 .7rem;font-size:.9rem}
.decision-grid{display:grid;grid-template-columns:1fr 1fr;gap:.7rem}
.decision button{min-height:64px;border:1px solid var(--brand);border-radius:.55rem;background:var(--brand-soft);color:#123a78;font-size:1rem;font-weight:700;padding:.6rem;cursor:pointer}
.decision button:hover{background:#d8e5fc}.decision button span{display:block;font-size:.76rem;font-weight:450;margin-top:.15rem;color:#3b4f73}
details.secondary,details.technical{background:var(--panel);border:1px solid var(--line);border-radius:.6rem;padding:.75rem 1.1rem;margin:0 0 1rem}
summary{cursor:pointer;font-weight:650}.secondary[open]>summary,.technical[open]>summary{margin-bottom:.7rem}
.defer-row{display:flex;gap:.6rem;align-items:end;flex-wrap:wrap}.defer-row label{flex:1;min-width:220px;font-size:.88rem}.defer-row input{width:100%;padding:.5rem;margin-top:.25rem;border:1px solid #c5ccd4;border-radius:.4rem}
.defer-row button,.toolbar button{padding:.45rem .75rem;border:1px solid #c5ccd4;background:#fff;border-radius:.4rem;cursor:pointer}.defer-row button:hover,.toolbar button:hover{background:#f3f5f7}
.editor-wrap{overflow:auto}.editor{border-collapse:collapse;width:100%}.editor th{font-size:.76rem;text-align:left;color:#3d444d;padding:.3rem}.editor td{padding:.2rem}
.editor input{width:100%;min-width:70px;padding:.35rem;border:1px solid #c5ccd4;border-radius:.3rem}.editor input[data-field=kanun_ad]{min-width:170px}.editor input[data-field=kanun_no]{max-width:90px}.editor input[data-field=madde]{min-width:120px}.editor td:last-child{width:1%}.editor .source-input{min-width:220px}
.editor button{border:1px solid #e3b4ae;background:#fff;color:var(--mix);border-radius:.3rem;padding:.3rem .5rem;cursor:pointer}.editor button.expand-range{border-color:#b9cdf6;color:var(--brand);white-space:nowrap}
.toolbar{display:flex;gap:.4rem;flex-wrap:wrap;margin:.6rem 0}
.save-revision{background:var(--brand);color:white;border:0;border-radius:.45rem;font-weight:700;padding:.6rem .95rem;cursor:pointer}
pre{white-space:pre-wrap;border:1px solid var(--line);padding:.8rem;border-radius:.4rem;font-size:.78rem;background:#fbfcfd}
textarea{width:100%;font:.8rem ui-monospace,monospace}
@media(max-width:1100px){.layout{grid-template-columns:1fr}.textpane{position:static;height:auto}.docwrap .doctext{max-height:60vh}.cmp th{top:0}}
@media(max-width:640px){:root{--head:auto}.topbar{position:relative;height:auto;padding-bottom:.6rem}.decision-grid{grid-template-columns:1fr}.keys{display:none}
.layout{padding:.7rem .6rem 2rem}.card{padding:.85rem .8rem}.pill{white-space:normal}.cmp td,.cmp th{padding:.4rem .35rem}.cmp col.c-p{width:9%}.cmp col.c-a{width:33%}.cmp col.c-b{width:29%}.cmp col.c-s{width:29%}}
</style>
<header class="topbar"><div class="topinner">
  <h1>{% if position %}Belge {{ position.index }} / {{ position.total }}{% else %}Belge inceleme{% endif %}</h1>
  <span class="bucket bucket-{{ doc.router_bucket }}">{{ {"RED": "Uyuşmazlık", "GREEN": "Denetim örneği", "QUARANTINE": "Teknik inceleme", "YELLOW": "Kısmi uyuşma"}.get(doc.router_bucket, doc.router_bucket) }}</span>
  <span class="docid">Kayıt: {{ doc.internal_doc_id }}</span>
  <span class="spacer"></span>
  <span class="keys"><kbd>H</kbd> İnsan doğru · <kbd>M</kbd> Model doğru{% if doc.judge_suggestion %} · <kbd>J</kbd> Hakem{% endif %} · <kbd>D</kbd> Ertele · <kbd>Esc</kbd> vurguyu temizle</span>
  <span class="durability">Kayıt koruması: Güncel · {{ durability.review_count }} karar yedeklendi</span>
</div>{% if position %}<div class="progress" aria-label="İnceleme ilerlemesi"><span style="width:{{ (100 * position.index / position.total)|round(1) }}%"></span></div>{% endif %}</header>

<div class="layout">
<section class="card textpane" aria-label="Belge metni">
  <h2>1. Belge metnini kontrol et</h2>
  <div class="legend"><span data-kind="both" class="{{ 'zero' if not legend_counts.both }}"><i style="background:var(--both-bg)"></i><span>İki tarafta da <b class="count">{{ legend_counts.both }}</b></span></span><span data-kind="a" class="{{ 'zero' if not legend_counts.a }}"><i style="background:var(--a-bg)"></i><span>Yalnız insan anotasyonunda <b class="count">{{ legend_counts.a }}</b></span></span><span data-kind="b" class="{{ 'zero' if not legend_counts.b }}"><i style="background:var(--b-bg)"></i><span>Yalnız modelde <b class="count">{{ legend_counts.b }}</b></span></span><span data-kind="mixed" class="{{ 'zero' if not legend_counts.mixed }}"><i style="background:var(--mix-bg)"></i><span>Aynı yerde farklı referanslar <b class="count">{{ legend_counts.mixed }}</b></span></span><span data-kind="neutral" class="{{ 'zero' if not legend_counts.neutral }}"><i style="background:#eceef1"></i><span>Fark sayılmaz <b class="count">{{ legend_counts.neutral }}</b></span></span></div>
  <div class="nav" id="nav" hidden><span class="nav-label" id="nav-label"></span><span class="spacer"></span><button type="button" id="nav-prev" title="Önceki geçiş">◀</button><span class="nav-count" id="nav-count"></span><button type="button" id="nav-next" title="Sonraki geçiş">▶</button><button type="button" id="nav-close" title="Vurguyu temizle (Esc)">✕</button></div>
  <div class="docwrap"><div class="doctext" id="doctext">{% for seg in segments %}{% if seg.candidates %}<mark class="ev ev-{{ seg.kind }}" data-rows="{{ seg.candidates|join(' ') }}" title="{{ seg.title }}">{{ seg.text }}</mark>{% else %}{{ seg.text }}{% endif %}{% endfor %}</div><div class="minimap" id="minimap" aria-hidden="true"></div></div>
  <p class="hint" style="margin:.5rem 0 0">Vurgular, karşılaştırmadaki referansların metinde anıldığı yerlerdir (bulucu); doğruluk anlamına gelmez. Tablodaki bir satıra tıklayınca o referansın geçtiği yerlere gidilir.</p>
</section>

<div class="side">
<section class="card guide"><h2>{{ doc.guidance.title }}</h2><p>{{ doc.guidance.explanation }}</p>
  <details><summary>Nasıl karar verilir?</summary><ol class="steps"><li>Belge metnindeki kanun ve madde atıflarını kontrol edin.</li><li>İnsan anotasyonu ile model çıktısını metinle karşılaştırın.</li><li>Doğru olanı seçin; ikisi de yanlışsa düzeltin.</li></ol></details>
</section>
{% if notes %}<section class="card notes-card"><h2>Bursiyer listesi notları</h2>
  {% for n in notes if n.level == 'attention' %}<p class="note note-attention">{{ n.text }}</p>{% endfor %}
  {% set infos = notes|selectattr('level', 'equalto', 'info')|list %}{% if infos %}<details class="note-info"><summary>{{ infos|length }} bilgi notu — kararınızı etkilemez</summary>{% for n in infos %}<p class="note">{{ n.text }}</p>{% endfor %}</details>{% endif %}
  <details class="note-why"><summary>Bu notlar neden var?</summary><p class="hint">Bursiyer etiketleri anotasyon platformundan aktarılırken değeri olmayan (boş) alanlar boş metne çevrilir ve kayda geçirilir. Notlar, bu boşlukların kaynaktan geldiğini ve karşılaştırmada ne anlama geldiğini söyler; sizin bir şey düzeltmeniz gerekmez.</p></details>
</section>{% endif %}

<section class="card"><h2>2. İnsan anotasyonu ile modeli karşılaştır</h2>
<p class="hint">Sol sütun insan anotasyonudur; sağ sütun modelin ({{ doc.generation }}) çıktısıdır.</p>
<div class="summary"><span class="pill pill-diff">{{ diff_summary.difference_count }} farklı</span><span class="pill pill-same">{{ diff_summary.same_count }} aynı referans grubu</span>
  <span class="spacer"></span><button type="button" class="tool" id="only-diff" aria-pressed="false">Yalnız farkları göster</button></div>
<div class="cmp-tools"><span class="muted">Seç:</span> <button type="button" class="link" data-pick="a">insanınkiler</button> · <button type="button" class="link" data-pick="b">modelinkiler</button> · <button type="button" class="link" data-pick="none">temizle</button>
  <span class="spacer"></span><button type="button" class="tool primary" id="compose" disabled>İşaretlenenlerle düzelt</button></div>
<table class="cmp"><colgroup><col class="c-p"><col class="c-a"><col class="c-b"><col class="c-s"></colgroup><thead><tr><th title="Doğru olan satırları işaretleyip düzeltme başlatabilirsiniz">✓</th><th>İnsan anotasyonu<br><small class="muted">Tamamlayan: {{ doc.human_attribution.display_name }}{% if doc.human_attribution.last_editor_name and doc.human_attribution.last_editor_name != doc.human_attribution.display_name %} · Son düzenleyen: {{ doc.human_attribution.last_editor_name }}{% endif %}</small></th><th>Model çıktısı<br><small class="muted">{{ doc.generation }} · {{ doc.model_name }}</small></th><th>Sonuç</th></tr></thead><tbody>
{% for row in diff %}<tr class="{{ row.display }}" data-row="{{ row.index }}" tabindex="0" title="{{ row.label }} — metindeki yerlerine gitmek için tıklayın">
  <td class="pickcell"><input type="checkbox" class="pick" data-row="{{ row.index }}" data-a="{{ 1 if row.a else 0 }}" data-b="{{ 1 if row.b else 0 }}" data-mentions="{{ row.mentions }}" data-label="{{ row.label }}" aria-label="{{ row.label }} doğru"></td><td>{{ refcell(row.a) }}</td><td>{{ refcell(row.b) }}</td>
  <td class="statuscell">{% if row.display == 'excused' %}<span class="pill pill-excused">Fark sayılmaz</span>{% else %}<span class="pill pill-{{ 'same-row' if row.display == 'same' else row.display }}">{{ row.status_label.split(' · ')[0] }}</span>{% endif %}
    {% if row.mentions %}<span class="pill pill-found">metinde {{ row.mentions }} yerde</span>{% else %}<span class="pill pill-missing">metinde bulunamadı</span>{% endif %}
    {% if row.duplicates.a or row.duplicates.b or row.subsumed.a or row.subsumed.b %}<div class="notes">{{ row.status_label.split(' · ')[1:]|join(' · ') }}</div>{% endif %}</td>
</tr>{% else %}<tr><td colspan="4" class="muted">İki adayda da referans yok.</td></tr>{% endfor %}
</tbody></table></section>

{% if doc.judge_suggestion %}<section class="card"><h2>Hakemin önerisi</h2>
<p class="hint">Yerel dil modeli hakemin bu belge için önerdiği referans listesi. Doğruluk kanıtı değildir; belge metnine göre kontrol edin.</p>
{{ refcell(doc.judge_suggestion.final_references) }}</section>{% endif %}

<form method="post" action="{{ url_for('submit_review', internal_doc_id=doc.internal_doc_id) }}">
  <input type="hidden" name="csrf_token" value="{{ csrf_token }}"><input type="hidden" name="row_version" value="{{ doc.row_version }}"><input type="hidden" name="opened_at" value="{{ opened_at }}">
  <section class="card decision"><h2>3. Karar ver</h2><p>Belgedeki tüm hedef kanun/madde referanslarını doğru veren adayı seçin.</p>
    <div class="decision-grid"><button name="action" value="accept_human" accesskey="h">İnsan anotasyonu doğru<span>{{ doc.human_attribution.display_name }} · Kısayol: H</span></button>
    <button name="action" value="accept_model" accesskey="m">Model çıktısı doğru<span>{{ doc.generation }} · Kısayol: M</span></button>
    {% if doc.judge_suggestion %}<button name="action" value="judge_override" accesskey="j">Hakemin önerisi doğru<span>Yerel hakem · Kısayol: J</span></button>{% endif %}</div>
  </section>
  <details class="secondary"><summary>Karar veremiyorum — bu belgeyi ertele</summary><div class="defer-row">
    <label>Erteleme gerekçesi<input name="reason" placeholder="Örn. metin eksik veya hukuki yorum gerekli"></label><button name="action" value="defer">Gerekçeyle ertele</button></div>
    <p class="warnbox" id="defer-msg" hidden>Ertelemek için bir gerekçe yazın.</p>
    <p class="hint">Ertelenen belge çözülmeden parti yayımlanamaz.</p>
  </details>
  <details class="secondary" id="revision-panel"><summary>İkisi de tam doğru değil — referansları düzelt</summary>
    <p class="hint">Başlangıç olarak insan anotasyonunu veya modeli seçin; yalnız yanlış satırları değiştirin. Aynı referansı iki kez yazmak gerekmez.</p>
    <p class="note note-attention" id="compose-warn" hidden></p>
    <div class="toolbar"><button type="button" id="fill-a">İnsan anotasyonundan başla</button><button type="button" id="fill-b">Model çıktısından başla</button>
    {% if doc.judge_suggestion %}<button type="button" id="fill-judge">Yardımcı görüşten başla</button>{% endif %}<button type="button" id="add-row">+ Referans ekle</button></div>
    <div class="editor-wrap"><table class="editor"><thead><tr><th>Kanun no</th><th>Kanun adı</th><th>Madde</th><th>Metindeki ifade (isteğe bağlı)</th><th></th></tr></thead><tbody id="ref-rows"></tbody></table></div>
    <details class="technical"><summary>Teknik JSON alanı (yalnız gerekirse)</summary><textarea name="references_json" rows="8" cols="80"></textarea></details>
    <p><button class="save-revision" name="action" value="revise">Düzeltmeyi kaydet ve devam et</button></p>
  </details>
</form>

<details class="technical"><summary>Teknik ayrıntılar</summary>
  <p class="hint">Aktif policy: {{ doc.reference_policy.policy_id }} · İki adaydan çıkarılan boilerplate: {{ doc.reference_policy.removed_reference_count }}</p>
  <details><summary>Ham insan/model JSON</summary><div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem"><div><strong>İnsan anotasyonu</strong><pre>{{ doc.candidate_a|tojson(indent=2) }}</pre></div><div><strong>Model çıktısı ({{ doc.generation }})</strong><pre>{{ doc.candidate_b|tojson(indent=2) }}</pre></div></div></details>
  {% if doc.judge_suggestion %}<details><summary>Otomatik yardımcı görüş</summary><pre>{{ doc.judge_suggestion|tojson(indent=2) }}</pre></details>{% endif %}
</details>
</div>
</div>

<script id="cand" type="application/json">{{ {"a": doc.candidate_a, "b": doc.candidate_b, "judge": doc.judge_suggestion, "rows": rows_payload}|tojson }}</script>
<script>
(function(){
  var DATA = JSON.parse(document.getElementById('cand').textContent);
  var FIELDS = ['kanun_no','kanun_ad','madde','source_text'];
  var rows = document.getElementById('ref-rows');
  var textarea = document.querySelector('textarea[name=references_json]');
  function collect(){
    var out=[]; rows.querySelectorAll('tr').forEach(function(tr){
      var o={}, any=false;
      tr.querySelectorAll('input').forEach(function(i){o[i.dataset.field]=i.value; if(i.value.trim())any=true;});
      if(any) out.push(o);
    }); return out;
  }
  // Whichever the expert edited last -- the table or the technical JSON --
  // is what gets saved; the table must not silently overwrite typed JSON.
  var typedJson = false;
  textarea.addEventListener('input', function(){ typedJson = true; });
  function sync(){ if(!typedJson) textarea.value = JSON.stringify(collect()); }
  function tableEdited(){ typedJson = false; sync(); }
  function addRow(ref){
    ref = ref||{}; var tr=document.createElement('tr');
    FIELDS.forEach(function(f){
      var td=document.createElement('td'), inp=document.createElement('input');
      inp.dataset.field=f; inp.value=ref[f]||''; inp.size={kanun_no:7,kanun_ad:24,madde:14,source_text:34}[f];
      if(f==='source_text') inp.className='source-input';
      inp.setAttribute('aria-label', {kanun_no:'Kanun no',kanun_ad:'Kanun adı',madde:'Madde',source_text:'Metindeki ifade'}[f]);
      inp.addEventListener('input', tableEdited); td.appendChild(inp); tr.appendChild(td);
    });
    var td=document.createElement('td'), del=document.createElement('button');
    del.type='button'; del.textContent='sil'; del.title='Bu satırı sil';
    del.addEventListener('click', function(){ tr.remove(); tableEdited(); });
    var articles = rangeArticles(ref.madde||'');
    if(articles){
      // "473 ila 475" means 473, 474 and 475, each its own row (user, 7 Oct 2026).
      var expand=document.createElement('button'); expand.type='button'; expand.className='expand-range';
      expand.textContent='aralığı aç (' + articles.length + ')'; expand.title='Aralığı her madde ayrı satır olacak biçimde aç';
      expand.addEventListener('click', function(){
        var current = {}; tr.querySelectorAll('input').forEach(function(i){ current[i.dataset.field] = i.value; });
        var anchor = tr.nextSibling;
        articles.forEach(function(madde){
          addRow({kanun_no: current.kanun_no, kanun_ad: current.kanun_ad, madde: madde, source_text: current.source_text});
          if(anchor){ rows.insertBefore(rows.lastChild, anchor); }
        });
        tr.remove(); tableEdited();
      });
      td.appendChild(expand); td.appendChild(document.createTextNode(' '));
    }
    td.appendChild(del); tr.appendChild(td); rows.appendChild(tr);
  }
  // The articles an inclusive range names: "N ila M" (also "N-M") and "N ila mükerrer M"
  // (N..M and mükerrer M). Anything else, or a range wider than 300, is left as written.
  function rangeArticles(madde){
    var m = String(madde).trim().match(/^(\\d+)\\s*(?:ila|ilâ|-|–|—)\\s*(mükerrer\\s+)?(\\d+)$/i);
    if(!m) return null;
    var a = Number(m[1]), b = Number(m[3]);
    if(!(a < b || (m[2] && a <= b)) || b - a > 300) return null;
    var out = []; for(var n = a; n <= b; n++){ out.push(String(n)); }
    if(m[2]){ out.push('mükerrer ' + b); }
    return out;
  }
  function fill(list){ rows.innerHTML=''; (list||[]).forEach(addRow); if(!rows.children.length) addRow(); tableEdited(); }
  function clearComposeWarn(){ var w = document.getElementById('compose-warn'); if(w){ w.hidden = true; } }
  document.getElementById('fill-a').addEventListener('click', function(){ clearComposeWarn(); fill(DATA.a); });
  document.getElementById('fill-b').addEventListener('click', function(){ clearComposeWarn(); fill(DATA.b); });
  var jb=document.getElementById('fill-judge');
  if(jb && DATA.judge){ jb.addEventListener('click', function(){ fill(DATA.judge.final_references||[]); }); }
  document.getElementById('add-row').addEventListener('click', function(){ addRow(); });
  document.querySelector('button[value=revise]').addEventListener('click', sync);
  var reasonInput = document.querySelector('input[name=reason]');
  document.querySelector('button[value=defer]').addEventListener('click', function(e){
    if(!reasonInput.value.trim()){
      e.preventDefault(); document.getElementById('defer-msg').hidden=false; reasonInput.focus();
    }
  });
  // One decision per page: a double click or a repeated shortcut must not send
  // a second, stale-version submission after the first one saved.
  var form = document.querySelector('form[action]'), sent = false;
  form.addEventListener('submit', function(e){ if(sent){ e.preventDefault(); } sent = true; });
  window.addEventListener('pageshow', function(){ sent = false; });
  fill(DATA.a);

  // Linking the comparison and the text: a row shows where its reference is
  // cited, the navigator steps through those places, a highlight shows its rows.
  var doctext = document.getElementById('doctext');
  var marks = Array.prototype.slice.call(doctext.querySelectorAll('mark.ev'));
  var tableRows = Array.prototype.slice.call(document.querySelectorAll('table.cmp tbody tr[data-row]'));
  var nav = document.getElementById('nav'), navLabel = document.getElementById('nav-label'), navCount = document.getElementById('nav-count');
  var navPrev = document.getElementById('nav-prev'), navNext = document.getElementById('nav-next');
  var active = null, cursor = -1;
  function marksOf(index){ return marks.filter(function(m){ return (' '+m.dataset.rows+' ').indexOf(' '+index+' ') >= 0; }); }
  function rowOf(index){ return document.querySelector('table.cmp tr[data-row="'+index+'"]'); }
  function labelOf(index){ var r = rowOf(index); return r ? (r.getAttribute('title')||'').split(' — ')[0] : ''; }
  function clearActive(){
    marks.forEach(function(m){ m.classList.remove('active','current'); });
    tableRows.forEach(function(r){ r.classList.remove('active'); });
    active = null; cursor = -1; nav.hidden = true;
  }
  function scrollIntoText(mark){
    var top = mark.offsetTop - doctext.offsetTop - doctext.clientHeight/3;
    doctext.scrollTo({top: Math.max(0, top), behavior: 'smooth'});
  }
  function showNav(index, own){
    nav.hidden = false; navLabel.textContent = labelOf(index);
    navPrev.disabled = navNext.disabled = own.length < 2;
    navCount.textContent = own.length ? (cursor + 1) + ' / ' + own.length : 'metinde bulunamadı';
  }
  function goTo(index, step){
    var own = marksOf(index);
    if(active !== String(index)){
      clearActive(); active = String(index); cursor = -1;
      own.forEach(function(m){ m.classList.add('active'); });
      var row = rowOf(index); if(row){ row.classList.add('active'); }
    }
    if(own.length){
      own.forEach(function(m){ m.classList.remove('current'); });
      cursor = (cursor + step + own.length) % own.length;
      own[cursor].classList.add('current');
      scrollIntoText(own[cursor]);
    }
    showNav(index, own);
  }
  tableRows.forEach(function(r){
    r.addEventListener('click', function(e){
      if(e.target.closest('.pickcell')) return;
      goTo(r.dataset.row, 1);
    });
    r.addEventListener('keydown', function(e){ if(e.key === 'Enter' || e.key === ' '){ e.preventDefault(); goTo(r.dataset.row, 1); } });
  });
  navNext.addEventListener('click', function(){ if(active !== null) goTo(active, 1); });
  navPrev.addEventListener('click', function(){ if(active !== null) goTo(active, -1); });
  document.getElementById('nav-close').addEventListener('click', clearActive);
  marks.forEach(function(m){
    m.addEventListener('click', function(){
      var cited = m.dataset.rows.split(' '), first = cited[0], own = marksOf(first);
      clearActive(); active = first; cursor = own.indexOf(m);
      own.forEach(function(x){ x.classList.add('active'); });
      m.classList.add('current');
      cited.forEach(function(index){ var row = rowOf(index); if(row){ row.classList.add('active'); } });
      showNav(first, own);
      var row = rowOf(first); if(row){ row.scrollIntoView({block: 'nearest', behavior: 'smooth'}); }
    });
  });

  // The scrollbar shows where the highlights are; a tick takes you there.
  var minimap = document.getElementById('minimap');
  function drawMinimap(){
    minimap.innerHTML = '';
    var height = doctext.scrollHeight; if(!height) return;
    marks.forEach(function(m){
      var tick = document.createElement('i');
      var kind = (m.className.match(/ev-(both|a|b|mixed|neutral)/)||[])[1] || 'both';
      tick.className = 'k-' + kind;
      tick.style.top = (100 * (m.offsetTop - doctext.offsetTop) / height).toFixed(2) + '%';
      tick.title = m.getAttribute('title') || '';
      tick.addEventListener('click', function(){ m.click(); scrollIntoText(m); });
      minimap.appendChild(tick);
    });
  }
  drawMinimap(); window.addEventListener('resize', drawMinimap);

  // Only the differences.
  var onlyDiff = document.getElementById('only-diff'), cmpTable = document.querySelector('table.cmp');
  onlyDiff.addEventListener('click', function(){
    var on = !cmpTable.classList.contains('only-diff');
    cmpTable.classList.toggle('only-diff', on);
    onlyDiff.setAttribute('aria-pressed', on ? 'true' : 'false');
    onlyDiff.textContent = on ? 'Tümünü göster' : 'Yalnız farkları göster';
  });

  // Compose a revision from the table: tick the rows that are right, then edit
  // and save them in the revision panel (saving stays an explicit click).
  var picks = Array.prototype.slice.call(document.querySelectorAll('input.pick'));
  var compose = document.getElementById('compose');
  function updateCompose(){
    var n = picks.filter(function(p){ return p.checked; }).length;
    compose.disabled = n === 0;
    compose.textContent = n ? 'İşaretlenenlerle düzelt (' + n + ')' : 'İşaretlenenlerle düzelt';
  }
  picks.forEach(function(p){ p.addEventListener('change', updateCompose); p.addEventListener('click', function(e){ e.stopPropagation(); }); });
  document.querySelectorAll('[data-pick]').forEach(function(b){
    b.addEventListener('click', function(){
      var side = b.dataset.pick;
      picks.forEach(function(p){ p.checked = side === 'none' ? false : p.dataset[side] === '1'; });
      updateCompose();
    });
  });
  compose.addEventListener('click', function(){
    var chosen = picks.filter(function(p){ return p.checked; }).map(function(p){
      var row = DATA.rows[Number(p.dataset.row)];
      return (row.a && row.a.length ? row.a[0] : row.b[0]);
    });
    if(!chosen.length) return;
    var missing = picks.filter(function(p){ return p.checked && p.dataset.mentions === '0'; }).map(function(p){ return p.dataset.label; });
    var warn = document.getElementById('compose-warn');
    warn.hidden = !missing.length;
    warn.textContent = missing.length ? 'Dikkat: seçtiğiniz ' + (missing.length > 1 ? missing.length + ' referans' : 'referans') +
      ' metinde bulunamadı (' + missing.join('; ') + '). Belgede gerçekten anıldığından emin olun.' : '';
    fill(chosen);
    var panel = document.getElementById('revision-panel');
    panel.open = true; panel.scrollIntoView({block: 'start', behavior: 'smooth'});
  });

  document.addEventListener('keydown', function(e){
    var t=(e.target.tagName||''); if(t==='INPUT'||t==='TEXTAREA'||t==='SELECT') return;
    if(e.key === 'Escape'){ clearActive(); return; }
    // Cmd+H / Ctrl+M are the system's, and a letter typed while the revision
    // panel is open was meant for a cell: neither may submit a decision.
    if(e.metaKey||e.ctrlKey||e.altKey||e.repeat) return;
    if(document.getElementById('revision-panel').open) return;
    var k=e.key.toLowerCase();
    if(k==='h'||k==='m'||k==='j'){
      var val = {h:'accept_human', m:'accept_model', j:'judge_override'}[k];
      var btn=document.querySelector('button[value='+val+']');
      if(btn){ e.preventDefault(); btn.click(); }
    } else if(k==='d'){
      var reason=document.querySelector('input[name=reason]'); if(reason){ e.preventDefault(); reason.closest('details').open=true; reason.focus(); }
    }
  });
})();
</script>
"""


_DIFF_STATUS_LABELS = {
    "same": "Aynı",
    "only_a": "Yalnız insan anotasyonunda",
    "only_b": "Yalnız model çıktısında",
}


def ab_diff(
    candidate_a: list[dict[str, Any]], candidate_b: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Align human (A) and model (B) references by law-article identity.

    A reference is a law and an article of it (1 Oct 2026), so a group is one
    provision and its status is all there is to say: on both sides (``same``),
    or on one only. The annotator's quote is shown (highlighting), never
    compared: it decides nothing.

    Pairing normalises first; display does not. ``core_identity`` reads its
    fields as given, and every other consumer in the pipeline -- the router,
    the loop metrics, the judge metrics, the reference audit -- hands it rows
    that ``compact_references`` has already normalised. This function was the
    one place passing raw references, so an annotator's "Ek 5" and the model's
    "ek 5" keyed differently and the expert was shown "only in the annotator"
    beside "only in the model" for a reference both sides agreed on. Measured
    on batch M01: 48 such phantom rows across 19 of 100 documents, five of them
    GREEN -- documents the router itself calls identical.

    What the expert reads stays exactly as written, though. The groups hold the
    original references and only the keys are normalised -- showing "ek 5"
    where the annotator typed "Ek 5" would misreport the record being audited.
    A reference written twice ("Geçici 21" and "geçici 21", R01 d000081) is one
    reference; the repetition is flagged, never hidden -- both rows are shown.

    Presence follows the router's own rule (`compact_references`): a law-only
    row beside an article of the same law adds no claim, so it does not make a
    side differ -- the screen must not contradict a GREEN bucket -- but it stays
    visible and flagged, so the expert can audit what was written.
    """

    # The status of a row is the project's one comparison's answer (2 Oct 2026,
    # `reference_matching`): matched, missed, extra -- or a law-only row beside an
    # article of its law in its own list, which is extra, not a difference.
    canonical_a = [canonical_reference(reference) for reference in candidate_a]
    canonical_b = [canonical_reference(reference) for reference in candidate_b]
    context = DocumentContext(canonical_a, canonical_b)
    comparison = compare_reference_lists(candidate_a, candidate_b)
    only_in_a = set(comparison.missed_identities)
    only_in_b = set(comparison.extra_identities)
    excused = {context.identity(row) for row in (*comparison.excused_reference, *comparison.excused_candidate)}

    order: list[tuple[str, str, str, str]] = []
    groups: dict[tuple[str, str, str, str], dict[str, list[dict[str, Any]]]] = {}
    for label, references, canonical in (("a", candidate_a, canonical_a), ("b", candidate_b, canonical_b)):
        for reference, row in zip(references, canonical, strict=True):
            key = context.identity(row)
            if key not in groups:
                groups[key] = {"a": [], "b": []}
                order.append(key)
            groups[key][label].append(reference)

    rows: list[dict[str, Any]] = []
    for key in order:
        a_refs = groups[key]["a"]
        b_refs = groups[key]["b"]
        duplicates = {"a": max(0, len(a_refs) - 1), "b": max(0, len(b_refs) - 1)}
        subsumed = {
            "a": len(a_refs) if a_refs and key in excused else 0,
            "b": len(b_refs) if b_refs and key in excused else 0,
        }
        if key[0] == "unknown":
            # A row naming no law: no reference on the human side, a wrong row on the model's.
            status = "same" if a_refs and b_refs else ("only_a" if a_refs else "only_b")
        elif key in only_in_a:
            status = "only_a"
        elif key in only_in_b:
            status = "only_b"
        else:
            status = "same"
        sample = (a_refs or b_refs)[0]
        rows.append(
            {
                "core": {
                    "kanun_no": sample["kanun_no"],
                    "kanun_ad": sample["kanun_ad"],
                    "madde": sample["madde"],
                },
                "status": status,
                "status_label": _DIFF_STATUS_LABELS[status]
                + "".join(
                    f" · {side} aynı referans {duplicates[key] + 1} kez yazılmış"
                    for key, side in (("a", "insan anotasyonunda"), ("b", "model çıktısında"))
                    if duplicates[key]
                )
                + "".join(
                    f" · {side} kanunun bir maddesi de yazıldığı için ayrıca sayılmaz"
                    for key, side in (("a", "insan anotasyonunda"), ("b", "model çıktısında"))
                    if subsumed[key]
                ),
                "duplicates": duplicates,
                "subsumed": subsumed,
                "a": a_refs,
                "b": b_refs,
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
    # The judge's pilot exam comes first: `judge-lock` needs every pilot
    # document finalized, and only after the lock can the judge help with the
    # rest. A GREEN pilot document is not necessarily in the GREEN audit
    # sample, so without this it would never reach the expert at all.
    # The GT750 loop has no judge pilot (user, 29 Sep 2026): its judge is fixed
    # in advance and never shown, so no document is reviewed for the judge's sake.
    pilot_path = config.public_root / "batches" / batch_id / "judge_pilot_selection.json"
    pilot_ids = (
        {
            str(value)
            for value in json.loads(pilot_path.read_text(encoding="utf-8"))["internal_doc_ids"]
        }
        if pilot_path.exists() and not has_loop_settings(config)
        else set()
    )
    required: set[str] = set(audit_ids) | pilot_ids
    for document in documents:
        # A QUARANTINE document (the model failed on it, e.g. R01 d000112 on
        # 3 Oct 2026) has only the annotator's list: nothing agreed with it, so the
        # expert reviews it like a RED one -- the screen already explains it as
        # "Teknik inceleme gerekiyor". Left out, the batch never had a truth for it.
        if document["router_bucket"] in {"RED", "YELLOW", "QUARANTINE"}:
            required.add(document["internal_doc_id"])
        elif escalated and document["router_bucket"] == "GREEN":
            required.add(document["internal_doc_id"])
    priority: dict[str, int] = {}
    for document in documents:
        doc_id = document["internal_doc_id"]
        if doc_id in pilot_ids:
            priority[doc_id] = 0
        elif document["router_bucket"] in {"RED", "QUARANTINE"}:
            priority[doc_id] = 1
        elif document["router_bucket"] == "YELLOW":
            priority[doc_id] = 2
        elif doc_id in audit_ids:
            priority[doc_id] = 3
        elif escalated and document["router_bucket"] == "GREEN":
            priority[doc_id] = 4
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
    """The judge the expert may see and take as the truth -- None when there is
    none, or when it is the loop's fixed judge, which is never shown."""
    if not judge_visible_to_expert(config, batch_id):
        return None
    payload = judge_lock_payload(config, batch_id)
    return str(payload["model"]) if payload else None


def _document_for_review(
    *,
    config: AppConfig,
    store: Store,
    batch_id: str,
    internal_doc_id: str,
    imported_attribution: dict[str, Any] | None = None,
    generation: str = "G0",
) -> dict[str, Any]:
    document = store.get_document(batch_id, internal_doc_id)
    review = store.get_review(batch_id, internal_doc_id)
    prediction = store.get_prediction(batch_id, internal_doc_id, generation)
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
        # Quote warnings would tell the expert the annotator's quote is off --
        # a signal about a field that decides nothing, so it is not shown.
        "warnings": [
            warning
            for warning in json.loads(document["warnings_json"])
            if not str(warning.get("code", "")).startswith("source_text")
        ],
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        # The annotator's list as prepared (before the 213/413 policy): the list a
        # preparation warning's `reference_index` counts in.
        "human_prepared": json.loads(document["human_references_json"]),
        "candidate_mapping": mapping,
        "human_attribution": human_attribution(metadata, imported_attribution),
        "model_name": config.model.model_id.rsplit("/", 1)[-1],
        "generation": generation,
        "judge_suggestion": judge_suggestion,
        "review_status": review["status"],
        "row_version": review["row_version"],
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


def row_highlights(text: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    """Where each row of the comparison (`ab_diff`) is cited in `text`.

    Every row -- the annotator's, the model's, or both sides' -- is placed by
    the locator at each of its mentions (`mention_offsets`), and each verbatim
    quote the annotator wrote for it is marked as well (all occurrences). The
    annotator's quote still decides nothing: it only adds a highlight. Returns
    spans labelled with the row's index (for `evidence_segments`) and, per row,
    how many places in the text cite it (0: cited nowhere, which the screen says).
    """
    spans: list[dict[str, Any]] = []
    counts: list[int] = []
    for index, row in enumerate(rows):
        placed: set[tuple[int, int]] = set()
        for reference in (*row["a"], *row["b"]):
            bare = {field: str(reference.get(field, "") or "") for field in ("kanun_no", "kanun_ad", "madde")}
            try:
                placed.update(mention_offsets(text, {**bare, "source_text": ""}))
            except Exception:  # a highlight is a help: a reference the locator chokes on stays unmarked
                continue
        located = len(placed)
        for reference in row["a"]:
            quote = str(reference.get("source_text") or "").strip()
            start = text.find(quote) if quote else -1
            while start >= 0:
                placed.add((start, start + len(quote)))
                start = text.find(quote, start + 1)
        counts.append(located)
        spans.extend({"start": start, "end": end, "candidate": str(index)} for start, end in sorted(placed))
    return sorted(spans, key=lambda item: (item["start"], item["end"], item["candidate"])), counts


# What an empty field of the annotator's row means for the comparison (6 Oct 2026:
# the screen showed preparation warnings as raw JSON). Preparation turns a field
# the annotation export gave as null, or left out, into "" and records it.
_EMPTY_FIELD_NOTES = {
    "kanun_ad": (
        "info",
        "kanun adı platformdan boş geldi. Karşılaştırma kanun numarası ve maddeyle yapıldığı için kararınızı etkilemez.",
    ),
    "madde": (
        "attention",
        "madde platformdan boş geldi: bu satır maddesiz, yalnız kanun düzeyinde bir referans sayılır (aynı kanunun bir "
        "maddesi de yazılmışsa fark sayılmaz). Bursiyer maddeyi yazmayı unutmuş olabilir; metinle karşılaştırın.",
    ),
    "kanun_no": (
        "attention",
        "kanun numarası platformdan boş geldi: bu satır yalnız kanun adıyla eşleştirilebilir; adı da yoksa hiçbir "
        "referansla eşleşmez.",
    ),
}


def _ordinals(indices: list[int]) -> str:
    numbers = [f"{index + 1}." for index in indices]
    if len(numbers) == 1:
        return f"{numbers[0]} satırında"
    return f"{', '.join(numbers[:-1])} ve {numbers[-1]} satırlarında"


def _reference_label(reference: dict[str, Any]) -> str:
    law = str(reference.get("kanun_no") or "").strip() or str(reference.get("kanun_ad") or "").strip() or "kanun belirtilmemiş"
    return f"{law} · madde {str(reference.get('madde') or '').strip() or '—'}"


def explain_warnings(warnings: list[dict[str, Any]], human: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Preparation warnings as sentences the expert can act on, attention first.

    `human` is the annotator's list as prepared (before the 213/413 policy), the
    list `reference_index` counts in. Warnings about one field are grouped into
    one sentence naming the rows (1-based) and their references.
    """
    grouped: dict[str, list[int]] = {}
    notes: list[dict[str, str]] = []
    for warning in warnings:
        code = str(warning.get("code", ""))
        if code.startswith("source_text"):
            continue
        if code in {"reference_null_field", "reference_missing_field"}:
            # Only the decision fields are explained; any other field (the quote,
            # which decides nothing, or one no longer part of a reference) is silent.
            field = str(warning.get("field", ""))
            if field in _EMPTY_FIELD_NOTES:
                grouped.setdefault(field, []).append(int(warning.get("reference_index", 0)))
            continue
        if code == "annotation_incomplete":
            notes.append({"level": "attention", "text": "Bursiyer bu belgeyi platformda “tamamlandı” olarak işaretlememiş; "
                          "listesi yarım kalmış olabilir."})
        elif code == "current_references_missing_or_null":
            notes.append({"level": "attention", "text": "Bursiyer bu belge için referans listesi göndermemiş (platformda "
                          "liste yok); insan anotasyonu boş görünür."})
        else:
            detail = ", ".join(f"{key}: {value}" for key, value in warning.items() if key != "code")
            notes.append({"level": "info", "text": f"Hazırlama notu ({code}){': ' + detail if detail else ''}."})
    for field, (level, explanation) in _EMPTY_FIELD_NOTES.items():
        indices = sorted(set(grouped.get(field, [])))
        if not indices:
            continue
        labels = "; ".join(_reference_label(human[index]) for index in indices if 0 <= index < len(human))
        notes.append({"level": level, "text": f"Bursiyer listesinin {_ordinals(indices)}"
                      f"{f' ({labels})' if labels else ''} {explanation}"})
    return sorted(notes, key=lambda note: note["level"] != "attention")


def _row_label(row: dict[str, Any]) -> str:
    core = row["core"]
    law = " ".join(part for part in (core["kanun_no"], core["kanun_ad"]) if part) or "Kanun belirtilmemiş"
    return f"{law} · madde {core['madde'] or '—'}"


def display_status(row: dict[str, Any]) -> str:
    """How the screen shows a comparison row: its status, or "excused".

    A law-only row beside an article of its law in the same list claims nothing
    more, so the engine counts it as no difference (`ab_diff` gives it "same").
    When only one side wrote it, calling it "Aynı" or colouring it as cited by
    both sides would mislead the expert; it is shown as "Fark sayılmaz".
    """
    if row["status"] == "same" and not (row["a"] and row["b"]) and (row["subsumed"]["a"] or row["subsumed"]["b"]):
        return "excused"
    return row["status"]


def segment_kind(statuses: list[str]) -> str:
    """The colour of a highlighted run: the display status of the rows it cites."""
    counted = [status for status in statuses if status != "excused"]
    if not counted:
        return "neutral"
    if "same" in counted:
        return "both"
    if set(counted) == {"only_a"}:
        return "a"
    if set(counted) == {"only_b"}:
        return "b"
    return "mixed"


def evidence_segments(text: str, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Slice ``text`` into contiguous runs annotated with covering candidates.

    Pure and blind. Given labelled spans (``row_highlights`` output), return
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


def validate_final_references(references: Any) -> list[dict[str, str]]:
    """Accept what the expert decided; refuse only what is not a submission.

    The expert is this pipeline's truth source. Three checks here used to
    reject a whole submission over the *content* of the references, and each
    was wrong for the same reason -- it told the expert their reading of a
    ruling was inadmissible:

    * a redundant row, `193/51` written beside `193/51/1-8`. The annotators do
      this in 177 of the 500 canonical documents; it is a writing convention,
      and `compact_references` already resolves it wherever it matters.
    * a "conflicting law identity", which fired on repealed/successor statute
      pairs sharing a name and is therefore present in the expert-validated
      ground truth itself. An expert could not save doc_234's own references.
    * a row with no law number and no law name, which rejected everything else
      the expert had written along with it.

    None of those are malformed input, so none of them is refused now. A row
    that is entirely blank is dropped, because a blank form row carries no
    claim to lose. A row that has content but no law identity is kept: it will
    match nothing, but the expert wrote it, and discarding it silently would be
    the same mistake in a quieter form.

    Structural validation stays. A payload that is not a list of objects is a
    fault in the page, not a judgement, and there is nothing to normalise.

    There is deliberately no `document_text` parameter. One survived the removal
    of those three checks and was still being passed by every caller, so the
    signature went on promising a grounding check the body had stopped doing.
    """
    validated = validate_reference_list(references)
    validated, _ = apply_reference_policy(validated)
    return [
        reference
        for reference in validated
        if any(str(reference.get(field, "")).strip() for field in REFERENCE_FIELDS)
    ]


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
    final_kept, _ = apply_reference_policy(list(final))
    if compare_reference_lists(original, final_kept).agree:
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
    generation: str = "G0",
) -> Flask:
    secret = validate_runtime_secret(
        session_secret or os.environ.get("DQCHECK_SESSION_SECRET"),
        env_name="DQCHECK_SESSION_SECRET",
    )
    # The GT750 loop's screen opens directly -- no access token, no login page
    # (user decision, 1 Oct 2026, told that anyone with a session on this shared
    # machine can reach 127.0.0.1). It still listens on 127.0.0.1 only, and every
    # decision still carries the page's CSRF token. The old screen keeps its token.
    open_access = has_loop_settings(config)
    token = (
        None
        if open_access
        else validate_runtime_secret(
            access_token or os.environ.get("DQCHECK_ACCESS_TOKEN"),
            env_name="DQCHECK_ACCESS_TOKEN",
        )
    )
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
        # Fail at startup, not per document. Serving a generation this batch was
        # never predicted under still fails -- get_prediction returns None and
        # the review routes raise KeyError -- but it fails one document at a
        # time, after the expert has already opened the queue. A round that
        # serves the wrong generation shows the expert another model's output,
        # so the cheapest place to catch it is before the server starts.
        if not store.list_predictions(batch_id, generation):
            available = store.list_prediction_generations(batch_id)
            raise GateBlocked(
                f"batch {batch_id} has no predictions for generation {generation!r}; "
                f"available generations: {available or 'none'}"
            )
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
        if request.endpoint is None:
            return None  # no such route: a 404 follows, nothing can change
        if open_access:
            session.setdefault("csrf_token", secrets.token_urlsafe(32))
        else:
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

    if not open_access:
        app.add_url_rule("/login", "login", login, methods=["GET", "POST"])

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
            return render_template_string(QUEUE_DONE_TEMPLATE, batch_id=batch_id)
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
                    generation=generation,
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
                generation=generation,
            )
            queue = review_queue(config=config, store=store, batch_id=batch_id)
            try:
                durability = _verified_backup(config=config, store=store, batch_id=batch_id)
            except (IntegrityError, OSError):
                return render_template_string(BACKUP_BLOCKED_TEMPLATE), 503
        diff = ab_diff(document["candidate_a"], document["candidate_b"])
        spans, mentions = row_highlights(document["text"], diff)
        for index, (row, count) in enumerate(zip(diff, mentions, strict=True)):
            row["index"] = index
            row["mentions"] = count
            row["label"] = _row_label(row)
            row["display"] = display_status(row)
        segments = evidence_segments(document["text"], spans)
        for segment in segments:
            cited = [diff[int(candidate)] for candidate in segment["candidates"]]
            segment["kind"] = segment_kind([row["display"] for row in cited]) if cited else ""
            segment["title"] = " | ".join(f"{row['label']} — {row['status_label'].split(' · ')[0]}" for row in cited)
        displays = [row["display"] for row in diff]
        legend_counts = {
            "both": displays.count("same"),
            "a": displays.count("only_a"),
            "b": displays.count("only_b"),
            "mixed": sum(1 for segment in segments if segment["kind"] == "mixed"),
            "neutral": displays.count("excused"),
        }
        return render_template_string(
            REVIEW_TEMPLATE,
            doc=document,
            diff=diff,
            notes=explain_warnings(document["warnings"], document["human_prepared"]),
            rows_payload=[{"a": row["a"], "b": row["b"]} for row in diff],
            legend_counts=legend_counts,
            diff_summary=summarize_diff(diff),
            position=review_position(queue, internal_doc_id),
            durability=durability,
            segments=segments,
            csrf_token=session["csrf_token"],
            opened_at=f"{time.time():.3f}",
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
            # The review row is created when the batch is prepared, so its age
            # is not the expert's time. The page stamps when it was opened.
            try:
                seconds_on_page: float | None = round(time.time() - float(payload["opened_at"]), 1)
            except (KeyError, TypeError, ValueError):
                seconds_on_page = None
            if seconds_on_page is not None and seconds_on_page < 0:
                seconds_on_page = None
            with (
                review_backup_lock(config=config, batch_id=batch_id),
                Store(
                    config.database_path,
                    busy_timeout_ms=config.runtime.busy_timeout_ms,
                ) as store,
            ):
                document = store.get_document(batch_id, internal_doc_id)
                prediction = store.get_prediction(batch_id, internal_doc_id, generation)
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
                    # The judge's list taken as the truth -- the third answer the
                    # expert can give, alongside the annotator's and the model's.
                    action = "judge_override"
                    judge_model = _locked_judge(config, batch_id)
                    if not judge_model and judge_lock_payload(config, batch_id):
                        raise GateBlocked("the judge is hidden from the expert in this loop")
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
                    validated_final = validate_final_references(final)
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
                        "seconds_on_page": seconds_on_page,
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
            if not request.is_json:
                return render_template_string(
                    FORM_ERROR_TEMPLATE,
                    message=(
                        "Bu belge siz sayfayı açtıktan sonra değişti (başka bir sekmede karar "
                        "verilmiş olabilir). Belgeye dönüp sayfayı yenileyin ve kararı yeniden "
                        "kontrol edin."
                    ),
                    detail=str(exc),
                    internal_doc_id=internal_doc_id,
                ), 409
            return jsonify({"error": "version_conflict", "detail": str(exc)}), 409
        except (ContractError, GateBlocked, KeyError, TypeError, ValueError) as exc:
            if not request.is_json:
                return render_template_string(
                    FORM_ERROR_TEMPLATE,
                    message=_form_error_message(str(exc)),
                    detail=str(exc),
                    internal_doc_id=internal_doc_id,
                ), 400
            return jsonify({"error": "invalid_review", "detail": str(exc)}), 400

    return app
