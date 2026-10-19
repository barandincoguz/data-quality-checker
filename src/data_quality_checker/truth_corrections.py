"""Corrections to a sealed release's truth: decided by the user, applied on reading.

A release is immutable (`release_batch`): a second release of a batch is
refused, the training data reads exactly one, and a round never moves back a
stage. So when a released truth turns out to be wrong, the fix is a dated file
beside the release -- `releases/<batch>/truth_corrections.json`, outside the
sealed directory -- that names the release it corrects and holds the user's
decisions in order, each quoted, each with its rule and its corrections. Every
reader of the truth applies them: the round training data
(`round_training_documents`, clean arm; the raw arm reads the annotator's own
lists) and the error report (`round_records`). Each correction re-checks, on
every read, that the list still holds what it corrects.

Two rules (7 Oct 2026, R01):

* `article_range_endpoints_v1` -- an article range is every article in it
  ("473 ila 475" means 473, 474 and 475, "HER MADDE AYRI SATIR OLARAK TUTULUR").
  Four decisions taken before the rule kept an annotator list holding only a
  range's endpoints (620 and 645 for "620 ila 645 inci maddeleri"); the user
  chose "Kuralla düzelt". A correction adds the articles in between.
* `expert_row_fix_v1` -- rows the user, as the expert, replaced or removed after
  the release: the round's training data refused to build because the locator
  could not place them, and each turned out wrong ("mükerrer 257" written
  "257", "4/1" written "41", rows of laws the text never cites). The user chose
  "Önerileri uygula".
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .errors import ContractError
from .reference_matching import canonical_reference

RANGE_RULE = "article_range_endpoints_v1"
ROW_FIX_RULE = "expert_row_fix_v1"
RULES = (RANGE_RULE, ROW_FIX_RULE)
SCHEMA_VERSION = 2
FILE_NAME = "truth_corrections.json"
_ORDINAL = r"(?:['’]?\s*(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü|\.))?"
# "620 ila 645 inci maddeleri", "620-645. maddeler": a range the text itself writes.
TEXT_ARTICLE_RANGE = re.compile(
    rf"(?<![\d/.,])(\d{{1,4}})\s*{_ORDINAL}\s*(?:madde\w*\s*)?(?:ila|-)\s*(\d{{1,4}})(?![\d/])\s*{_ORDINAL}\s*madde",
    re.IGNORECASE,
)
MAX_RANGE_ARTICLES = 300


def _articles_by_law(references: list[dict[str, Any]]) -> dict[str, set[str]]:
    articles: dict[str, set[str]] = defaultdict(set)
    for row in (canonical_reference(r) for r in references):
        if row["kanun_no"]:
            articles[row["kanun_no"]].add(row["madde"])
    return articles


def endpoint_only_ranges(text: str, references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ranges the text writes whose law the list holds by the two endpoints alone."""
    articles = _articles_by_law(references)
    found: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for match in TEXT_ARTICLE_RANGE.finditer(text):
        low, high = int(match.group(1)), int(match.group(2))
        if high - low < 2 or high - low > MAX_RANGE_ARTICLES or (low, high) in seen:
            continue
        seen.add((low, high))
        interior = {str(n) for n in range(low + 1, high)}
        laws = [
            law
            for law, held in sorted(articles.items())
            if {str(low), str(high)} <= held and not interior & held
        ]
        if len(laws) == 1:
            found.append(
                {"kanun_no": laws[0], "low": low, "high": high, "text_writes": f"{low} ila {high}"}
            )
    return found


def _key(row: dict[str, Any]) -> tuple[str, str]:
    canonical = canonical_reference(row)
    return canonical["kanun_no"], canonical["madde"]


def _apply_range(rows: list[dict[str, Any]], correction: dict[str, Any]) -> list[dict[str, Any]]:
    law, low, high = str(correction["kanun_no"]), int(correction["low"]), int(correction["high"])
    canonical = [canonical_reference(r) for r in rows]
    index = {
        end: next(
            (i for i, r in enumerate(canonical) if r["kanun_no"] == law and r["madde"] == str(end)),
            None,
        )
        for end in (low, high)
    }
    if index[low] is None or index[high] is None:
        raise ContractError(
            f"truth correction {law} {low}-{high} no longer applies: the list does not hold both endpoints"
        )
    held = {r["madde"] for r in canonical if r["kanun_no"] == law}
    added = [
        {
            "kanun_no": law,
            "kanun_ad": canonical[index[low]]["kanun_ad"],
            "madde": str(n),
            "source_text": "",
        }
        for n in range(low + 1, high)
        if str(n) not in held
    ]
    return rows[: index[low] + 1] + added + rows[index[low] + 1 :]


def _apply_row_fix(rows: list[dict[str, Any]], correction: dict[str, Any]) -> list[dict[str, Any]]:
    target = (str(correction["row"]["kanun_no"]), str(correction["row"]["madde"]))
    hits = [i for i, row in enumerate(rows) if _key(row) == target]
    if not hits:
        raise ContractError(
            f"truth correction {correction['kind']} {target} no longer applies: the list lacks it"
        )
    kept = [row for i, row in enumerate(rows) if i not in hits]
    if correction["kind"] == "replace":
        new = {**{"kanun_ad": ""}, **correction["with"], "source_text": ""}
        if _key(new) not in {_key(row) for row in kept}:
            kept.insert(hits[0], new)
    return kept


def apply_corrections(
    references: list[dict[str, Any]], corrections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The list with each correction applied in order; the input is never changed.

    A range adds the articles in between after its first endpoint; a replace puts
    its row where the corrected one stood (every repeat of it removed); a remove
    drops every repeat.
    """
    rows = [dict(r) for r in references]
    for correction in corrections:
        kind = correction.get("kind", "range")
        if kind == "range":
            rows = _apply_range(rows, correction)
        elif kind in ("replace", "remove"):
            rows = _apply_row_fix(rows, correction)
        else:
            raise ContractError(f"unknown truth correction kind {kind!r}")
    return rows


def corrections_path(config: Any, batch_id: str) -> Path:
    return Path(config.sensitive_root) / "releases" / batch_id / FILE_NAME


def _read(config: Any, batch_id: str) -> dict[str, Any] | None:
    path = corrections_path(config, batch_id)
    if not path.is_file():
        return None
    from .loop_training import sealed_release_dir

    payload = json.loads(path.read_text(encoding="utf-8"))
    release_id = sealed_release_dir(config, batch_id).name
    rules = [decision.get("rule") for decision in payload.get("decisions", [])]
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("batch_id") != batch_id
        or payload.get("release_id") != release_id
        or not rules
        or any(rule not in RULES for rule in rules)
    ):
        raise ContractError(
            f"{path} corrects batch {payload.get('batch_id')!r} release {payload.get('release_id')!r} "
            f"(schema {payload.get('schema_version')!r}, rules {rules}); the sealed release is {release_id!r}"
        )
    return payload


def load_corrections(config: Any, batch_id: str) -> dict[str, list[dict[str, Any]]]:
    """A batch's corrections by document, in decision order, checked against its one sealed release."""
    payload = _read(config, batch_id)
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for decision in (payload or {}).get("decisions", []):
        for correction in decision["corrections"]:
            by_document[str(correction["internal_doc_id"])].append(correction)
    return dict(by_document)


def _record(
    config: Any, batch_id: str, release_id: str, decision: dict[str, Any]
) -> dict[str, Any]:
    """Add one decision to the batch's file; the same decision again changes nothing, another
    one under the same rule is refused -- a decision is never overwritten."""
    path = corrections_path(config, batch_id)
    payload = _read(config, batch_id) or {
        "schema_version": SCHEMA_VERSION,
        "batch_id": batch_id,
        "release_id": release_id,
        "decisions": [],
    }
    for existing in payload["decisions"]:
        if existing["rule"] != decision["rule"]:
            continue
        if (existing["quote"], existing["corrections"]) == (
            decision["quote"],
            decision["corrections"],
        ):
            return payload
        raise ContractError(
            f"{path} already records another {decision['rule']} decision; it is never overwritten"
        )
    payload["decisions"].append({**decision, "written_at": datetime.now(UTC).isoformat()})
    write_json_atomic(path, payload)
    return payload


def propose_range_corrections(config: Any, batch_id: str) -> dict[str, Any]:
    """Every endpoint-only range of a released batch, nothing written (`loop-correct-ranges`)."""
    from .loop_training import _released_rows, sealed_release_dir

    release_id = sealed_release_dir(config, batch_id).name
    corrections, not_corrected = [], []
    for doc_id, row in sorted(_released_rows(config, [batch_id]).items()):
        for found in endpoint_only_ranges(row["text"], row["references"]):
            entry = {
                "internal_doc_id": doc_id,
                "kind": "range",
                **found,
                "rows_added": found["high"] - found["low"] - 1,
            }
            # The decision is about the expert's lists. A consensus document's truth is the
            # annotator's list, scored as a source too: it is listed, never corrected here.
            (corrections if row["tier"] == "expert_adjudicated" else not_corrected).append(entry)
    return {
        "release_id": release_id,
        "corrections": corrections,
        "consensus_documents_not_corrected": not_corrected,
    }


def write_range_corrections(config: Any, *, batch_id: str, decision: str) -> dict[str, Any]:
    """Record the user's decision to correct every endpoint-only range of a released batch."""
    proposal = propose_range_corrections(config, batch_id)
    return _record(
        config,
        batch_id,
        proposal["release_id"],
        {
            "rule": RANGE_RULE,
            "by": "user",
            "quote": decision,
            "rule_text": '"473 ila 475" = 473, 474 ve 475; her madde ayrı satır (kullanıcı, 7 Ekim 2026)',
            "corrections": proposal["corrections"],
            "consensus_documents_not_corrected": proposal["consensus_documents_not_corrected"],
        },
    )


def write_row_fixes(
    config: Any, *, batch_id: str, decision: str, fixes: list[dict[str, Any]], dry_run: bool = False
) -> dict[str, Any]:
    """Record the user's replacements and removals of released expert rows, each checked to apply.

    A fix is `{"internal_doc_id", "kind": "replace" | "remove", "row": {"kanun_no", "madde"},
    "with": {"kanun_no", "kanun_ad", "madde"} (replace only), "reason"}`.
    """
    from .loop_training import _released_rows, sealed_release_dir

    release_id = sealed_release_dir(config, batch_id).name
    released = _released_rows(config, [batch_id])
    corrections = []
    for fix in fixes:
        doc_id = str(fix["internal_doc_id"])
        if fix.get("kind") not in ("replace", "remove") or ("with" in fix) != (
            fix["kind"] == "replace"
        ):
            raise ContractError(f"a row fix is a replace with `with` or a remove without it: {fix}")
        if doc_id not in released or released[doc_id]["tier"] != "expert_adjudicated":
            raise ContractError(f"{doc_id!r} is not an expert-adjudicated document of {batch_id!r}")
        if not fix.get("reason"):
            raise ContractError(f"a row fix says why: {fix}")
        corrections.append(fix)
    # Every fix must apply to the released list as the other rules' decisions leave it.
    earlier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for decision_on_disk in (_read(config, batch_id) or {}).get("decisions", []):
        if decision_on_disk["rule"] != ROW_FIX_RULE:
            for correction in decision_on_disk["corrections"]:
                earlier[str(correction["internal_doc_id"])].append(correction)
    for doc_id in sorted({c["internal_doc_id"] for c in corrections}):
        apply_corrections(
            apply_corrections(released[doc_id]["references"], earlier.get(doc_id, [])),
            [c for c in corrections if c["internal_doc_id"] == doc_id],
        )
    if dry_run:
        return {"release_id": release_id, "corrections": corrections}
    return _record(
        config,
        batch_id,
        release_id,
        {
            "rule": ROW_FIX_RULE,
            "by": "user",
            "quote": decision,
            "corrections": corrections,
        },
    )


def corrections_summary(config: Any, batch_id: str) -> dict[str, Any] | None:
    """What an error report says about the corrections it scored with; None when there are none."""
    from .fingerprints import sha256_file

    payload = _read(config, batch_id)
    if payload is None:
        return None
    return {
        "file_sha256": sha256_file(corrections_path(config, batch_id)),
        "decisions": [
            {
                "rule": decision["rule"],
                "quote": decision["quote"],
                "documents": sorted({c["internal_doc_id"] for c in decision["corrections"]}),
                "corrections": len(decision["corrections"]),
                "kinds": dict(
                    sorted(_count(c.get("kind", "range") for c in decision["corrections"]).items())
                ),
            }
            for decision in payload["decisions"]
        ],
    }


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return counts
