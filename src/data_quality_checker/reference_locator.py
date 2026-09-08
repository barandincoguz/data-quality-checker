"""Locate a known reference's span in raw document text.

Not extraction. `attach_missing_spans` is handed references whose identity is
already known -- the DQ-Loop harvests them from a prediction's output, absent
from the annotator's, by five-field identity -- and only needs a verbatim
span of the real document that supports each one, because `context_windows.py`
requires `source_text` to be a verbatim substring of the body (`body.find`).

Measured against the 231 references on batch M01 that are in M000's output
and absent from the annotator's:

    article-level span, automatic        197   (85.3%)
    law-level reference, law mention      23   (10.0%)
    refused, goes to the expert            11    (4.8%)

All 220 automatic spans are verbatim substrings of the raw document text.

`annotator/reference/regex.py` was measured on the same 231 first and is not
good enough: it locates 129 (55.8%), and only 62 of those (27% of the total)
are verbatim substrings, because it extracts from `normalize_reference_text`'s
output rather than the raw body. Finding *where a known reference is* is a
narrower problem than finding *which references exist*, so this module is
purpose-built instead of reusing that engine. It imports nothing from
`annotator/` or `benchmark/`, and nothing there should ever import this.

One behaviour is deliberately not copied from the probe that produced the
table above (`scratchpad/locate_probe.py`): that probe fell back to the law
mention whenever it found no article candidate. This module refuses instead,
whenever `madde` is non-empty -- a law-level fallback is correct only for a
reference that names no article at all. Inspecting all 11 refusals in the
M01 document text showed they are mostly the model's own errors: `d000052`
claims VUK articles 350/362/381/396 where the ruling names communiqué serial
numbers ("350, 362, 381 ve 396 Sıra No.lu Vergi Usul Kanunu Genel
Tebliğleri"), not articles. Handing such a reference the law's mention as its
span would teach the model that an article claim is supported by a passage
about something else entirely. An unlocatable article is evidence the
reference is wrong, and the honest response is to surface it to the expert
rather than invent a span.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# Turkish ordinal suffixes that can follow an article number before "madde".
_ORDINAL_SUFFIX = (
    r"(?:\s*['’`´]?\s*(?:inci|ıncı|nci|ncı|uncu|üncü|üncu|ncu|i̇nci))?"
)
_CONTEXT_WINDOW = 90  # characters of context kept on each side of a match


def _article_candidates(text: str, madde: str) -> list[tuple[int, int]]:
    """Character ranges in `text` that plausibly name article `madde`."""
    if not madde:
        return []
    core = re.escape(madde.strip())
    patterns = [
        rf"\b{core}{_ORDINAL_SUFFIX}\s*(?:\.|['’])?\s*madde\w*",  # "5 inci maddesinde"
        rf"madde\w*\s*{core}\b",  # "madde 5"
        rf"\b{core}\s*/\s*\w+",  # "17/4-k"
    ]
    matches: list[tuple[int, int]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append((match.start(), match.end()))
    # A comma series sharing one trailing ordinal + "maddeleri": in
    # "2, 3 ve 4 üncü maddeleri" only the last number carries the suffix, yet
    # every number in the series names an article. Common drafting shape.
    series = re.compile(
        r"(?:\d+\s*(?:,|ve)\s*)+\d+" + _ORDINAL_SUFFIX + r"\s*madde\w*",
        flags=re.IGNORECASE,
    )
    for match in series.finditer(text):
        if re.search(rf"(?<!\d){core}(?!\d)", match.group(0)):
            matches.append((match.start(), match.end()))
    return matches


def _law_positions(text: str, kanun_no: str, kanun_ad: str) -> list[int]:
    """Character offsets in `text` that mention the law by number or name."""
    positions: list[int] = []
    if kanun_no:
        positions += [m.start() for m in re.finditer(rf"\b{re.escape(kanun_no)}\b", text)]
    if kanun_ad:
        words = kanun_ad.split()
        head = words[0] if words else ""
        if len(head) > 3:
            positions += [m.start() for m in re.finditer(re.escape(head), text, re.IGNORECASE)]
    return sorted(positions)


def locate_reference_span(document_text: str, reference: Mapping[str, str]) -> str | None:
    """Return a verbatim span of `document_text` that supports `reference`.

    `reference` carries (at least) the same five decision fields matching and
    routing key on: `kanun_no`, `kanun_ad`, `madde`, `fikra`, `bent`.

    Returns `None` when the reference names an article that cannot be placed
    in the text -- this never falls back to a law-level span for an
    article-level claim; see the module docstring for why. When `madde` is
    empty, the reference names only a law, and the law's own mention in the
    text is the span that supports it.
    """
    madde = str(reference.get("madde", "")).strip()
    kanun_no = str(reference.get("kanun_no", ""))
    kanun_ad = str(reference.get("kanun_ad", ""))
    candidates = _article_candidates(document_text, madde)
    if not candidates:
        if madde:
            # An article was claimed but the text does not place it: evidence
            # the reference itself is wrong. Refuse rather than invent a span.
            return None
        law_positions = _law_positions(document_text, kanun_no, kanun_ad)
        if not law_positions:
            return None
        start = law_positions[0]
        span = document_text[
            max(0, start - 20) : min(len(document_text), start + 2 * _CONTEXT_WINDOW)
        ]
        return span.strip() or None

    law_positions = _law_positions(document_text, kanun_no, kanun_ad)

    def _score(candidate: tuple[int, int]) -> int:
        start, _ = candidate
        if not law_positions:
            return 10**9
        preceding = [position for position in law_positions if position <= start]
        return (start - preceding[-1]) if preceding else 10**9

    start, end = min(candidates, key=_score)
    span = document_text[
        max(0, start - _CONTEXT_WINDOW) : min(len(document_text), end + _CONTEXT_WINDOW)
    ]
    return span.strip() or None


def attach_missing_spans(
    documents: Mapping[Any, dict[str, Any]],
) -> tuple[dict[Any, dict[str, Any]], list[dict[str, str]]]:
    """Fill every empty `source_text` in `documents` with a located span.

    `documents` maps a document id to `{"text": ..., "references": [...]}`, in
    the shape `round_training_documents` reads off a release. Neither the
    mapping nor any reference dict inside it is mutated; a new structure is
    returned with each empty `source_text` replaced by a located span where
    one could be found.

    Also returns a refusal list of `{"doc_id", "kanun_no", "kanun_ad",
    "madde", "fikra", "bent"}`, one entry per reference the locator could not
    place -- this function only reports refusals, it never raises for them.
    `round_training_documents` is the layer that turns a non-empty refusal
    list into one `ContractError` naming all of them at once.
    """
    located: dict[Any, dict[str, Any]] = {}
    refusals: list[dict[str, str]] = []
    for doc_id, document in documents.items():
        text = document["text"]
        new_references: list[Any] = []
        for reference in document["references"]:
            source_text = reference.get("source_text") if isinstance(reference, dict) else None
            if isinstance(source_text, str) and source_text.strip():
                new_references.append(reference)
                continue
            span = locate_reference_span(text, reference)
            if span is None:
                refusals.append(
                    {
                        "doc_id": str(doc_id),
                        "kanun_no": str(reference.get("kanun_no", "")),
                        "kanun_ad": str(reference.get("kanun_ad", "")),
                        "madde": str(reference.get("madde", "")),
                        "fikra": str(reference.get("fikra", "")),
                        "bent": str(reference.get("bent", "")),
                    }
                )
                new_references.append(reference)
                continue
            new_references.append({**reference, "source_text": span})
        located[doc_id] = {**document, "references": new_references}
    return located, refusals
