"""Text normalization and evidence matching shared by preparation and routing."""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style).*?>.*?</\1>")
_BLOCK_END_RE = re.compile(r"(?i)<br\s*/?>|</p>|</div>|</tr>|</li>|</table>")
_TAG_RE = re.compile(r"(?s)<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_LOOSE_RE = re.compile(r"[^0-9a-zçğıöşü]+")
_PUNCT_TRANSLATION = str.maketrans({"“": '"', "”": '"', "’": "'", "‘": "'", "–": "-", "—": "-"})


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(str(value or "")))
    text = text.replace("\u00ad", "").replace("\u200b", "")
    return _SPACE_RE.sub(" ", text).strip()


def html_to_text(value: Any) -> str:
    text = _SCRIPT_STYLE_RE.sub(" ", str(value or ""))
    text = _BLOCK_END_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    return normalize_text(text)


def folded_text(value: Any) -> str:
    return normalize_text(value).casefold().translate(_PUNCT_TRANSLATION)


def loose_text(value: Any) -> str:
    return _LOOSE_RE.sub(" ", folded_text(value)).strip()


def evidence_match_mode(source_text: Any, document_text: str) -> str | None:
    source = normalize_text(source_text)
    if not source:
        return None
    if source in document_text:
        return "normalized_exact"
    if folded_text(source) in folded_text(document_text):
        return "case_punctuation_normalized"
    source_loose = loose_text(source)
    if source_loose and source_loose in loose_text(document_text):
        return "loose_alphanumeric"
    return None


def evidence_coverage(references: list[dict[str, str]], document_text: str) -> float:
    sources = [ref["source_text"] for ref in references if normalize_text(ref["source_text"])]
    if not sources:
        return 0.0
    matched = sum(evidence_match_mode(source, document_text) is not None for source in sources)
    return matched / len(sources)
