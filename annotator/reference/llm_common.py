"""Shared parsing and normalization helpers for LLM reference extraction."""

from __future__ import annotations

import json
import re
from typing import Any

from annotator.reference.common import (
    REFERENCE_TYPE,
    compact_law_references,
    deduplicate_references,
    extract_ozelge_no,
)

SUCCESS_STATUS = "success"
ERROR_STATUS = "error"

_FATAL_ERROR_KEYWORDS = (
    "session usage limit",
    "upgrade for higher limits",
    "upgrade for access",
    "account suspended",
    "payment required",
    "requires a subscription",
)

_RETRYABLE_ERROR_KEYWORDS = (
    "403",
    "429",
    "500",
    "502",
    "503",
    "504",
    "deadline exceeded",
    "internal error",
    "internal",
    "json parse",
    "quota",
    "rate limit",
    "resource exhausted",
    "resourceexhausted",
    "server disconnected",
    "server overloaded",
    "remoteprotocolerror",
    "connection reset",
    "connection aborted",
    "temporarily unavailable",
    "timeout",
    "too many requests",
    "unavailable",
)


def repair_json(raw: str) -> dict[str, Any] | list[Any]:
    """Repair common JSON formatting issues returned by LLMs."""
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    def fix_source_text(match: re.Match[str]) -> str:
        inner = match.group(1)
        escaped = inner.replace('\\"', "\x00").replace('"', '\\"').replace("\x00", '\\"')
        return f'"source_text": "{escaped}"'

    fixed = re.sub(r'"source_text"\s*:\s*"((?:[^"\\]|\\.)*)"', fix_source_text, cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    references: list[dict[str, Any]] = []
    # Salvage partially valid objects even when one law field is omitted;
    # this benchmark can still score `kanun_ad + madde` and `kanun_no + madde`.
    for match in re.finditer(r"\{[^{}]*?(?:\"kanun_no\"|\"kanun_ad\"|\"madde\")[^{}]*?\}", cleaned, re.DOTALL):
        block = re.sub(r",\s*\}", "}", match.group(0))
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            references.append(payload)
    if references:
        return {"references": references}

    raise json.JSONDecodeError("Tüm repair denemeleri başarısız", text, 0)


def extract_references(payload: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    """Extract a reference list from an LLM JSON payload."""
    references: Any
    if isinstance(payload, list):
        references = payload
    elif isinstance(payload, dict):
        references = payload.get("references", [])
    else:
        raise TypeError(f"Beklenmeyen LLM çıktı tipi: {type(payload).__name__}")

    if not isinstance(references, list):
        raise TypeError(f"'references' alanı liste olmalı, gelen: {type(references).__name__}")
    return [item for item in references if isinstance(item, dict)]


def normalize_references(references: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Normalize raw LLM references into the benchmark schema."""
    normalized: list[dict[str, Any]] = []
    for reference in references:
        normalized.append(
            {
                "type": REFERENCE_TYPE,
                "kanun_no": str(reference.get("kanun_no", "")).strip(),
                "kanun_ad": str(reference.get("kanun_ad", "")).strip(),
                "madde": str(reference.get("madde", "")).strip(),
                "fikra": str(reference.get("fikra", "")).strip(),
                "bent": str(reference.get("bent", "")).strip(),
                "source_text": str(reference.get("source_text", "")).strip(),
            }
        )
    # LLM benchmark outputs follow the same document-level policy as annotator assistance:
    # keep at most one generic law-only row per law family and collapse repeated same tuples.
    compacted, _ = compact_law_references(normalized)
    return deduplicate_references(compacted)


def parse_response_text(raw_text: str) -> list[dict[str, str]]:
    """Parse and normalize an LLM JSON response body."""
    payload = repair_json(raw_text)
    references = extract_references(payload)
    return normalize_references(references)


def build_success_result(text: str, references: list[dict[str, str]]) -> dict[str, Any]:
    """Build a success result in benchmark-compatible shape."""
    return {
        "references": references,
        "ozelge_no": extract_ozelge_no(text),
        "status": SUCCESS_STATUS,
    }


def build_error_result(text: str, error: str) -> dict[str, Any]:
    """Build an error result in benchmark-compatible shape."""
    return {
        "references": [],
        "ozelge_no": extract_ozelge_no(text),
        "status": ERROR_STATUS,
        "error": str(error),
    }


def enforce_output_contract(result: dict[str, Any], text: str, doc_id: Any | None = None) -> dict[str, Any]:
    """Force a stable output contract for evaluator compatibility."""
    status = str(result.get("status", SUCCESS_STATUS)).strip().lower() or SUCCESS_STATUS
    if status not in {SUCCESS_STATUS, ERROR_STATUS}:
        status = SUCCESS_STATUS

    references = result.get("references", [])
    if not isinstance(references, list):
        references = []

    normalized = {
        "references": normalize_references([ref for ref in references if isinstance(ref, dict)]),
        "ozelge_no": str(result.get("ozelge_no") or extract_ozelge_no(text)).strip(),
        "status": status,
    }

    if doc_id is not None:
        normalized["doc_id"] = doc_id
    if status == ERROR_STATUS and result.get("error"):
        normalized["error"] = str(result.get("error"))
    return normalized


def is_retryable_error(error: str) -> bool:
    """Return whether an error string is retryable for LLM execution."""
    lowered = str(error or "").strip().lower()
    if any(keyword in lowered for keyword in _FATAL_ERROR_KEYWORDS):
        return False
    return any(keyword in lowered for keyword in _RETRYABLE_ERROR_KEYWORDS)
