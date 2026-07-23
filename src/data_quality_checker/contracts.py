"""Reference schema normalization and validation."""

from __future__ import annotations

from typing import Any

from .constants import REFERENCE_FIELDS
from .errors import ContractError


def normalize_references(
    raw_references: Any,
    *,
    missing_is_empty: bool = True,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    if raw_references is None:
        if not missing_is_empty:
            raise ContractError("current_references is required")
        warnings.append({"code": "current_references_missing_or_null"})
        return [], warnings
    if not isinstance(raw_references, list):
        raise ContractError("current_references must be a list, null, or missing")

    normalized: list[dict[str, str]] = []
    for index, raw in enumerate(raw_references):
        if not isinstance(raw, dict):
            raise ContractError(f"reference[{index}] must be an object")
        reference: dict[str, str] = {}
        for field in REFERENCE_FIELDS:
            if field not in raw:
                warnings.append(
                    {"code": "reference_missing_field", "reference_index": index, "field": field}
                )
                value: Any = ""
            else:
                value = raw.get(field)
                if value is None:
                    warnings.append(
                        {"code": "reference_null_field", "reference_index": index, "field": field}
                    )
                    value = ""
            reference[field] = str(value).strip()
        normalized.append(reference)
    return normalized, warnings


def validate_reference_list(payload: Any) -> list[dict[str, str]]:
    if not isinstance(payload, list):
        raise ContractError("references must be a list")
    normalized, _ = normalize_references(payload, missing_is_empty=False)
    return normalized
