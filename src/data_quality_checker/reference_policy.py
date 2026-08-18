"""Versioned reference policies shared by DQCheck routing and experiments."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .fingerprints import fingerprint_json
from .normalization import normalize_reference

NO_REFERENCE_FILTER_POLICY_ID = "none"
DEFAULT_REFERENCE_POLICY_ID = "ignore_vuk_213_article_413_v1"

_POLICY_SPECS: dict[str, dict[str, Any]] = {
    NO_REFERENCE_FILTER_POLICY_ID: {
        "schema_version": 1,
        "ignored_normalized_core_identities": [],
    },
    DEFAULT_REFERENCE_POLICY_ID: {
        "schema_version": 1,
        "ignored_normalized_core_identities": [
            {"kanun_no": "213", "madde": "413"},
        ],
        "reason": (
            "VUK article 413 is a systematic tax-ruling boilerplate identity and "
            "is outside the DQCheck annotation-comparison target."
        ),
    },
}


def reference_policy_spec(
    policy_id: str = DEFAULT_REFERENCE_POLICY_ID,
) -> dict[str, Any]:
    try:
        spec = _POLICY_SPECS[policy_id]
    except KeyError as exc:
        raise ValueError(f"unknown reference policy: {policy_id}") from exc
    return {
        "policy_id": policy_id,
        **spec,
        "fingerprint": fingerprint_json({"policy_id": policy_id, **spec}),
    }


def reference_policy_fingerprint(
    policy_id: str = DEFAULT_REFERENCE_POLICY_ID,
) -> str:
    return str(reference_policy_spec(policy_id)["fingerprint"])


def _is_ignored(reference: dict[str, Any], *, policy_id: str) -> bool:
    normalized = normalize_reference(reference)
    ignored = reference_policy_spec(policy_id)["ignored_normalized_core_identities"]
    return any(
        normalized["kanun_no"] == row["kanun_no"] and normalized["madde"] == row["madde"]
        for row in ignored
    )


def apply_reference_policy(
    references: Iterable[dict[str, Any]],
    *,
    policy_id: str = DEFAULT_REFERENCE_POLICY_ID,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return an auditable filtered view without mutating raw references."""

    reference_policy_spec(policy_id)  # fail closed on an unknown policy
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, str]] = []
    for raw in references:
        row = dict(raw)
        if _is_ignored(row, policy_id=policy_id):
            normalized = normalize_reference(row)
            removed.append(
                {
                    "kanun_no": normalized["kanun_no"],
                    "madde": normalized["madde"],
                }
            )
        else:
            kept.append(row)
    return kept, {
        "policy_id": policy_id,
        "policy_fingerprint": reference_policy_fingerprint(policy_id),
        "input_reference_count": len(kept) + len(removed),
        "output_reference_count": len(kept),
        "removed_reference_count": len(removed),
        "removed_normalized_core_identities": removed,
    }
