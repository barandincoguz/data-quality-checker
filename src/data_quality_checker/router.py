"""Deterministic semantic router; similarity never changes the bucket.

A reference is a law and an article of it (user decision, 1 Oct 2026), so a
document is GREEN when both sides name the same provisions and RED when they do
not. There is no YELLOW any more: it meant "same provisions, different
paragraph or sub-item", a difference that no longer exists -- no flag is
raised for it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .normalization import (
    compact_references,
    conflicting_law_identity,
    core_identity,
)
from .reference_policy import (
    DEFAULT_REFERENCE_POLICY_ID,
    apply_reference_policy,
)


@dataclass(frozen=True)
class RouteDecision:
    bucket: str
    reasons: tuple[str, ...]
    similarity: float
    human_references: tuple[dict[str, str], ...]
    model_references: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["human_references"] = list(self.human_references)
        payload["model_references"] = list(self.model_references)
        return payload


def _similarity(human: set[Any], model: set[Any]) -> float:
    if not human and not model:
        return 1.0
    return len(human & model) / len(human | model)


def route_document(
    *,
    human_references: list[dict[str, Any]],
    model_references: list[dict[str, Any]],
    preparation_status: str = "ready",
    model_status: str = "success",
    model_truncated: bool = False,
    has_safe_text: bool = True,
    reference_policy_id: str = DEFAULT_REFERENCE_POLICY_ID,
) -> RouteDecision:
    policy_human, _ = apply_reference_policy(human_references, policy_id=reference_policy_id)
    policy_model, _ = apply_reference_policy(model_references, policy_id=reference_policy_id)
    human = compact_references(policy_human)
    model = compact_references(policy_model)
    core_human = {core_identity(reference) for reference in human}
    core_model = {core_identity(reference) for reference in model}
    similarity = _similarity(core_human, core_model)

    if (
        preparation_status != "ready"
        or model_status != "success"
        or model_truncated
        or not has_safe_text
    ):
        reasons = []
        if preparation_status != "ready":
            reasons.append("preparation_hard_error")
        if model_status != "success":
            reasons.append("model_processing_error")
        if model_truncated:
            reasons.append("model_output_truncated")
        if not has_safe_text:
            reasons.append("safe_text_missing")
        return RouteDecision("QUARANTINE", tuple(reasons), similarity, tuple(human), tuple(model))

    if conflicting_law_identity(human) or conflicting_law_identity(model):
        return RouteDecision(
            "RED",
            ("conflicting_law_identity",),
            similarity,
            tuple(human),
            tuple(model),
        )

    if core_human != core_model:
        reasons = []
        if core_human - core_model:
            reasons.append("missing_core_reference")
        if core_model - core_human:
            reasons.append("extra_or_different_core_reference")
        return RouteDecision("RED", tuple(reasons), similarity, tuple(human), tuple(model))

    return RouteDecision(
        "GREEN",
        ("normalized_law_article_set_equal",),
        similarity,
        tuple(human),
        tuple(model),
    )
