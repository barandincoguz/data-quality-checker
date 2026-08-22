"""Deterministic semantic router; similarity never changes the bucket."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .normalization import (
    compact_references,
    conflicting_law_identity,
    core_identity,
    full_identity,
)
from .reference_policy import (
    DEFAULT_REFERENCE_POLICY_ID,
    apply_reference_policy,
)
from .text import folded_text, loose_text


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


def _evidence_compatible(left: str, right: str) -> bool:
    if left == right:
        return True
    folded_left, folded_right = folded_text(left), folded_text(right)
    if folded_left == folded_right:
        return True
    loose_left, loose_right = loose_text(left), loose_text(right)
    if not loose_left and not loose_right:
        return True
    if not loose_left or not loose_right:
        return False
    shorter, longer = sorted((loose_left, loose_right), key=len)
    if shorter not in longer:
        return False
    return len(shorter) >= 16 and len(shorter) / len(longer) >= 0.25


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
    full_human = {full_identity(reference) for reference in human}
    full_model = {full_identity(reference) for reference in model}
    similarity = _similarity(full_human, full_model)

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

    core_human = {core_identity(reference) for reference in human}
    core_model = {core_identity(reference) for reference in model}
    if core_human != core_model:
        reasons = []
        if core_human - core_model:
            reasons.append("missing_core_reference")
        if core_model - core_human:
            reasons.append("extra_or_different_core_reference")
        return RouteDecision("RED", tuple(reasons), similarity, tuple(human), tuple(model))

    if full_human != full_model:
        return RouteDecision(
            "YELLOW",
            ("extension_mismatch",),
            similarity,
            tuple(human),
            tuple(model),
        )

    human_by_key = {full_identity(reference): reference for reference in human}
    model_by_key = {full_identity(reference): reference for reference in model}
    if any(
        not _evidence_compatible(human_by_key[key]["source_text"], model_by_key[key]["source_text"])
        for key in full_human
    ):
        return RouteDecision(
            "YELLOW",
            ("evidence_mismatch",),
            similarity,
            tuple(human),
            tuple(model),
        )
    return RouteDecision(
        "GREEN",
        ("normalized_five_field_set_equal", "evidence_format_or_length_only"),
        similarity,
        tuple(human),
        tuple(model),
    )
