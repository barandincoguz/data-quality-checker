"""Three-way error attribution, rater agreement and expert workload over the DQ-loop.

Three raters produce a legal-reference list for a document -- the
scholarship annotator, the fine-tuned model, and the LLM judge -- and a
human expert then finalizes the truth. The expert is the sole source of
correctness: a metric that scores one rater against another (judge vs.
model, model vs. annotator, ...) instead of against the expert is a
defect this module must not reproduce.

Every reference is compared at *core identity*
(`normalization.core_identity`, a 4-tuple of law-identity, law-number,
law-name and article) after the shared VUK-213/article-413 reference
policy is applied (`reference_policy.apply_reference_policy`) -- uniformly
to all four sources, regardless of whether the row an input list came from
had already been filtered upstream (the judge's stored `final_references`
has; the annotator's and model's raw lists have not). This module composes
those existing primitives; it does not implement a second comparison.

The paper's claim has two halves: each round of cleaned data improves the
model *and* reduces expert workload. `attribute_errors`/`rater_agreement`
above measure the first half; `expert_workload` below measures the second,
from the `reviews` table -- `documents_requiring_review`, the action mix,
`seconds_to_decision`, and `edit_distance_from_best_candidate`. A `defer`
row is the hinge between the two halves: it is real work (it counts toward
workload) but it is not truth (`final_references_json` is stored `null`,
so it must never reach an accuracy computation -- see
`documents_with_finalized_truth`).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .constants import EXPERT_ACTIONS, REVIEW_STATUSES
from .errors import ContractError
from .normalization import compact_references, core_identity
from .performance import _percentile
from .reference_policy import apply_reference_policy

Identity = tuple[str, str, str, str]

_RATER_NAMES: tuple[str, str, str] = ("annotator", "model", "judge")
_FINALIZING_ACTIONS = frozenset(EXPERT_ACTIONS - {"defer"})


def _core_identity_set(references: Iterable[Mapping[str, Any]] | None) -> set[Identity] | None:
    """Return the core-identity set a rater's raw reference list resolves to.

    `None` means the rater has no usable output for this document at all
    (for example a judge result whose `status` never reached `valid`, or a
    review that is not yet `finalized`) and is kept distinct from an empty
    list, which means the rater was consulted and reported nothing. Blurring
    that distinction would turn a data gap into a manufactured miss, so
    `None` propagates as `None` rather than being coerced to an empty set.
    """
    if references is None:
        return None
    kept, _ = apply_reference_policy(list(references))
    compacted = compact_references(kept)
    return {core_identity(reference) for reference in compacted}


def attribute_errors(
    *,
    expert: Iterable[Mapping[str, Any]],
    annotator: Iterable[Mapping[str, Any]] | None,
    model: Iterable[Mapping[str, Any]] | None,
    judge: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Score each rater's references against the expert's, for one document.

    `expert` is the document's finalized truth (`reviews.final_references_json`
    on a row whose `status` is `finalized`) and must not be `None`. Each of
    `annotator`, `model` and `judge` is that rater's raw reference list for
    the same document (`documents.human_references_json`,
    `predictions.references_json`, and `judge_results.result_json ->
    final_references` on a row whose `status` is `valid`, respectively), or
    `None` if the rater has no usable output for this document.

    Returns, per rater, one of:
    - `{"available": False, "correct": None, "missed": None, "spurious": None}`
      when the rater has no data for this document; or
    - `{"available": True, "correct": [...], "missed": [...], "spurious": [...]}`
      where `correct` is the core identities in both the expert's and the
      rater's set, `missed` is in the expert's set but not the rater's, and
      `spurious` is in the rater's set but not the expert's -- each a sorted
      list of core-identity 4-tuples.
    """
    if expert is None:
        raise ContractError("attribute_errors requires the expert's finalized references")
    expert_set = _core_identity_set(expert)
    assert expert_set is not None  # narrows for mypy; expert is never None here

    raw_by_rater = {"annotator": annotator, "model": model, "judge": judge}
    result: dict[str, Any] = {}
    for name in _RATER_NAMES:
        rater_set = _core_identity_set(raw_by_rater[name])
        if rater_set is None:
            result[name] = {
                "available": False,
                "correct": None,
                "missed": None,
                "spurious": None,
            }
            continue
        result[name] = {
            "available": True,
            "correct": sorted(expert_set & rater_set),
            "missed": sorted(expert_set - rater_set),
            "spurious": sorted(rater_set - expert_set),
        }
    return result


def _rate(numerator: int, denominator: int, *, vacuous: float = 1.0) -> float:
    """A ratio with an explicit convention for the empty-denominator case.

    `vacuous` is returned only when the denominator is zero *because there
    was nothing to divide by* (e.g. a rater raised no false positives because
    it produced nothing) -- a legitimate, data-backed vacuous truth, not a
    stand-in for a missing measurement. Callers that instead have *no
    document at all* to measure over must not call this helper; they must
    report `None`, exactly as Cohen's/Fleiss' kappa do below.
    """
    return numerator / denominator if denominator else vacuous


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _cohens_kappa(pairs: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Cohen's kappa over a list of (rater_a_bit, rater_b_bit) item judgments.

    Returns `None` for `cohens_kappa` -- never a fabricated `0.0` -- both
    when the pair shares no items at all (`items == 0`) and when the two
    raters' marginal proportions leave no chance-agreement to correct for
    (`pe == 1`, the degenerate case where both raters are constant over the
    shared items): in neither case does the data support a measurement.
    """
    items = len(pairs)
    if items == 0:
        return {"items": 0, "agreement_matches": 0, "agreement": None, "cohens_kappa": None}
    matches = sum(1 for a, b in pairs if a == b)
    agreement = matches / items
    p_a1 = sum(a for a, _ in pairs) / items
    p_b1 = sum(b for _, b in pairs) / items
    chance_agreement = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    kappa = (
        None if chance_agreement >= 1.0 else (agreement - chance_agreement) / (1 - chance_agreement)
    )
    return {
        "items": items,
        "agreement_matches": matches,
        "agreement": agreement,
        "cohens_kappa": kappa,
    }


def _fleiss_kappa(item_present_counts: Sequence[int], *, rater_count: int) -> dict[str, Any]:
    """Fleiss' kappa over items each rated present/absent by `rater_count` raters.

    `item_present_counts[i]` is how many of the `rater_count` raters marked
    item `i` present. Returns `None` under the same two conditions as
    `_cohens_kappa`: no items, or no chance-agreement left to correct for.
    """
    items = len(item_present_counts)
    if items == 0:
        return {"items": 0, "fleiss_kappa": None}
    total_present = 0
    per_item_agreement_sum = 0.0
    for present in item_present_counts:
        absent = rater_count - present
        per_item_agreement_sum += (present * present + absent * absent - rater_count) / (
            rater_count * (rater_count - 1)
        )
        total_present += present
    mean_agreement = per_item_agreement_sum / items
    total_marks = items * rater_count
    p_present = total_present / total_marks
    p_absent = 1 - p_present
    chance_agreement = p_present * p_present + p_absent * p_absent
    kappa = (
        None
        if chance_agreement >= 1.0
        else (mean_agreement - chance_agreement) / (1 - chance_agreement)
    )
    return {"items": items, "fleiss_kappa": kappa}


def rater_agreement(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate error attribution and inter-rater agreement across documents.

    `records` is one entry per document, each a mapping with an `expert` key
    (that document's finalized truth; required, never `None`) and optional
    `annotator` / `model` / `judge` keys, each that rater's raw reference
    list for the document or `None`/absent if the rater has no usable output
    for it -- the same shapes `attribute_errors` takes.

    **Item universe.** One item is a single core-identity 4-tuple within one
    document: the same identity in two different documents is two different
    items, and two references that normalize to the same core identity
    within one document are one item. For a given document, the universe of
    items is the union of the core identities produced by *every* source
    that has data for it -- expert included -- so a reference only the
    expert kept, that no rater proposed, is still an item every available
    rater is scored absent (`0`) on, exactly as an identity two raters
    proposed and a third did not is an item that third rater is scored `0`
    on. A rater with no data for a document contributes no items for it and
    is excluded, together with that document, from every computation
    involving that rater (its own precision/recall, any pairwise kappa it is
    part of, and Fleiss' kappa).

    Emits, per rater, precision/recall/F1/exact-set-agreement each with
    their tp/fp/fn or match/document counts; `joint_miss_count` -- references
    the expert kept that *no* rater produced, counted only over documents
    where all three raters have data, since a document where a rater is
    simply missing cannot honestly be called a joint miss -- alongside its
    denominator and rate; `annotator_only_correct` / `model_only_correct` /
    `judge_only_correct`, likewise; pairwise agreement and Cohen's kappa for
    each of the three rater pairs; and Fleiss' kappa across all three.
    """
    per_rater_tp = {name: 0 for name in _RATER_NAMES}
    per_rater_fp = {name: 0 for name in _RATER_NAMES}
    per_rater_fn = {name: 0 for name in _RATER_NAMES}
    per_rater_exact_matches = {name: 0 for name in _RATER_NAMES}
    per_rater_documents = {name: 0 for name in _RATER_NAMES}

    joint_miss_count = 0
    total_expert_reference_count = 0
    joint_miss_eligible_document_count = 0
    only_correct = {name: 0 for name in _RATER_NAMES}

    pair_samples: dict[tuple[str, str], list[tuple[int, int]]] = {
        pair: [] for pair in itertools.combinations(_RATER_NAMES, 2)
    }
    fleiss_present_counts: list[int] = []
    fleiss_document_count = 0

    for record in records:
        if "expert" not in record or record["expert"] is None:
            raise ContractError("rater_agreement requires an 'expert' entry for every record")

        identity_sets = {
            "expert": _core_identity_set(record["expert"]),
            **{name: _core_identity_set(record.get(name)) for name in _RATER_NAMES},
        }
        universe: set[Identity] = set()
        for identity_set in identity_sets.values():
            if identity_set is not None:
                universe |= identity_set

        attribution = attribute_errors(
            expert=record["expert"],
            annotator=record.get("annotator"),
            model=record.get("model"),
            judge=record.get("judge"),
        )

        for name in _RATER_NAMES:
            rater = attribution[name]
            if not rater["available"]:
                continue
            per_rater_documents[name] += 1
            per_rater_tp[name] += len(rater["correct"])
            per_rater_fp[name] += len(rater["spurious"])
            per_rater_fn[name] += len(rater["missed"])
            if not rater["missed"] and not rater["spurious"]:
                per_rater_exact_matches[name] += 1

        if all(attribution[name]["available"] for name in _RATER_NAMES):
            joint_miss_eligible_document_count += 1
            total_expert_reference_count += len(identity_sets["expert"])
            missed_sets = {name: set(attribution[name]["missed"]) for name in _RATER_NAMES}
            joint_miss_count += len(set.intersection(*missed_sets.values()))
            correct_sets = {name: set(attribution[name]["correct"]) for name in _RATER_NAMES}
            for name in _RATER_NAMES:
                others = [correct_sets[other] for other in _RATER_NAMES if other != name]
                only_correct[name] += len(correct_sets[name].difference(*others))

        for pair in pair_samples:
            first, second = pair
            if identity_sets[first] is None or identity_sets[second] is None:
                continue
            for item in universe:
                pair_samples[pair].append(
                    (
                        int(item in identity_sets[first]),
                        int(item in identity_sets[second]),
                    )
                )

        if all(identity_sets[name] is not None for name in _RATER_NAMES):
            fleiss_document_count += 1
            for item in universe:
                fleiss_present_counts.append(
                    sum(int(item in identity_sets[name]) for name in _RATER_NAMES)
                )

    per_rater: dict[str, Any] = {}
    for name in _RATER_NAMES:
        tp, fp, fn = per_rater_tp[name], per_rater_fp[name], per_rater_fn[name]
        documents = per_rater_documents[name]
        if documents == 0:
            # No document ever had usable output from this rater -- there is
            # no measurement to report, not a vacuous perfect score.
            per_rater[name] = {
                "available_document_count": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": None,
                "recall": None,
                "f1": None,
                "exact_set_matches": 0,
                "exact_set_documents": 0,
                "exact_set_agreement": None,
            }
            continue
        precision = _rate(tp, tp + fp)
        recall = _rate(tp, tp + fn)
        per_rater[name] = {
            "available_document_count": documents,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "exact_set_matches": per_rater_exact_matches[name],
            "exact_set_documents": documents,
            "exact_set_agreement": per_rater_exact_matches[name] / documents,
        }

    pairwise = {
        f"{first}_{second}": _cohens_kappa(samples)
        for (first, second), samples in pair_samples.items()
    }
    fleiss = _fleiss_kappa(fleiss_present_counts, rater_count=len(_RATER_NAMES))

    return {
        "document_count": len(records),
        "per_rater": per_rater,
        "joint_miss_count": joint_miss_count,
        "total_expert_reference_count": total_expert_reference_count,
        "joint_miss_eligible_document_count": joint_miss_eligible_document_count,
        "joint_miss_rate": (
            joint_miss_count / total_expert_reference_count
            if total_expert_reference_count
            else None
        ),
        "annotator_only_correct": only_correct["annotator"],
        "model_only_correct": only_correct["model"],
        "judge_only_correct": only_correct["judge"],
        "pairwise_agreement": pairwise,
        "fleiss_kappa": fleiss["fleiss_kappa"],
        "fleiss_kappa_items": fleiss["items"],
        "fleiss_kappa_document_count": fleiss_document_count,
    }


# ---------------------------------------------------------------------------
# Expert workload (Task 3): the neglected half of the paper's claim
# ---------------------------------------------------------------------------


def edit_distance_from_best_candidate(
    *,
    final: Iterable[Mapping[str, Any]],
    annotator: Iterable[Mapping[str, Any]] | None,
    model: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """How many references the expert added, removed or altered on a `revise`
    row, relative to whichever candidate -- annotator or model -- was already
    closest to what the expert kept.

    Reuses `attribute_errors` rather than writing a second comparison:
    `final` (the expert's own reference list for this row) stands in for
    `attribute_errors`'s `expert` argument, and `annotator`/`model` are
    scored against it exactly as any rater would be, so the same
    core-identity normalization and VUK-213/article-413 policy
    (`apply_reference_policy`) apply uniformly. A candidate's distance is
    `len(missed) + len(spurious)`: one unit for each reference the expert
    added over that candidate (`missed` -- in `final`, not in the
    candidate) and one for each the expert removed (`spurious` -- in the
    candidate, not in `final`); a reference the expert altered costs two,
    since its old identity is spurious and its new identity is missed.
    Ties are broken toward `"annotator"` for determinism.

    Returns `{"available": False, "closest_candidate": None,
    "edit_distance": None}` when neither candidate has usable data for this
    document -- there is nothing to measure the revision against -- else
    `{"available": True, "closest_candidate": "annotator" | "model",
    "edit_distance": <int>}`.
    """
    attribution = attribute_errors(expert=final, annotator=annotator, model=model, judge=None)
    distances = {
        name: len(attribution[name]["missed"]) + len(attribution[name]["spurious"])
        for name in ("annotator", "model")
        if attribution[name]["available"]
    }
    if not distances:
        return {"available": False, "closest_candidate": None, "edit_distance": None}
    closest_candidate = min(distances, key=lambda name: (distances[name], name != "annotator"))
    return {
        "available": True,
        "closest_candidate": closest_candidate,
        "edit_distance": distances[closest_candidate],
    }


def documents_with_finalized_truth(
    records: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """The subset of `expert_workload`'s records usable as expert truth.

    Only a row whose `status` is `"finalized"` carries a real
    `final_references_json`: `"deferred"` stores it as `null` (real work
    performed, but not truth) and `"pending"` has no decision at all yet.
    Handing a `rater_agreement`/`attribute_errors` call the wrong subset
    would either crash (both require a non-`None` `expert`) or, worse, be
    made to silently accept `None` as "no references" and manufacture an
    accuracy number out of a document the loop never actually finished
    with. This filter is kept in one place so a caller cannot reintroduce a
    deferred or pending row as if it were expert truth.
    """
    return [record for record in records if record.get("status") == "finalized"]


def expert_workload(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate `reviews`-table workload for one round.

    Each entry in `records` is one document's review row plus the routing
    and candidate context needed to place it in the workload picture:
    `router_bucket` (`documents.router_bucket`), `escalated` (whether GREEN
    escalation is active for this document's batch; default `False` --
    see `hitl._trigger_green_escalation`), `status` and `action` (the
    `reviews` row's own values), `created_at_epoch` / `updated_at_epoch`
    (the row's own epoch timestamps), and -- present only where the
    corresponding source has data -- `annotator`
    (`documents.human_references_json`), `model`
    (`predictions.references_json`) and `final`
    (`reviews.final_references_json`, parsed; required non-`None` when
    `status` is `"finalized"`, always `None` when `"deferred"`).

    **`defer` is workload, not truth.** A deferred row means the expert
    opened and triaged the document -- real work -- so it counts in
    `document_count`, `action_distribution["defer"]` and
    `deferred_count`. But its `final_references_json` is `null`, so it is
    never handed to `edit_distance_from_best_candidate` (which only runs
    for `"revise"` rows) or to any other accuracy computation; see
    `documents_with_finalized_truth` for the filter a caller must use
    before computing `rater_agreement` over the same round. Do not fold
    `deferred_count` into a completion measure -- it is reported on its
    own precisely so a deferred document is never mistaken for either a
    finished one or a merely slow one.

    **`seconds_to_decision` is wall time on an open review, not attention.**
    An expert can open a document and return to it an hour later; the
    interval still counts in full. It is reported as p50/p95 over
    `updated_at_epoch - created_at_epoch`, computed only over rows where a
    decision was actually reached (`status` is `"finalized"` or
    `"deferred"`); a `"pending"` row has no decision yet, so its interval
    would describe how long the document has sat untouched, not a
    decision, and is excluded.

    Returns a dict with `document_count`, `documents_requiring_review_count`
    (router `RED`/`YELLOW`, plus `GREEN` under an active escalation) and
    `documents_requiring_review_rate` (`None` when `document_count` is
    `0`); `action_distribution` (one count per `EXPERT_ACTIONS` member plus
    `"total"`, over decided rows only); `deferred_count`;
    `seconds_to_decision_p50` / `seconds_to_decision_p95` (`None` when no
    row has reached a decision) and `seconds_to_decision_sample_count`;
    and `edit_distance_from_best_candidate`, itself
    `{"revise_count", "measured_count", "edit_distances",
    "total_edit_distance", "mean_edit_distance",
    "closest_candidate_counts"}` -- `measured_count` can be less than
    `revise_count` when neither candidate had usable data for a revised
    document, in which case the unmeasured row is simply absent from
    `edit_distances` rather than padded with a fabricated `0`.
    """
    document_count = len(records)
    documents_requiring_review = 0
    action_counts = {action: 0 for action in EXPERT_ACTIONS}
    deferred_count = 0
    decision_seconds: list[float] = []
    revise_count = 0
    edit_distances: list[int] = []
    closest_candidate_counts = {"annotator": 0, "model": 0}

    for record in records:
        if "router_bucket" not in record:
            raise ContractError("expert_workload requires a 'router_bucket' entry for every record")
        if "status" not in record:
            raise ContractError("expert_workload requires a 'status' entry for every record")
        router_bucket = record["router_bucket"]
        escalated = bool(record.get("escalated", False))
        status = record["status"]
        if status not in REVIEW_STATUSES:
            raise ContractError(f"unsupported review status: {status!r}")
        action = record.get("action")
        if action is not None and action not in EXPERT_ACTIONS:
            raise ContractError(f"unsupported review action: {action!r}")

        if router_bucket in {"RED", "YELLOW"} or (escalated and router_bucket == "GREEN"):
            documents_requiring_review += 1

        if status == "pending":
            if action is not None:
                raise ContractError("a pending review must not carry an action")
            continue

        if action is None:
            raise ContractError(f"a {status} review must carry an action")
        if status == "deferred" and action != "defer":
            raise ContractError("a deferred review's action must be 'defer'")
        if status == "finalized" and action not in _FINALIZING_ACTIONS:
            raise ContractError("a finalized review's action must not be 'defer'")
        if status == "finalized" and record.get("final") is None:
            raise ContractError("a finalized review must carry final references")

        action_counts[action] += 1
        if status == "deferred":
            deferred_count += 1

        decision_seconds.append(
            float(record["updated_at_epoch"]) - float(record["created_at_epoch"])
        )

        if action == "revise":
            revise_count += 1
            distance = edit_distance_from_best_candidate(
                final=record.get("final"),
                annotator=record.get("annotator"),
                model=record.get("model"),
            )
            if distance["available"]:
                edit_distances.append(distance["edit_distance"])
                closest_candidate_counts[distance["closest_candidate"]] += 1

    sorted_seconds = sorted(decision_seconds)
    return {
        "document_count": document_count,
        "documents_requiring_review_count": documents_requiring_review,
        "documents_requiring_review_rate": (
            documents_requiring_review / document_count if document_count else None
        ),
        "action_distribution": {**action_counts, "total": sum(action_counts.values())},
        "deferred_count": deferred_count,
        "seconds_to_decision_sample_count": len(decision_seconds),
        "seconds_to_decision_p50": (
            round(_percentile(sorted_seconds, 50), 4) if sorted_seconds else None
        ),
        "seconds_to_decision_p95": (
            round(_percentile(sorted_seconds, 95), 4) if sorted_seconds else None
        ),
        "edit_distance_from_best_candidate": {
            "revise_count": revise_count,
            "measured_count": len(edit_distances),
            "edit_distances": edit_distances,
            "total_edit_distance": sum(edit_distances) if edit_distances else None,
            "mean_edit_distance": (
                sum(edit_distances) / len(edit_distances) if edit_distances else None
            ),
            "closest_candidate_counts": closest_candidate_counts,
        },
    }
