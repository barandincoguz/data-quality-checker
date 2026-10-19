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

`model_round_metrics` and `paired_delta` (Task 4) complete the first half
with the model's own numbers on the frozen test split, and the paired
significance test that decides whether a round-over-round change in them is
real rather than resampling noise. Neither reimplements reference scoring:
`model_round_metrics` wraps the repo's own evaluator
(`g0_training.canonical_evaluate`), and `paired_delta` replicates -- rather
than imports, since it lives in another repository -- the exact bootstrap
method (`paired_core_bootstrap`) that produced the project's one published
external-selection number, parameter for parameter.
"""

from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .constants import EXPERT_ACTIONS, REVIEW_STATUSES
from .errors import ContractError, IntegrityError
from .g0_training import canonical_evaluate
from .performance import _percentile, summarize_operational_records
from .reference_matching import compare_with_policy, reference_identities
from .reference_policy import apply_reference_policy

Identity = tuple[str, str, str, str]

_RATER_NAMES: tuple[str, str, str] = ("annotator", "model", "judge")
_FINALIZING_ACTIONS = frozenset(EXPERT_ACTIONS - {"defer"})


def _core_identity_set(
    references: Iterable[Mapping[str, Any]] | None,
    *,
    context_lists: Sequence[Iterable[Mapping[str, Any]]] = (),
) -> set[Identity] | None:
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
    return reference_identities(
        kept, context_lists=[apply_reference_policy(list(rows))[0] for rows in context_lists]
    )


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
      where `correct` is the core identities (law + article) in both the
      expert's and the rater's set, `missed` is in the expert's set but not the
      rater's, and `spurious` is in the rater's set but not the expert's. Each
      is a sorted list of core-identity 4-tuples. Nothing below the article is
      scored (1 Oct 2026).
    """
    if expert is None:
        raise ContractError("attribute_errors requires the expert's finalized references")
    raw_by_rater = {"annotator": annotator, "model": model, "judge": judge}
    # Every list of the document aligns the same way (a law named without its
    # number takes it from wherever the document states it), so an identity
    # here is the identity the kappas and the journal table use.
    available = [list(rows) for rows in raw_by_rater.values() if rows is not None]
    result: dict[str, Any] = {}
    for name in _RATER_NAMES:
        if raw_by_rater[name] is None:
            result[name] = {
                "available": False,
                "correct": None,
                "missed": None,
                "spurious": None,
            }
            continue
        comparison = compare_with_policy(
            list(expert), list(raw_by_rater[name]), context_lists=available
        )
        result[name] = {
            "available": True,
            "correct": sorted(comparison.matched_identities),
            "missed": sorted(comparison.missed_identities),
            "spurious": sorted(comparison.extra_identities),
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
    their tp/fp/fn or match/document counts. The annotator's `source_text` is never read: it
    decides nothing (user rule, 27 September 2026); `joint_miss_count` -- references
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

        document_lists = [record["expert"], *(record.get(name) for name in _RATER_NAMES)]
        document_lists = [list(rows) for rows in document_lists if rows is not None]
        identity_sets = {
            "expert": _core_identity_set(record["expert"], context_lists=document_lists),
            **{
                name: _core_identity_set(record.get(name), context_lists=document_lists)
                for name in _RATER_NAMES
            },
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


def complementarity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How the annotator and the model complement each other against the expert's truth.

    For every finalized document with both sources (7 Oct 2026, user: what is
    measured when the expert decides "neither is right" and writes the list?):
    each reference of the expert's truth is counted by who had it -- the
    annotator, the model, both, either, neither (only the expert brought it),
    and none of the three when the judge is counted too -- and the references
    both sources wrote that the expert did not keep. `union_recall` is the
    share of the truth the two sources together already held. Computed on the
    project's one comparison (`attribute_errors`), for all finalized documents
    and for the `revise` decisions alone. A document missing a source is left
    out and counted, never scored as a miss.
    """

    def block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        counts: Counter[str] = Counter()
        documents = skipped = 0
        for row in rows:
            if row.get("status") != "finalized" or row.get("final") is None:
                continue
            if row.get("annotator") is None or row.get("model") is None:
                skipped += 1
                continue
            documents += 1
            att = attribute_errors(
                expert=row["final"],
                annotator=row["annotator"],
                model=row["model"],
                judge=row.get("judge"),
            )
            found = {
                name: {tuple(i) for i in att[name]["correct"]}
                for name in _RATER_NAMES
                if att[name]["available"]
            }
            truth = {tuple(i) for i in att["annotator"]["correct"] + att["annotator"]["missed"]}
            annotator, model, judge = found["annotator"], found["model"], found.get("judge", set())
            counts["truth_references"] += len(truth)
            counts["found_by_annotator"] += len(annotator)
            counts["found_by_model"] += len(model)
            counts["found_by_judge"] += len(judge)
            counts["found_by_both"] += len(annotator & model)
            counts["found_by_either"] += len(annotator | model)
            counts["found_by_neither"] += len(truth - (annotator | model))
            counts["found_by_none_of_three"] += len(truth - (annotator | model | judge))
            counts["wrong_in_both"] += len(
                {tuple(i) for i in att["annotator"]["spurious"]}
                & {tuple(i) for i in att["model"]["spurious"]}
            )
        keys = (
            "truth_references",
            "found_by_annotator",
            "found_by_model",
            "found_by_judge",
            "found_by_both",
            "found_by_either",
            "found_by_neither",
            "found_by_none_of_three",
            "wrong_in_both",
        )
        return {
            "documents": documents,
            "documents_without_both_sources": skipped,
            **{key: counts[key] for key in keys},
            "union_recall": counts["found_by_either"] / counts["truth_references"]
            if counts["truth_references"]
            else None,
        }

    return {
        "all_finalized": block(records),
        "revise": block([row for row in records if row.get("action") == "revise"]),
        # Decisions carried from an earlier loop chose among that loop's lists
        # (R01: its "accept_model" took the earlier model), so the figures that
        # describe this loop's sources leave them out.
        "new_decisions": block([row for row in records if not row.get("carried")]),
        "carried_documents": sum(
            1 for row in records if row.get("carried") and row.get("status") == "finalized"
        ),
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
    `router_bucket` (`documents.router_bucket`), `in_green_audit_sample`
    (whether the document is in this batch's GREEN audit sample, from
    `ensure_green_audit_plan`'s `sample_internal_doc_ids`), `escalated` (whether GREEN
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

    **`seconds_to_decision` is time with the review page open, not attention.**
    It is each record's `seconds_on_page` -- the page stamps when it was
    opened, the decision event stores the interval, summed over every
    decision on the document (`loop_bindings.round_records`). An expert can
    open a document and return to it an hour later; the interval still
    counts in full. It used to be `updated_at_epoch - created_at_epoch`, but a
    review row is created when the batch is prepared, so that measured the
    batch's age. A decided row without a recorded page time is left out of the
    sample rather than guessed; `expert_seconds_total` is the sum over the
    sample. A `"pending"` row has no decision yet and is excluded.

    Returns a dict with `document_count`, `documents_requiring_review_count`
    (router `RED`/`YELLOW`, the GREEN audit sample, `GREEN` under an active
    escalation, and the judge's pilot documents -- `in_judge_pilot`), of which
    `judge_pilot_only_review_count` were required by the pilot alone, and
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
    judge_pilot_only = 0
    action_counts = {action: 0 for action in EXPERT_ACTIONS}
    deferred_count = 0
    decision_seconds: list[float] = []
    # A decision carried from an earlier loop (R01: 23 from the fıkra/bent loop)
    # was timed on that loop's screen; the new-decision figures leave it out.
    new_decision_seconds: list[float] = []
    carried_decisions = 0
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

        # Mirrors hitl._review_requirements exactly: the GREEN audit sample is
        # required review regardless of escalation. Leaving it out would make
        # workload appear to fall faster than it does -- as the model improves,
        # more documents route GREEN, but the audit sample does not shrink with
        # them, so review work has a floor that this metric must show.
        in_green_audit_sample = bool(record.get("in_green_audit_sample", False))
        routed_to_review = (
            router_bucket in {"RED", "YELLOW"}
            or in_green_audit_sample
            or (escalated and router_bucket == "GREEN")
        )
        # The judge's pilot exam is review work too, but it is the price of
        # measuring the judge, not of routing: reported apart so the routing
        # half of the claim is not flattered or penalised by it.
        in_judge_pilot = bool(record.get("in_judge_pilot", False))
        if routed_to_review or in_judge_pilot:
            documents_requiring_review += 1
        if in_judge_pilot and not routed_to_review:
            judge_pilot_only += 1

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

        if record.get("seconds_on_page") is not None:
            decision_seconds.append(float(record["seconds_on_page"]))
            if not record.get("carried"):
                new_decision_seconds.append(float(record["seconds_on_page"]))
        if record.get("carried") and status == "finalized":
            carried_decisions += 1

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
        "judge_pilot_only_review_count": judge_pilot_only,
        "documents_requiring_review_rate": (
            documents_requiring_review / document_count if document_count else None
        ),
        "action_distribution": {**action_counts, "total": sum(action_counts.values())},
        "deferred_count": deferred_count,
        "seconds_to_decision_sample_count": len(decision_seconds),
        "carried_decision_count": carried_decisions,
        "seconds_to_decision_new_p50": (
            round(_percentile(sorted(new_decision_seconds), 50), 4)
            if new_decision_seconds
            else None
        ),
        "seconds_to_decision_new_p95": (
            round(_percentile(sorted(new_decision_seconds), 95), 4)
            if new_decision_seconds
            else None
        ),
        "expert_seconds_new_total": round(sum(new_decision_seconds), 1)
        if new_decision_seconds
        else None,
        "expert_seconds_total": round(sum(decision_seconds), 1) if decision_seconds else None,
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


# ---------------------------------------------------------------------------
# Model round metrics (Task 4): the model half of the paper's claim, on the
# frozen test split, plus the paired significance test that decides whether a
# round-over-round change in it is real.
# ---------------------------------------------------------------------------

_CORE_METRIC_FIELDS = ("tp", "fp", "fn", "precision", "recall", "f1")


def model_round_metrics(
    *,
    config: Any,
    predictions_path: Path,
    doc_ids_path: Path,
    output_dir: Path,
    expected_doc_count: int,
) -> dict[str, Any]:
    """Score one round's model on the frozen test split.

    Wraps the repo's own evaluator (`g0_training.canonical_evaluate`) rather
    than reimplementing reference matching; this function contributes no
    scoring logic of its own, only extraction of what the evaluator already
    computed plus the operational counters scoring never touches.

    `expected_doc_count` has no default on purpose. `canonical_evaluate`
    defaults it to 50 -- the G0 development-validation set size -- which is
    the wrong number for a round's frozen test split, and silently passing
    that default here would score against the wrong document count without
    any error. A caller must state the real split size explicitly, exactly as
    `loop_bindings.measure_step` already requires for the same reason.
    Stating the wrong count does not fail silently either way: the
    evaluator's own coverage gate raises `IntegrityError` the moment
    `evaluated_doc_count` disagrees with it.

    `predictions_path` is read twice: once by the evaluator itself (for
    scoring) and once here (for the `operational` counters -- latency,
    tokens, peak memory, truncation -- that scoring never touches, and that
    `canonical_evaluate`'s report does not carry). Rows are restricted to the
    `doc_ids_path` universe before either use, so a predictions file that
    happens to carry extra rows outside this round's frozen split cannot
    inflate the operational counters or the status counts below.

    Returns a dict with:
    - `document_count`: the evaluated document count (`== expected_doc_count`
      once the coverage gate above has passed).
    - `core`: `{tp, fp, fn, precision, recall, f1}` at core identity (law +
      article; the evaluator's `core_law_article_strict`) -- the only level a
      reference has (1 Oct 2026), the one `paired_delta` below and the loop's
      own checkpoint selection both use.
    - `docwise_f1_distribution`: the evaluator's own per-document core-F1
      distribution (`core_doc_level_diagnostics`) -- `perfect_f1_count`,
      `zero_f1_count`, and the min/max/mean/median/p25/p75 of per-document F1
      -- alongside `total_docs` as their shared denominator. This is a
      *distribution* of how thoroughly each document was solved; it is not
      `docwise_core_accuracy` (a single pass/fail rate at one threshold),
      which this function does not emit.
    - `core_per_document`: `{doc_id: {"tp", "fp", "fn"}}` over the evaluated
      universe, at core identity -- exactly the shape `paired_delta` below
      takes as its `previous`/`current` arguments, so a caller never has to
      reshape this function's own output to run the significance test.
    - `parse_failure_count`, `truncation_count`: a genuine model-output
      failure (produced nothing parseable, or was cut off at the generation
      limit) is exactly one of these two, never both, so they never double
      count the same document. `truncation_count` is the project's
      established `operational.truncated` count
      (`performance.summarize_operational_records`'s `reliability` block --
      the same field `router.route_document` and `hitl` already key off);
      `parse_failure_count` is what remains of the evaluator's own
      `error`-status count once truncations are subtracted out, so a
      document whose input alone exceeded the context window before
      generation was ever attempted is counted here too -- it likewise never
      produced a valid reference list, and it was not a generation-limit
      truncation.
    - `prediction_status_counts`: the raw status tally (`success`/`error`/...)
      over the restricted rows, so `parse_failure_count`'s derivation from it
      is auditable rather than opaque.
    - `operational`: `performance.summarize_operational_records` over the
      same restricted rows -- latency (p50/p95 among other percentiles),
      input/output token distributions, and peak memory, so no accuracy
      number here ships without them.
    """
    predictions_path = Path(predictions_path)
    doc_ids_path = Path(doc_ids_path)
    output_dir = Path(output_dir)

    doc_ids = {int(doc_id) for doc_id in json.loads(doc_ids_path.read_text(encoding="utf-8"))}
    raw_predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(raw_predictions, list):
        raise ContractError("model_round_metrics requires predictions_path to hold a JSON array")

    rows_by_doc_id: dict[int, dict[str, Any]] = {}
    for row in raw_predictions:
        if isinstance(row, dict) and row.get("doc_id") in doc_ids:
            # A repeated doc_id keeps the last block, mirroring the evaluator's
            # own `load_predictions` duplicate policy exactly.
            rows_by_doc_id[row["doc_id"]] = row
    rows = [rows_by_doc_id[doc_id] for doc_id in sorted(rows_by_doc_id)]

    report = canonical_evaluate(
        config=config,
        predictions_path=predictions_path,
        doc_ids_path=doc_ids_path,
        output_dir=output_dir,
        expected_doc_count=expected_doc_count,
    )
    result = report["results"][0]
    if "core_per_doc_metrics" not in result:
        # canonical_evaluate always passes `--per-doc`; this can only fire if
        # that changes underneath this function, and it must fail loudly
        # rather than silently short the per-document universe paired_delta needs.
        raise ContractError("canonical evaluator report is missing per-document core metrics")

    core_metric = result["core_law_article_strict"]
    docwise = result["core_doc_level_diagnostics"]

    status_counts: Counter[str] = Counter(
        str(row.get("status", "success")).strip().lower() or "success" for row in rows
    )
    error_count = status_counts.get("error", 0)

    operational_records = [dict(row.get("operational") or {}) for row in rows]
    operational_summary = summarize_operational_records(
        operational_records,
        provenance={"source": "model_round_metrics", "document_count": len(rows)},
    )
    truncation_count = operational_summary["reliability"]["truncated_count"]
    parse_failure_count = error_count - truncation_count
    if parse_failure_count < 0:
        raise IntegrityError(
            "truncated-prediction count exceeds the evaluator's own error-status "
            "count; the operational and status counters disagree"
        )

    core_per_document = {
        int(entry["doc_id"]): {
            "tp": int(entry["tp"]),
            "fp": int(entry["fp"]),
            "fn": int(entry["fn"]),
        }
        for entry in result["core_per_doc_metrics"]
    }

    f1_distribution = docwise["f1_distribution"]
    return {
        "document_count": result["evaluated_doc_count"],
        "core": {field: core_metric[field] for field in _CORE_METRIC_FIELDS},
        "docwise_f1_distribution": {
            "total_docs": docwise["total_docs"],
            "perfect_f1_count": docwise["perfect_f1_count"],
            "zero_f1_count": docwise["zero_f1_count"],
            "min_f1": f1_distribution["min"],
            "max_f1": f1_distribution["max"],
            "mean_f1": f1_distribution["mean"],
            "median_f1": f1_distribution["median"],
            "p25_f1": f1_distribution["p25"],
            "p75_f1": f1_distribution["p75"],
        },
        "core_per_document": core_per_document,
        "prediction_status_counts": dict(status_counts),
        "parse_failure_count": parse_failure_count,
        "truncation_count": truncation_count,
        "operational": operational_summary,
    }


# `paired_core_bootstrap`'s own bootstrap parameters
# (`ner-project/scripts/dqcheck_g0_external_selection.py:908-909`), echoed
# here as named constants rather than repeated literals so the two numbers
# that must never drift from the published method live in exactly one place.
PAIRED_BOOTSTRAP_SAMPLES = 10_000
PAIRED_BOOTSTRAP_SEED = 42


def _bootstrap_quantile(values: Sequence[float], fraction: float) -> float:
    """Linearly-interpolated quantile between order statistics.

    Bit-for-bit the published method's own `_quantile`
    (`ner-project/scripts/dqcheck_g0_external_selection.py:895-903`), and
    deliberately *not* this repo's `performance._percentile` (nearest-rank):
    the two give different numbers for the same sample, and this function
    exists to reproduce the one number the project has already published, not
    a differently-rounded one.
    """
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires values")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_f1(tp: int, fp: int, fn: int) -> float:
    """F1 from summed tp/fp/fn, with a 0.0 (not 1.0) vacuous convention.

    Deliberately not this module's own `_rate`/`_f1` (which default an
    empty-denominator precision or recall to a vacuous 1.0, appropriate for
    error attribution against the expert): the published bootstrap method
    being replicated here (`paired_core_bootstrap`'s own `_metric_dict`) uses
    0.0 for both, and matching it bit-for-bit -- not merely approximately --
    is the entire point of this function.
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _require_core_counts(mapping: Mapping[Any, Mapping[str, int]], *, name: str) -> None:
    for doc_id, counts in mapping.items():
        missing = {"tp", "fp", "fn"} - counts.keys()
        if missing:
            raise ContractError(f"paired_delta: {name}[{doc_id!r}] is missing {sorted(missing)}")


def paired_delta(
    previous: Mapping[Any, Mapping[str, int]],
    current: Mapping[Any, Mapping[str, int]],
    *,
    samples: int = PAIRED_BOOTSTRAP_SAMPLES,
    seed: int = PAIRED_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired bootstrap over the frozen test split's shared documents.

    Replicates -- rather than imports, since it lives in another repository
    -- `paired_core_bootstrap`
    (`ner-project/scripts/dqcheck_g0_external_selection.py:910-957`), the
    method that produced the only external-selection number this project has
    published, parameter for parameter: `samples=10_000`, `seed=42`
    (`random.Random(seed)`), resampling document ids *with replacement* from
    the shared universe (`[rng.choice(doc_ids) for _ in doc_ids]`), and
    computing F1 from tp/fp/fn *summed* across each resample -- never from
    averaging each document's own F1, which is a different estimator and
    would not reproduce that number.

    `previous` and `current` are each `{doc_id: {"tp", "fp", "fn"}}` at core
    identity -- exactly `model_round_metrics`'s own `core_per_document`
    output -- for the round being compared against and the round being
    evaluated, respectively. The two must share the exact same document
    universe: a paired test compares each document against itself across
    rounds, so a mismatched universe (documents scored in one round but not
    the other) cannot honestly be paired, and is refused with a
    `ContractError` rather than silently scored over whichever documents
    happen to be common or padded with a fabricated pairing.

    A round that reports ΔF1 without this interval is not evidence -- the
    interval is the entire deliverable. This is why `verdict` is added on top
    of the published method's own fields: it is one of `"improved"`,
    `"regressed"` or `"inconclusive"`, and `"inconclusive"` fires whenever the
    95% interval (`lower_2_5`, `upper_97_5`) spans zero, *regardless of the
    sign of `bootstrap_mean`* -- a positive mean with an interval that still
    reaches below zero is exactly the noisy case a round-over-round claim
    must not read as improvement.

    Returns `samples`, `seed` (both echoed so a report is self-describing);
    `document_count` (the shared universe size); `lower_2_5`, `upper_97_5`,
    `bootstrap_mean` (of `current_f1 - previous_f1` over the bootstrap
    resamples); `delta_gt_zero_count` alongside `probability_delta_gt_zero`
    (`delta_gt_zero_count / samples`, never a bare rate); `per_document`
    (`challenger_win` / `tie` / `incumbent_win`, from each document's own
    observed F1 -- current vs. previous -- once, not resampled); and
    `verdict`.
    """
    if not isinstance(samples, int) or samples <= 0:
        raise ContractError("paired_delta requires a positive integer sample count")

    doc_ids = sorted(current)
    if doc_ids != sorted(previous):
        raise ContractError("paired_delta requires the same document universe for both rounds")
    if not doc_ids:
        raise ContractError("paired_delta requires at least one shared document")
    _require_core_counts(previous, name="previous")
    _require_core_counts(current, name="current")

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        sampled = [rng.choice(doc_ids) for _ in doc_ids]
        current_f1 = _bootstrap_f1(
            sum(current[doc_id]["tp"] for doc_id in sampled),
            sum(current[doc_id]["fp"] for doc_id in sampled),
            sum(current[doc_id]["fn"] for doc_id in sampled),
        )
        previous_f1 = _bootstrap_f1(
            sum(previous[doc_id]["tp"] for doc_id in sampled),
            sum(previous[doc_id]["fp"] for doc_id in sampled),
            sum(previous[doc_id]["fn"] for doc_id in sampled),
        )
        deltas.append(current_f1 - previous_f1)

    lower = _bootstrap_quantile(deltas, 0.025)
    upper = _bootstrap_quantile(deltas, 0.975)
    if lower > 0:
        verdict = "improved"
    elif upper < 0:
        verdict = "regressed"
    else:
        verdict = "inconclusive"

    win_tie_loss = {"challenger_win": 0, "tie": 0, "incumbent_win": 0}

    def document_score(counts: Mapping[str, int]) -> float:
        # A document with no gold reference that is left empty is right (1.0),
        # as the evaluator's document accuracy counts it; under the bootstrap's
        # 0.0 convention it tied with a document given a spurious row (engine
        # audit, 2 Oct 2026; validation doc 262). Only this count uses it.
        if not (counts["tp"] or counts["fp"] or counts["fn"]):
            return 1.0
        return _bootstrap_f1(counts["tp"], counts["fp"], counts["fn"])

    for doc_id in doc_ids:
        current_doc_f1 = document_score(current[doc_id])
        previous_doc_f1 = document_score(previous[doc_id])
        if current_doc_f1 > previous_doc_f1:
            win_tie_loss["challenger_win"] += 1
        elif current_doc_f1 < previous_doc_f1:
            win_tie_loss["incumbent_win"] += 1
        else:
            win_tie_loss["tie"] += 1

    delta_gt_zero_count = sum(1 for delta in deltas if delta > 0)
    return {
        "metric": "core_law_article_strict.f1",
        "delta_direction": "current_minus_previous",
        "document_count": len(doc_ids),
        "samples": samples,
        "seed": seed,
        "lower_2_5": lower,
        "upper_97_5": upper,
        "bootstrap_mean": sum(deltas) / len(deltas),
        "delta_gt_zero_count": delta_gt_zero_count,
        "probability_delta_gt_zero": delta_gt_zero_count / len(deltas),
        "per_document": win_tie_loss,
        "verdict": verdict,
    }


# --- the journal's per-batch table (user story, 1 Oct 2026) -----------------


def _claims(
    references: Iterable[Mapping[str, Any]],
    *,
    context: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[int, int, set[Any], int]:
    """(rows after the policy, rows the policy dropped, reference set, distinct
    references among the rows).

    A reference is a law + article claim (1 Oct 2026: nothing below the
    article). Rows another row already accounts for are one claim
    (`compact_references`); 213/413 is dropped (`apply_reference_policy`). The
    distinct references tell a row written twice ("Geçici 21", "geçici 21") from
    a law-only row an article of the same law covers ("193" beside "193/94").
    """
    rows = list(references)
    kept, _ = apply_reference_policy(rows)
    other = [apply_reference_policy(list(context))[0]] if context is not None else []
    claims = reference_identities(kept, context_lists=other)
    # Distinct references among the rows, a law-only row beside an article of
    # its law included (written, but extra): the difference to `claims` is the
    # table's `subsumed_rows`.
    from .reference_matching import DocumentContext, canonical_reference

    canonical = [canonical_reference(reference) for reference in kept]
    aligned = DocumentContext(
        canonical, *([canonical_reference(r) for r in rows_] for rows_ in other)
    )
    distinct = {aligned.identity(row) for row in canonical}
    return (len(kept), len(rows) - len(kept), claims, len(distinct))


def _table_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _table_bootstrap_f1(
    per_document: Sequence[tuple[int, int, int]], *, samples: int, seed: int
) -> list[float]:
    """F1 of summed tp/fp/fn over documents resampled with replacement (as `paired_delta`)."""
    if not per_document:
        return []
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        draw = rng.choices(per_document, k=len(per_document))
        values.append(
            _table_prf(sum(d[0] for d in draw), sum(d[1] for d in draw), sum(d[2] for d in draw))[2]
        )
    return sorted(values)


def _interval(values: Sequence[float]) -> list[float] | None:
    """95% interval by the published method's own quantile (`_bootstrap_quantile`),
    so every interval in the paper is computed one way."""
    if not values:
        return None
    return [_bootstrap_quantile(values, 0.025), _bootstrap_quantile(values, 0.975)]


def reference_table(
    records: Sequence[Mapping[str, Any]],
    *,
    samples: int = PAIRED_BOOTSTRAP_SAMPLES,
    seed: int = PAIRED_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Per source: what it found, what it got wrong, what it missed, against the truth.

    `records`: one per document, `{"doc_id", "truth", "truth_source", "annotator",
    "model", "judge"}` (`scoped_truth_records`); a source's list is `None` when it
    has nothing for that document, which then leaves its counts. A reference is a
    law and an article of it -- the only level there is (1 Oct 2026). For each
    source:

    - `written_rows` / `policy_dropped`: rows it wrote after / dropped by 213/413;
      of the written rows, `duplicate_rows` repeat a reference already written
      (case/spelling apart) and `subsumed_rows` are law-only rows an article of
      the same law covers ("193" beside "193/94");
    - `written_references`: distinct references it made;
    - `truth_references`: the same for the truth, over the documents the source has;
    - `core`: tp / fp / fn, P / R / F1 and a 95% document-bootstrap interval for
      F1 (`samples`, `seed`);
    - `wrong_references` (its references the truth does not hold) and
      `missed_references` (the truth's it does not).

    `per_document` keeps each document's counts, for batch-to-batch deltas.
    """
    sources: dict[str, Any] = {}
    per_document: dict[str, dict[str, Any]] = {}
    for name in _RATER_NAMES:
        totals = Counter()
        rows_core: list[tuple[int, int, int]] = []
        documents = 0
        for record in records:
            if record.get(name) is None:
                continue
            documents += 1
            kept, dropped, _, distinct = _claims(record[name], context=record["truth"])
            comparison = compare_with_policy(list(record["truth"]), list(record[name]))
            counts_core = (comparison.tp, comparison.fp, comparison.fn)
            rows_core.append(counts_core)
            # Every count from the one comparison: what it wrote that counts
            # (found + wrong), the truth's references (found + missed), and the
            # law-only rows beside an article of their law that were not scored.
            totals.update(
                written_rows=kept,
                policy_dropped=dropped,
                duplicate_rows=kept - distinct,
                subsumed_rows=len(comparison.excused_candidate),
                written_references=comparison.tp + comparison.fp,
                truth_references=comparison.tp + comparison.fn,
            )
            per_document.setdefault(str(record["doc_id"]), {})[name] = {"core": list(counts_core)}
        tp, fp, fn = (sum(row[i] for row in rows_core) for i in range(3))
        precision, recall, f1 = _table_prf(tp, fp, fn)
        level = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f1_ci95": _interval(_table_bootstrap_f1(rows_core, samples=samples, seed=seed)),
        }
        sources[name] = {
            "documents": documents,
            **{
                key: totals.get(key, 0)
                for key in (
                    "written_rows",
                    "policy_dropped",
                    "duplicate_rows",
                    "subsumed_rows",
                    "written_references",
                    "truth_references",
                )
            },
            "core": level,
            "wrong_references": fp,
            "missed_references": fn,
        }
    return {
        "documents": len(records),
        "documents_by_truth_source": dict(
            Counter(str(record["truth_source"]) for record in records)
        ),
        "bootstrap": {"samples": samples, "seed": seed, "unit": "document"},
        "level": "law_article",
        "sources": sources,
        "per_document": per_document,
    }


def scoped_truth_records(records: Sequence[Mapping[str, Any]], *, scope: str) -> dict[str, Any]:
    """The documents of a batch that have a truth, under one of two scopes.

    `reviewed`: the expert's finalized lists only. `batch`: those, plus every
    GREEN document the expert did not review -- whose truth is the list annotator
    and model agreed on (GREEN means they agree) -- but only once the GREEN audit
    sample is fully decided and nothing escalated: until then the agreed lists
    are not truth yet. Records are `loop_bindings.round_records`'.
    """
    if scope not in ("reviewed", "batch"):
        raise ContractError(f"scope must be 'reviewed' or 'batch', got {scope!r}")
    escalated = any(record.get("escalated") for record in records)
    audit = [record for record in records if record.get("in_green_audit_sample")]
    audit_passed = not escalated and all(record.get("status") == "finalized" for record in audit)
    out, without = [], []
    for record in records:
        doc_id = str(record["internal_doc_id"])
        if record.get("status") == "finalized":
            truth, source = record["final"], "expert"
        elif scope == "batch" and record.get("router_bucket") == "GREEN" and audit_passed:
            truth, source = record["annotator"], "consensus"
        else:
            if scope == "batch":
                without.append(doc_id)
            continue
        out.append(
            {
                "doc_id": doc_id,
                "truth": truth,
                "truth_source": source,
                **{name: record.get(name) for name in _RATER_NAMES},
            }
        )
    return {
        "scope": scope,
        "records": out,
        "documents_without_truth": sorted(without),
        "green_audit_passed": audit_passed,
    }


def unpaired_f1_delta(
    previous: Mapping[Any, Mapping[str, int]],
    current: Mapping[Any, Mapping[str, int]],
    *,
    samples: int = PAIRED_BOOTSTRAP_SAMPLES,
    seed: int = PAIRED_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """F1 change from one batch to the next: different documents, so each batch is
    resampled on its own (`paired_delta` is for the same documents twice)."""

    def counts(mapping):
        return [(int(v["tp"]), int(v["fp"]), int(v["fn"])) for _, v in sorted(mapping.items())]

    before, after = counts(previous), counts(current)
    if not before or not after:
        raise ContractError("unpaired_f1_delta needs documents on both sides")
    point = _table_prf(*map(sum, zip(*after)))[2] - _table_prf(*map(sum, zip(*before)))[2]
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        a = rng.choices(after, k=len(after))
        b = rng.choices(before, k=len(before))
        deltas.append(_table_prf(*map(sum, zip(*a)))[2] - _table_prf(*map(sum, zip(*b)))[2])
    deltas.sort()
    low, high = _interval(deltas)
    verdict = "improved" if low > 0 else "regressed" if high < 0 else "inconclusive"
    return {
        "delta": point,
        "lower_2_5": low,
        "upper_97_5": high,
        "probability_delta_gt_zero": sum(1 for d in deltas if d > 0) / samples,
        "previous_documents": len(before),
        "current_documents": len(after),
        "samples": samples,
        "seed": seed,
        "verdict": verdict,
    }
