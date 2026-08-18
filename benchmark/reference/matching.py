"""Order-independent optimal reference matching used by metric profile ``strict-v2``.

The legacy core matcher in :mod:`benchmark.reference.evaluate` pairs gold and
predicted references with a staged first-fit greedy walk. That walk is correct on
the current corpus but it is not a maximum-cardinality matching, its result can
depend on row order, and it silently drops gold rows that carry a law identity
without a ``madde``.

This module implements the replacement semantics:

* **Symmetric eligibility.** A row is eligible when it carries a law identity.
  Rows without ``madde`` are *law-level references* and are matched against
  law-level predictions instead of being discarded on the gold side and charged
  as false positives on the prediction side.
* **Unification-based compatibility.** Two rows are compatible when they do not
  conflict on any law field that both sides populate and they agree on at least
  one populated field. This replaces the ``triplet`` / ``no_pair`` / ``ad_pair``
  tier ladder plus its one-directional fallbacks with a single symmetric rule.
* **Optimal matching.** Pairing is a minimum-cost maximum-flow assignment, so the
  true-positive count is the maximum-cardinality matching of the compatibility
  graph -- a graph invariant, hence independent of row order. Ties between
  equal-cardinality matchings are broken by a deterministic quality score so the
  reported ``fikra``/``bent`` diagnostics are also stable.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from annotator.reference.common import GENERIC_LAW_NAMES, normalize_reference

#: A single letter after ``/`` at the end of an article number. Its *case*
#: decides what it means in Turkish legal citation:
#:
#: * **lowercase** -- a subparagraph (*bent*), as in KDV Kanunu ``13/a``.
#:   Subparagraphs are normally written ``(a) bendi``, ``a)`` or ``5/1-a``.
#: * **UPPERCASE** -- part of the article number itself. ``32/A`` (KVK indirimli
#:   kurumlar vergisi), ``38/A``, ``39/A``, ``14/A``, ``3/A`` and ``3/B`` are
#:   separate articles whose number happens to contain a letter; the correct
#:   reading is "32/A maddesi", never "32. maddenin A bendi". In GT v3 half of
#:   those rows carry a ``fikra`` of their own, which corroborates this.
RE_ARTICLE_LETTER_SUFFIX = re.compile(
    r"^(?P<head>.+?)/(?P<suffix>[A-Za-zÇĞİÖŞÜçğıöşü])$"
)

#: ``466-469`` and ``466 ila 469`` are the same span written two ways.
RE_NUMERIC_ARTICLE_RANGE = re.compile(r"^(?P<start>\d+)\s*-\s*(?P<end>\d+)$")


def _fold_article_text(text: str) -> str:
    """Case-fold an article string and canonicalize its range spelling."""
    return RE_NUMERIC_ARTICLE_RANGE.sub(r"\g<start> ila \g<end>", text).casefold()


def split_article_subparagraph(
    madde: str,
    fikra: str,
    bent: str,
) -> tuple[str, str, str]:
    """Move a *lowercase* subparagraph letter out of ``madde`` into ``bent``.

    GT v3 stores KDV ``13/a`` style citations entirely inside ``madde``, which
    silently gives those rows strict subparagraph matching while every other row
    treats ``bent`` as soft detail. Splitting them restores one uniform rule.

    An uppercase suffix is never split: ``32/A`` is a whole article number, not
    article 32 subparagraph A. A row that already carries a ``bent`` is left
    alone too, so no existing subparagraph can be overwritten.
    """
    match = RE_ARTICLE_LETTER_SUFFIX.match(madde)
    if match and match.group("suffix").islower() and not bent:
        return match.group("head"), fikra, match.group("suffix")
    return madde, fikra, bent


def canonical_article_fields(ref: dict[str, str]) -> tuple[str, str, str]:
    """Return ``(madde, fikra, bent)`` in a comparison-stable canonical form.

    Folds the encoding inconsistencies found in GT v3: the subparagraph split
    above, the ``466-469`` / ``466 ila 469`` range spelling, and the
    ``mükerrer 257`` / ``Mükerrer 257`` capitalisation split that made 88 gold
    rows fail to compare equal to their own duplicates.

    The letter suffix of an article number keeps its original case, which is
    what makes this function idempotent. Folding it would rewrite ``32/A`` to
    ``32/a``, and a second application would then mistake that for a
    subparagraph and destroy the article number.
    """
    madde, fikra, bent = split_article_subparagraph(
        ref.get("madde", ""), ref.get("fikra", ""), ref.get("bent", "")
    )
    match = RE_ARTICLE_LETTER_SUFFIX.match(madde)
    if match:
        madde = f"{_fold_article_text(match.group('head'))}/{match.group('suffix')}"
    else:
        madde = _fold_article_text(madde)
    return (madde, fikra.casefold(), bent.casefold())


def canonical_article_key(ref: dict[str, str]) -> str:
    """Return the article identity used by the core metric."""
    return canonical_article_fields(ref)[0]


def canonicalize_reference(reference: dict[str, str]) -> dict[str, str]:
    """Normalize a reference and rewrite its article fields into canonical form.

    Applied once per row before matching so deduplication, pairing and the
    ``fikra``/``bent`` diagnostics all observe the same values. The operation is
    idempotent.
    """
    normalized = normalize_reference(reference)
    madde, fikra, bent = canonical_article_fields(normalized)
    normalized["madde"] = madde
    normalized["fikra"] = fikra
    normalized["bent"] = bent
    return normalized

#: Cardinality must dominate every quality bonus so the assignment stays maximum.
CARDINALITY_WEIGHT = 1000

LAW_LEVEL_KIND = "law_level"
LAW_ARTICLE_KIND = "law_article"


@dataclass(frozen=True)
class MatchEntry:
    """One reference prepared for optimal matching."""

    ref: dict[str, str]
    kind: str


def empty_optimal_diagnostics() -> dict[str, int]:
    """Initialize flat count diagnostics for the optimal matcher."""
    return {
        "eligible_gold_count": 0,
        "eligible_gold_law_article_count": 0,
        "eligible_gold_law_level_count": 0,
        "eligible_prediction_count": 0,
        "eligible_prediction_law_article_count": 0,
        "eligible_prediction_law_level_count": 0,
        "excluded_gold_missing_law_count": 0,
        "invalid_prediction_missing_law_count": 0,
        "matched_law_article_count": 0,
        "matched_law_level_count": 0,
        "matched_exact_law_identity_count": 0,
        "matched_with_missing_pred_kanun_no_count": 0,
        "matched_with_missing_pred_kanun_ad_count": 0,
        "matched_with_extra_pred_kanun_no_count": 0,
        "matched_with_extra_pred_kanun_ad_count": 0,
        "deduplicated_gold_rows": 0,
        "deduplicated_prediction_rows": 0,
    }


def effective_law_name(ref: dict[str, str]) -> str:
    """Return ``kanun_ad`` unless it is a bare placeholder such as ``"Kanun"``.

    ``normalize_kanun_adi`` falls back to the literal ``"Kanun"`` when a row names
    no law and its number is outside :data:`CANONICAL_LAW_BY_NO`. That placeholder
    carries no identity, so treating it as a distinct name would both split one
    law into two references and make deduplication depend on row order.
    """
    kanun_ad = ref.get("kanun_ad", "")
    return "" if kanun_ad.strip().lower() in GENERIC_LAW_NAMES else kanun_ad


def core_identity_key(ref: dict[str, str]) -> tuple[str, str, str]:
    """Return the article-level identity used for *matching* and reporting."""
    return (ref.get("kanun_no", ""), effective_law_name(ref), canonical_article_key(ref))


def reference_identity_key(ref: dict[str, str]) -> tuple[str, str, str, str, str]:
    """Return the full reference identity, including ``fikra`` and ``bent``.

    Not used for core deduplication -- ``core_law_article_strict`` scores at
    article level, so VUK ``17/4-a`` and ``17/1-b`` are one core reference. Kept
    for callers that need reference-level rather than article-level identity.
    """
    madde, fikra, bent = canonical_article_fields(ref)
    return (ref.get("kanun_no", ""), effective_law_name(ref), madde, fikra, bent)


def _survivor_rank(row: dict[str, str]) -> tuple[int, int, int, tuple[str, ...]]:
    """Rank duplicate rows sharing one identity; the highest rank survives.

    The final component is the row's own content, never its position. Ranking by
    insertion order is what made the upstream ``compact_law_references`` survivor
    choice depend on row order and leak that nondeterminism into the metric.
    """
    status = str(row.get("status", "")).strip().lower()
    status_score = 2 if status == "approved" else 1 if status else 0
    detail_score = sum(1 for field in ("fikra", "bent") if row.get(field))
    source_score = 1 if row.get("source_text") else 0
    content = (
        row.get("fikra", ""),
        row.get("bent", ""),
        row.get("kanun_no", ""),
        row.get("kanun_ad", ""),
        row.get("source_text", ""),
    )
    return (status_score, detail_score, source_score, content)


def canonical_sort_key(ref: dict[str, str]) -> tuple[str, ...]:
    """Total order over references, used to make the prepared set positional-free."""
    return (
        ref.get("kanun_no", ""),
        effective_law_name(ref),
        ref.get("madde", ""),
        ref.get("fikra", ""),
        ref.get("bent", ""),
        ref.get("source_text", ""),
    )


def prepare_reference_set(
    references: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    """Build the deterministic reference set the optimal matcher scores.

    The optimal engine owns this whole chain rather than consuming
    ``--reference-postprocess`` / ``--core-reference-view`` output, because that
    upstream path selects duplicate survivors by row order and therefore is not
    reproducible under permutation. Two further consequences are deliberate:

    * **No generic suppression.** Upstream drops a law-level row whenever the
      same law also has an article-level row. On GT v3 that rule deletes zero
      gold rows but removes 385 Regex and 144 spaCy prediction rows while
      touching none of the seven LLM methods, i.e. it hands the two rule-based
      baselines an advantage no other method gets. With a law-level tier those
      rows are scoreable claims, so they are kept and scored on both sides.
    * **Canonical ordering.** The returned rows are sorted by
      :func:`canonical_sort_key`, so edge construction in the assignment problem
      cannot depend on input order either.

    Returns ``(rows, collapsed_row_count)``.
    """
    rows = [canonicalize_reference(ref) for ref in references]
    rows, deduplicated = deduplicate_core_identities(rows)
    rows, absorbed = absorb_subsumed_identities(rows)
    rows.sort(key=canonical_sort_key)
    return rows, deduplicated + absorbed


def deduplicate_core_identities(
    references: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    """Collapse rows sharing the article-level identity ``(kanun_no, kanun_ad, madde)``.

    Matches the article-level semantics of ``core_law_article_strict``: VUK
    ``17/4-a`` and ``17/1-b`` are one core reference, and after the subparagraph
    split so are KDV ``13/a`` and ``13/d``. The subparagraph itself is still
    scored, in the soft ``fikra``/``bent`` diagnostics. Unlike the legacy
    ``core_set`` view this also folds law-level rows, so duplicates are removed
    on both sides regardless of whether ``madde`` is set.
    """
    normalized = [canonicalize_reference(ref) for ref in references]
    kept: list[dict[str, str]] = []
    position: dict[tuple[str, str, str], int] = {}
    ranks: dict[tuple[str, str, str], tuple[int, int, int, tuple[str, ...]]] = {}
    removed = 0

    for row in normalized:
        key = core_identity_key(row)
        existing = position.get(key)
        rank = _survivor_rank(row)
        if existing is None:
            position[key] = len(kept)
            ranks[key] = rank
            kept.append(row)
            continue
        removed += 1
        if rank > ranks[key]:
            kept[existing] = row
            ranks[key] = rank
    return kept, removed


def _identity_fields(ref: dict[str, str]) -> tuple[str, str]:
    """Return the populated law-identity fields of a reference."""
    return (ref.get("kanun_no", ""), effective_law_name(ref))


def _subsumes(specific: dict[str, str], general: dict[str, str]) -> bool:
    """Return whether `specific` states everything `general` does, and more."""
    specific_no, specific_ad = _identity_fields(specific)
    general_no, general_ad = _identity_fields(general)
    if general_no and general_no != specific_no:
        return False
    if general_ad and general_ad != specific_ad:
        return False
    return (bool(specific_no) + bool(specific_ad)) > (bool(general_no) + bool(general_ad))


def absorb_subsumed_identities(
    references: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    """Fold under-specified rows into the fully-specified row for the same law.

    A row naming only ``kanun_no`` for article 5 and another naming both the
    number and the law for article 5 are one reference, not two. The absorbing
    row must be unique, so genuinely ambiguous cases are left untouched. Because
    the decision is computed against the whole input set rather than a running
    survivor, the result does not depend on row order.
    """
    kept: list[dict[str, str]] = []
    removed = 0
    for index, row in enumerate(references):
        absorbers = [
            other_index
            for other_index, other in enumerate(references)
            if other_index != index
            and canonical_article_fields(other) == canonical_article_fields(row)
            and _subsumes(other, row)
        ]
        if len(absorbers) == 1:
            removed += 1
            continue
        kept.append(row)
    return kept, removed


def classify_entry(
    ref: dict[str, str],
    diagnostics: dict[str, int],
    *,
    gold: bool,
) -> MatchEntry | None:
    """Classify a reference as law-article, law-level, or ineligible."""
    kanun_no = ref.get("kanun_no", "")
    kanun_ad = effective_law_name(ref)
    if not kanun_no and not kanun_ad:
        key = ("excluded_gold_missing_law_count" if gold
               else "invalid_prediction_missing_law_count")
        diagnostics[key] += 1
        return None

    kind = LAW_ARTICLE_KIND if ref.get("madde", "") else LAW_LEVEL_KIND
    side = "gold" if gold else "prediction"
    diagnostics[f"eligible_{side}_count"] += 1
    diagnostics[f"eligible_{side}_{kind}_count"] += 1
    return MatchEntry(ref=ref, kind=kind)


def law_identity_compatible(gold_ref: dict[str, str], pred_ref: dict[str, str]) -> bool:
    """Unify two law identities.

    Compatible when no field populated on *both* sides conflicts and at least one
    such field agrees. This is symmetric, so a prediction that omits ``kanun_no``
    but names the law correctly is no longer punished, and neither is one that
    supplies a number the gold row left blank.
    """
    agreed = False
    for gold_value, pred_value in (
        (gold_ref.get("kanun_no", ""), pred_ref.get("kanun_no", "")),
        (effective_law_name(gold_ref), effective_law_name(pred_ref)),
    ):
        if gold_value and pred_value:
            if gold_value != pred_value:
                return False
            agreed = True
    return agreed


def references_compatible(gold_entry: MatchEntry, pred_entry: MatchEntry) -> bool:
    """Return whether a gold and predicted reference may be paired."""
    if gold_entry.kind != pred_entry.kind:
        return False
    if canonical_article_key(gold_entry.ref) != canonical_article_key(pred_entry.ref):
        return False
    return law_identity_compatible(gold_entry.ref, pred_entry.ref)


def match_quality(gold_ref: dict[str, str], pred_ref: dict[str, str]) -> int:
    """Deterministic tie-breaking bonus among equal-cardinality matchings.

    Prefers pairs whose optional ``fikra``/``bent`` detail agrees and whose law
    identity is stated with the same completeness on both sides.
    """
    quality = 0
    for field in ("fikra", "bent"):
        if gold_ref.get(field, "") == pred_ref.get(field, ""):
            quality += 4
    for field in ("kanun_no", "kanun_ad"):
        if bool(gold_ref.get(field, "")) == bool(pred_ref.get(field, "")):
            quality += 1
    return quality


class _MinCostFlow:
    """Successive-shortest-path min-cost max-flow over unit-capacity edges."""

    def __init__(self, node_count: int) -> None:
        self.graph: list[list[list[int]]] = [[] for _ in range(node_count)]

    def add_edge(self, src: int, dst: int, capacity: int, cost: int) -> None:
        """Add a directed edge and its residual counterpart."""
        forward = [dst, capacity, cost, len(self.graph[dst])]
        backward = [src, 0, -cost, len(self.graph[src])]
        self.graph[src].append(forward)
        self.graph[dst].append(backward)

    def run(self, source: int, sink: int) -> None:
        """Push flow until the maximum is reached, minimizing cost at each step."""
        node_count = len(self.graph)
        while True:
            distance = [None] * node_count
            distance[source] = 0
            in_queue = [False] * node_count
            previous: list[tuple[int, int] | None] = [None] * node_count
            queue = [source]
            in_queue[source] = True

            while queue:
                node = queue.pop(0)
                in_queue[node] = False
                for edge_index, edge in enumerate(self.graph[node]):
                    dst, capacity, cost, _ = edge
                    if capacity <= 0:
                        continue
                    candidate = distance[node] + cost
                    if distance[dst] is None or candidate < distance[dst]:
                        distance[dst] = candidate
                        previous[dst] = (node, edge_index)
                        if not in_queue[dst]:
                            in_queue[dst] = True
                            queue.append(dst)

            if distance[sink] is None:
                return

            node = sink
            while node != source:
                parent, edge_index = previous[node]
                edge = self.graph[parent][edge_index]
                edge[1] -= 1
                self.graph[node][edge[3]][1] += 1
                node = parent


def optimal_matching(
    gold_entries: list[MatchEntry],
    pred_entries: list[MatchEntry],
) -> list[tuple[int, int]]:
    """Return a maximum-cardinality, maximum-quality gold/prediction pairing.

    The returned pair count equals the maximum matching of the compatibility
    graph, which is a property of the graph rather than of row order.
    """
    if not gold_entries or not pred_entries:
        return []

    gold_count = len(gold_entries)
    pred_count = len(pred_entries)
    source = 0
    sink = 1 + gold_count + pred_count
    flow = _MinCostFlow(sink + 1)

    for gold_index in range(gold_count):
        flow.add_edge(source, 1 + gold_index, 1, 0)
    for pred_index in range(pred_count):
        flow.add_edge(1 + gold_count + pred_index, sink, 1, 0)

    edge_slots: dict[tuple[int, int], int] = {}
    for gold_index, gold_entry in enumerate(gold_entries):
        for pred_index, pred_entry in enumerate(pred_entries):
            if not references_compatible(gold_entry, pred_entry):
                continue
            cost = -(CARDINALITY_WEIGHT + match_quality(gold_entry.ref, pred_entry.ref))
            edge_slots[(gold_index, pred_index)] = len(flow.graph[1 + gold_index])
            flow.add_edge(1 + gold_index, 1 + gold_count + pred_index, 1, cost)

    flow.run(source, sink)

    matched: list[tuple[int, int]] = []
    for (gold_index, pred_index), slot in edge_slots.items():
        if flow.graph[1 + gold_index][slot][1] == 0:
            matched.append((gold_index, pred_index))
    matched.sort()
    return matched


def _accumulate_identity_diagnostics(
    diagnostics: dict[str, int],
    gold_entry: MatchEntry,
    pred_entry: MatchEntry,
) -> None:
    """Record how a matched pair's law identity differed, if at all."""
    diagnostics[f"matched_{gold_entry.kind}_count"] += 1
    gold_ref, pred_ref = gold_entry.ref, pred_entry.ref
    if core_identity_key(gold_ref) == core_identity_key(pred_ref):
        diagnostics["matched_exact_law_identity_count"] += 1
        return
    for field, label in (("kanun_no", "kanun_no"), ("kanun_ad", "kanun_ad")):
        if gold_ref.get(field, "") and not pred_ref.get(field, ""):
            diagnostics[f"matched_with_missing_pred_{label}_count"] += 1
        elif pred_ref.get(field, "") and not gold_ref.get(field, ""):
            diagnostics[f"matched_with_extra_pred_{label}_count"] += 1


def evaluate_doc_optimal(
    pred_refs: list[dict[str, str]],
    gold_refs: list[dict[str, str]],
) -> tuple[int, int, int, dict[str, int], list[tuple[dict[str, str], dict[str, str]]]]:
    """Evaluate one document with the optimal matcher.

    Returns ``(tp, fp, fn, diagnostics, matched_pairs)``.
    """
    diagnostics = empty_optimal_diagnostics()

    gold_rows, collapsed_gold = prepare_reference_set(gold_refs)
    pred_rows, collapsed_pred = prepare_reference_set(pred_refs)
    diagnostics["deduplicated_gold_rows"] += collapsed_gold
    diagnostics["deduplicated_prediction_rows"] += collapsed_pred

    gold_entries = [
        entry
        for row in gold_rows
        if (entry := classify_entry(row, diagnostics, gold=True)) is not None
    ]
    pred_entries = [
        entry
        for row in pred_rows
        if (entry := classify_entry(row, diagnostics, gold=False)) is not None
    ]

    matched = optimal_matching(gold_entries, pred_entries)
    matched_pairs = []
    for gold_index, pred_index in matched:
        gold_entry = gold_entries[gold_index]
        pred_entry = pred_entries[pred_index]
        _accumulate_identity_diagnostics(diagnostics, gold_entry, pred_entry)
        matched_pairs.append((gold_entry.ref, pred_entry.ref))

    tp = len(matched)
    fp = (
        len(pred_entries) - tp
        + diagnostics["invalid_prediction_missing_law_count"]
    )
    fn = len(gold_entries) - tp
    return tp, fp, fn, diagnostics, matched_pairs


def summarize_kind_counts(references: list[dict[str, str]]) -> Counter[str]:
    """Count law-article vs law-level rows; used by reporting helpers."""
    counts: Counter[str] = Counter()
    for ref in references:
        normalized = normalize_reference(ref)
        if not normalized.get("kanun_no") and not normalized.get("kanun_ad"):
            counts["ineligible"] += 1
        elif normalized.get("madde"):
            counts[LAW_ARTICLE_KIND] += 1
        else:
            counts[LAW_LEVEL_KIND] += 1
    return counts


def optimal_diagnostics_payload(diagnostics: dict[str, int]) -> dict[str, Any]:
    """Add derived rates to the optimal-matcher diagnostics."""
    payload: dict[str, Any] = dict(diagnostics)
    matched_total = (
        diagnostics["matched_law_article_count"] + diagnostics["matched_law_level_count"]
    )
    payload["matched_total"] = matched_total
    payload["matched_exact_law_identity_rate"] = (
        diagnostics["matched_exact_law_identity_count"] / matched_total
        if matched_total
        else 0.0
    )
    return payload
