"""Per-round record of which checkpoint was selected and on what basis.

The record deliberately carries validation metrics only. The test split is
measured every round for the learning curve, but it may never influence a
decision -- checkpoint choice, learning rate, repeating a round, stopping the
loop. `write_selection_record` re-runs `select_checkpoint` itself (forwarding
`minimum_parse_count`, an input to the selection, so the record reflects the
same gate the caller actually used) and refuses to write a record whose
`selected` checkpoint disagrees with what the selector actually chooses for
these candidates. Agreement is judged on the declared `tie_break_order` key
`(core_f1, docwise_accuracy, recall, -validation_loss)`, not object identity:
a `selected` that exactly ties the recomputed winner on that key is accepted,
since the declared ordering cannot distinguish them, and only a candidate
that genuinely ranks lower is rejected. So the record cannot claim a
selection that didn't happen. `tests/test_loop_selection.py` asserts the
record's fields are exactly an allowed set (a new field -- such as a leaked
test-set metric -- fails the test until it is deliberately added to the
allowlist), and that a disagreeing `selected` is rejected.

If no candidate is eligible, `select_checkpoint` raises `GateBlocked` rather
than `ContractError` -- there genuinely is no selection to record, so this
function lets that propagate unconverted. A caller catching only
`ContractError` must also catch `GateBlocked` (both are `DQCheckError`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .errors import ContractError
from .fingerprints import sha256_file
from .g0 import CHECKPOINT_TIE_BREAK_ORDER, CheckpointCandidate, select_checkpoint


def _tie_break_key(candidate: CheckpointCandidate) -> tuple[float, ...]:
    """The declared ranking key, `CHECKPOINT_TIE_BREAK_ORDER` applied to `candidate`.

    Mirrors `select_checkpoint`'s own key construction so two candidates that
    tie on this key are indistinguishable under the ordering this record
    declares -- exactly the comparison `write_selection_record` needs to
    accept a tied `selected` rather than only an identical object.
    """
    key: list[float] = []
    for field in CHECKPOINT_TIE_BREAK_ORDER:
        descending, name = (True, field[1:]) if field.startswith("-") else (False, field)
        value = float(getattr(candidate, name))
        key.append(-value if descending else value)
    return tuple(key)


def _row(candidate: CheckpointCandidate) -> dict[str, Any]:
    return {
        "update": candidate.update,
        "coverage_count": candidate.coverage_count,
        "parse_count": candidate.parse_count,
        "empty_output_count": candidate.empty_output_count,
        "runaway_output_count": candidate.runaway_output_count,
        "core_f1": candidate.core_f1,
        "docwise_accuracy": candidate.docwise_accuracy,
        "recall": candidate.recall,
        "validation_loss": candidate.validation_loss,
    }


def write_selection_record(
    output_dir: Path,
    round_index: int,
    selected: CheckpointCandidate,
    candidates: Sequence[CheckpointCandidate],
    *,
    validation_documents: int,
    minimum_parse_count: int = 49,
) -> dict[str, Any]:
    if round_index < 0:
        raise ContractError("round_index must be non-negative")
    if selected not in candidates:
        raise ContractError("the selected checkpoint must be one of the candidates")
    recomputed = select_checkpoint(
        list(candidates),
        validation_documents=validation_documents,
        minimum_parse_count=minimum_parse_count,
    )
    if _tie_break_key(selected) != _tie_break_key(recomputed):
        raise ContractError(
            f"selected checkpoint (update {selected.update}) is not what "
            f"select_checkpoint chooses (update {recomputed.update}) for these candidates"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "round": round_index,
        "selection_basis": "validation",
        "validation_documents": validation_documents,
        "minimum_parse_count": minimum_parse_count,
        "tie_break_order": list(CHECKPOINT_TIE_BREAK_ORDER),
        "selected": _row(selected),
        "candidates": [_row(candidate) for candidate in candidates],
    }
    path = output_dir / f"round_{round_index:03d}_checkpoint.json"
    write_json_atomic(path, payload)
    return {"path": str(path), "record_sha256": sha256_file(path)}
