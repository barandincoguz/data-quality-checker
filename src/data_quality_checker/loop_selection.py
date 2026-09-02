"""Per-round record of which checkpoint was selected and on what basis.

The record deliberately carries validation metrics only. The test split is
measured every round for the learning curve, but it may never influence a
decision -- checkpoint choice, learning rate, repeating a round, stopping the
loop. `write_selection_record` re-runs `select_checkpoint` itself and refuses
to write a record whose `selected` checkpoint disagrees with what the
selector actually chooses for these candidates, so the record cannot claim a
selection that didn't happen. `tests/test_loop_selection.py` asserts the
record's fields are exactly an allowed set (a new field -- such as a leaked
test-set metric -- fails the test until it is deliberately added to the
allowlist), and that a disagreeing `selected` is rejected.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .errors import ContractError
from .fingerprints import sha256_file
from .g0 import CHECKPOINT_TIE_BREAK_ORDER, CheckpointCandidate, select_checkpoint


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
) -> dict[str, Any]:
    if round_index < 0:
        raise ContractError("round_index must be non-negative")
    if selected not in candidates:
        raise ContractError("the selected checkpoint must be one of the candidates")
    recomputed = select_checkpoint(list(candidates), validation_documents=validation_documents)
    if recomputed != selected:
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
        "tie_break_order": list(CHECKPOINT_TIE_BREAK_ORDER),
        "selected": _row(selected),
        "candidates": [_row(candidate) for candidate in candidates],
    }
    path = output_dir / f"round_{round_index:03d}_checkpoint.json"
    write_json_atomic(path, payload)
    return {"path": str(path), "record_sha256": sha256_file(path)}
