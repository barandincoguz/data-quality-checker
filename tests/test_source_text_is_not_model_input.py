"""`source_text` is a locator and a database field, never model-served text.

The annotators clip their evidence from several places in a ruling, so
`source_text` is their prose about a document, not the document. It was
nonetheless reaching two models: the judge saw it on every annotator
candidate and none of M000's, which identified the human side outright, and
the trainee was served 324 of 2022 rows whose user turn was annotator spans
joined by newlines -- an input shape that never occurs at inference.

Both guards here are content assertions on purpose. `test_judges.py`'s
`test_blind_pilot_sends_only_text_and_candidates_and_persists_both_models`
already asserted the payload's *keys* and passed throughout the leak.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from data_quality_checker import context_windows
from data_quality_checker.context_windows import build_context_window_view
from data_quality_checker.judges import blind_candidates

ANNOTATOR_REFERENCE = {
    "kanun_no": "213",
    "kanun_ad": "Vergi Usul Kanunu",
    "madde": "mükerrer 20",
    "source_text": "mükerrer 20 nci maddesi",
}
MODEL_REFERENCE = {**ANNOTATOR_REFERENCE, "source_text": ""}

# The decision fields -- a law and an article (1 Oct 2026) -- written out rather than imported from
# `constants.REFERENCE_DECISION_FIELDS`. A guard that compared a payload
# against that tuple would still pass if the tuple itself ever regained
# `source_text` -- the exact class of defect this file exists to catch, and
# the reason the previous guard (which asserted the payload's keys but never
# a candidate's contents) let a broken judge pilot through. Writing them out
# also keeps this file collectable, so its behavioural failure is
# reproducible in the repo instead of only in a scratch copy.
DECISION_FIELDS = frozenset({"kanun_no", "kanun_ad", "madde"})


def test_judge_candidates_carry_only_the_decision_fields() -> None:
    """Round 1 measured the tell: 494 annotator rows carried `source_text`
    and 610 model rows carried none, so in the GREEN bucket -- where the five
    decision fields are identical by construction -- the judge chose the
    annotator in 17 of 18 decisive verdicts."""
    mapping, candidate_a, candidate_b = blind_candidates(
        batch_id="M01",
        internal_doc_id="d000001",
        model="gemma-4-31b-it-optiq-4bit",
        human_references=[dict(ANNOTATOR_REFERENCE)],
        model_references=[dict(MODEL_REFERENCE)],
    )
    assert mapping in {"A=human,B=model", "A=model,B=human"}
    for candidate in (candidate_a, candidate_b):
        for reference in candidate:
            assert set(reference) == DECISION_FIELDS
    # The two sides are now indistinguishable, which is what blind means.
    assert candidate_a == candidate_b


def test_blinding_survives_a_field_added_to_the_reference_schema() -> None:
    """A positive projection, not a `source_text` blacklist: a field added
    to the schema later must not leak into a comparison by default."""
    _, candidate_a, candidate_b = blind_candidates(
        batch_id="M01",
        internal_doc_id="d000002",
        model="gemma-4-31b-it-optiq-4bit",
        human_references=[{**ANNOTATOR_REFERENCE, "annotator_note": "kırpıldı"}],
        model_references=[dict(MODEL_REFERENCE)],
    )
    assert candidate_a == candidate_b
    assert all("annotator_note" not in reference for reference in candidate_a)


class _OffsetTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, list[tuple[int, int]]]:
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}


class _Tokenizer:
    def __init__(self) -> None:
        self._tokenizer = _OffsetTokenizer()

    def apply_chat_template(self, messages, **_):
        tokens: list[int] = []
        for message in messages:
            tokens.extend(range(len(str(message["content"]).split())))
        return tokens


def _source_row(user: str, references: list[dict[str, str]]) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": "extract"},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": json.dumps(references, ensure_ascii=False, separators=(",", ":")),
            },
        ]
    }


def test_every_training_user_turn_is_real_document_text(
    tmp_path: Path, monkeypatch
) -> None:
    """The trainee is served document prose or nothing.

    The document below is long enough to be windowed and carries two
    references far apart, which is exactly the shape that produced a
    dense-replay row: on round 0's 300 documents that shape accounted for
    324 of 2022 training rows whose user turn appeared nowhere in the ruling.
    """
    monkeypatch.setattr(context_windows, "_primary_window_geometry", lambda _: (6, 2))
    monkeypatch.setattr(context_windows, "FALLBACK_WINDOW_TOKENS", 3)
    monkeypatch.setattr(context_windows, "FALLBACK_OVERLAP_TOKENS", 1)
    body = "a0 a1 madde bir a4 a5 a6 a7 a8 a9 b0 b1 madde iki b4 b5 b6 b7"
    references = [
        {**ANNOTATOR_REFERENCE, "madde": "1", "source_text": "madde bir"},
        {**ANNOTATOR_REFERENCE, "madde": "2", "source_text": "madde iki"},
    ]
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(_source_row(body, references), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    doc_ids = tmp_path / "train_doc_ids.json"
    doc_ids.write_text("[1]\n", encoding="utf-8")
    output = tmp_path / "view.jsonl"

    manifest = build_context_window_view(
        source_path=source,
        source_doc_ids_path=doc_ids,
        output_path=output,
        output_doc_ids_path=tmp_path / "view_doc_ids.json",
        manifest_path=tmp_path / "view_manifest.json",
        tokenizer=_Tokenizer(),
        model_fingerprint="a" * 64,
        max_sequence_length=12,
    )
    assert manifest["windowed_document_count"] == 1
    assert manifest["reference_coverage"] == "complete"

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows
    for row in rows:
        user_turn = row["messages"][1]["content"]
        assert user_turn in body, f"user turn is not document text: {user_turn!r}"
        for reference in json.loads(row["messages"][2]["content"]):
            assert "source_text" not in reference
