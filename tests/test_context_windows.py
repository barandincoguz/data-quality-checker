from __future__ import annotations

import json
import re
from pathlib import Path

from data_quality_checker import context_windows
from data_quality_checker.context_windows import _expanded_interval, build_context_window_view


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


def _row(user: str, references: list[dict[str, str]]) -> dict[str, object]:
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


def test_context_window_view_is_lossless_bounded_and_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(context_windows, "_primary_window_geometry", lambda _: (4, 2))
    monkeypatch.setattr(context_windows, "FALLBACK_WINDOW_TOKENS", 2)
    monkeypatch.setattr(context_windows, "FALLBACK_OVERLAP_TOKENS", 1)
    source = tmp_path / "train.jsonl"
    doc_ids = tmp_path / "train_doc_ids.json"
    output = tmp_path / "train_context.jsonl"
    output_ids = tmp_path / "train_context_doc_ids.json"
    manifest_path = tmp_path / "train_context_manifest.json"
    rows = [
        _row("short text", []),
        _row(
            "w0 w1 w2 w3 w4 w5 w6 w7 w8",
            [{"kanun_no": "1", "source_text": "w3 w4"}],
        ),
    ]
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    doc_ids.write_text("[10, 20]\n", encoding="utf-8")

    manifest = build_context_window_view(
        source_path=source,
        source_doc_ids_path=doc_ids,
        output_path=output,
        output_doc_ids_path=output_ids,
        manifest_path=manifest_path,
        tokenizer=_Tokenizer(),
        model_fingerprint="a" * 64,
        max_sequence_length=8,
    )

    assert manifest["source_document_count"] == 2
    assert manifest["native_document_count"] == 1
    assert manifest["windowed_document_count"] == 1
    assert manifest["maximum_tokens"] <= 8
    assert manifest["candidate_text_coverage"] == "complete_before_negative_sampling"
    assert manifest["reference_coverage"] == "complete"
    rendered_rows = [json.loads(line) for line in output.read_text().splitlines()]
    rendered_doc_ids = json.loads(output_ids.read_text(encoding="utf-8"))
    assert rendered_doc_ids[0] == 10
    assert all(
        json.loads(row["messages"][-1]["content"])
        for row, doc_id in zip(rendered_rows, rendered_doc_ids, strict=True)
        if doc_id == 20
    )
    # source_text locates a reference inside the ruling while a window is
    # chosen, but every assistant target actually served to the model -
    # native or windowed - must come out with the field stripped.
    assert all(
        "source_text" not in reference
        for row in rendered_rows
        for reference in json.loads(row["messages"][-1]["content"])
    )
    assert (
        build_context_window_view(
            source_path=source,
            source_doc_ids_path=doc_ids,
            output_path=output,
            output_doc_ids_path=output_ids,
            manifest_path=manifest_path,
            tokenizer=_Tokenizer(),
            model_fingerprint="a" * 64,
            max_sequence_length=8,
        )
        == manifest
    )


def test_native_document_row_strips_source_text_from_assistant_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "train.jsonl"
    doc_ids = tmp_path / "train_doc_ids.json"
    output = tmp_path / "train_context.jsonl"
    output_ids = tmp_path / "train_context_doc_ids.json"
    manifest_path = tmp_path / "train_context_manifest.json"
    reference = {
        "kanun_no": "213",
        "kanun_ad": "Vergi Usul Kanunu",
        "madde": "mükerrer 20",
        "fikra": "",
        "bent": "",
        "source_text": "mükerrer 20 nci maddesi",
    }
    row = _row("kısa metin mükerrer 20 nci maddesi", [reference])
    source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    doc_ids.write_text("[7]\n", encoding="utf-8")

    manifest = build_context_window_view(
        source_path=source,
        source_doc_ids_path=doc_ids,
        output_path=output,
        output_doc_ids_path=output_ids,
        manifest_path=manifest_path,
        tokenizer=_Tokenizer(),
        model_fingerprint="c" * 64,
        max_sequence_length=64,
    )

    assert manifest["native_document_count"] == 1
    rendered_rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rendered_rows) == 1
    target = json.loads(rendered_rows[0]["messages"][-1]["content"])
    assert target == [
        {
            "kanun_no": "213",
            "kanun_ad": "Vergi Usul Kanunu",
            "madde": "mükerrer 20",
            "fikra": "",
            "bent": "",
        }
    ]


def test_a_string_document_id_survives_the_view_unchanged(tmp_path: Path) -> None:
    """The DQ-Loop trains on canonical documents (integer ids, resolvable in the
    ground-truth directory) mixed with round-cleaned documents whose ids look
    like ``d000001``. The view only ever needs a stable key -- for the replay
    sample and for the per-chunk record -- so it must carry whatever id it was
    given through untouched rather than force it to be a number.
    """
    source = tmp_path / "train.jsonl"
    doc_ids = tmp_path / "train_doc_ids.json"
    output = tmp_path / "train_context.jsonl"
    output_ids = tmp_path / "train_context_doc_ids.json"
    manifest_path = tmp_path / "train_context_manifest.json"
    rows = [_row("kısa metin", []), _row("baska kısa metin", [])]
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    doc_ids.write_text('[7, "d000001"]\n', encoding="utf-8")

    build_context_window_view(
        source_path=source,
        source_doc_ids_path=doc_ids,
        output_path=output,
        output_doc_ids_path=output_ids,
        manifest_path=manifest_path,
        tokenizer=_Tokenizer(),
        model_fingerprint="d" * 64,
        max_sequence_length=64,
    )

    assert json.loads(output_ids.read_text(encoding="utf-8")) == [7, "d000001"]


def test_native_decision_ignores_the_source_text_it_is_about_to_strip(
    tmp_path: Path,
) -> None:
    """A document is windowed only if what the model is served is too long.

    The sizing count used to include `source_text`, a field the very next
    statement strips. On the sealed 300-document split that spent a median
    132 tokens per document on text the model never sees and fragmented 7
    documents that fit whole.
    """
    reference = {
        "kanun_no": "213",
        "kanun_ad": "Vergi Usul Kanunu",
        "madde": "5",
        "fikra": "",
        "bent": "",
        "source_text": "w0 w1 w2 w3 w4 w5 w6 w7",
    }
    row = _row("w0 w1 w2 w3 w4 w5 w6 w7", [reference])
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    doc_ids = tmp_path / "train_doc_ids.json"
    doc_ids.write_text("[1]\n", encoding="utf-8")

    manifest = build_context_window_view(
        source_path=source,
        source_doc_ids_path=doc_ids,
        output_path=tmp_path / "view.jsonl",
        output_doc_ids_path=tmp_path / "view_doc_ids.json",
        manifest_path=tmp_path / "view_manifest.json",
        tokenizer=_Tokenizer(),
        # 15 is chosen, not arbitrary: this row renders to 19 tokens with
        # `source_text` in the assistant target and 12 without it. Any limit
        # from 12 to 18 therefore separates the two; at 19 or above the
        # document fits either way and the test passes without measuring
        # anything. 22 was the original value and was exactly that mistake.
        max_sequence_length=15,
        model_fingerprint="a" * 64,
    )
    assert manifest["native_document_count"] == 1
    assert manifest["windowed_document_count"] == 0


def test_reference_rescue_serves_a_real_window_not_the_annotator_span(
    tmp_path: Path, monkeypatch
) -> None:
    """A reference no window makes visible is still taught -- from the
    ruling. The rescue row used to serve the annotator's bare span as the
    user turn, which is prose about the document, not the document."""
    monkeypatch.setattr(context_windows, "_primary_window_geometry", lambda _: (4, 2))
    monkeypatch.setattr(context_windows, "FALLBACK_WINDOW_TOKENS", 3)
    monkeypatch.setattr(context_windows, "FALLBACK_OVERLAP_TOKENS", 1)
    body = "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9"
    # The span straddles the primary window boundary, so no window contains
    # it as a substring and rescue is the only path that can cover it -- but
    # only once the primary window (2, 6) is actually forced to fall back.
    # That window renders at 8 tokens and already covers the span by
    # overlap, so a limit of 12 (as loose prose describing this fixture
    # might suggest) never exercises rescue at all. 7 is the only value
    # that forces (2, 6) to fall back to sub-windows too narrow to contain
    # the span (5 and 4 tokens, verified below) while still admitting the
    # 7-token rescue row this test expects.
    reference = {
        "kanun_no": "213",
        "kanun_ad": "Vergi Usul Kanunu",
        "madde": "5",
        "fikra": "",
        "bent": "",
        "source_text": "w3 w4 w5",
    }
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(_row(body, [reference]), ensure_ascii=False) + "\n", encoding="utf-8"
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
        max_sequence_length=7,
    )
    assert manifest["reference_rescue_count"] >= 1
    assert manifest["reference_coverage"] == "complete"
    assert manifest["request"]["reference_rescue"] == "real_window_v2"
    # The count alone doesn't prove rescue is what covered this document --
    # confirm a chunk actually recorded that mode.
    assert any(chunk["mode"] == "reference_rescue" for chunk in manifest["chunks"])

    rescue_rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["messages"][1]["content"] in body for row in rescue_rows)
    assert any(
        "w3 w4 w5" in row["messages"][1]["content"]
        and json.loads(row["messages"][2]["content"])
        for row in rescue_rows
    )


def test_expanded_interval_contains_the_original_span_and_stays_in_bounds() -> None:
    """`_expanded_interval` grows a token span to a minimum width. Both
    invariants matter for real reasons: leaving `[0, total)` would index
    past the token offsets, and failing to contain the original span would
    build a rescue window that does not actually contain the reference it
    was built for -- exactly what the `IntegrityError` in the rescue loop
    guards against on real data.
    """
    cases = [
        # (start, end, total, minimum_tokens) -- the awkward edges named by
        # the task: span at index 0, span at the very end, a body shorter
        # than minimum_tokens, and a span already wider than minimum_tokens.
        (0, 1, 10, 4),  # span at index 0
        (0, 2, 5, 4),  # span at index 0, body barely fits minimum_tokens
        (8, 10, 10, 4),  # span at the very end
        (9, 10, 10, 4),  # single-token span at the very end
        (1, 2, 3, 5),  # body shorter than minimum_tokens
        (0, 1, 1, 5),  # body of a single token, far shorter than minimum
        (2, 9, 10, 3),  # span already wider than minimum_tokens
        (0, 10, 10, 3),  # span already spans the entire body
        (5, 6, 10, 1),  # minimum_tokens smaller than the span itself
    ]
    for start, end, total, minimum_tokens in cases:
        begin, stop = _expanded_interval(start, end, total, minimum_tokens)
        assert 0 <= begin, (start, end, total, minimum_tokens, begin, stop)
        assert stop <= total, (start, end, total, minimum_tokens, begin, stop)
        assert begin < stop, (start, end, total, minimum_tokens, begin, stop)
        assert begin <= start, (start, end, total, minimum_tokens, begin, stop)
        assert end <= stop, (start, end, total, minimum_tokens, begin, stop)

    # A broader systematic sweep of the same two invariants -- not just the
    # named edges -- since the brief's own integration test only samples one
    # geometry and could easily pass by luck.
    for total in range(1, 13):
        for minimum_tokens in range(1, 13):
            for start in range(total):
                for end in range(start + 1, total + 1):
                    begin, stop = _expanded_interval(start, end, total, minimum_tokens)
                    assert 0 <= begin <= start
                    assert end <= stop <= total
                    assert begin < stop
