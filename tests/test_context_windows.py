from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from data_quality_checker import context_windows
from data_quality_checker.context_windows import (
    _dense_replay_rows,
    _expanded_interval,
    build_context_window_view,
)
from data_quality_checker.errors import ContractError


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
    monkeypatch.setattr(context_windows, "_primary_window_geometry", lambda _: (5, 1))
    monkeypatch.setattr(context_windows, "FALLBACK_WINDOW_TOKENS", 5)
    monkeypatch.setattr(context_windows, "FALLBACK_OVERLAP_TOKENS", 1)
    body = "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9"
    # Two things must be true at once or this test means nothing.
    #
    # Rescue must fire: no ordinary window may contain the span as a
    # substring, which a primary geometry of (5, 1) achieves for "w3 w4 w5".
    #
    # And the rescue window must be strictly WIDER than the span. With a
    # fallback as narrow as the span, the old bare-span row and a real
    # window are byte identical, and only the strategy-id string would
    # differ -- a label, not a behaviour. A fallback of 5 against the
    # 3-token span yields "w2 w3 w4 w5 w6".
    #
    # The budget is the whole three-message chat, not the user turn: this
    # reference row costs a fixed 4 tokens (system "extract" = 1, stripped
    # assistant target = 3, because "Vergi Usul Kanunu" holds two spaces the
    # toy tokenizer splits on). So a 5-token window needs 9, and a smaller
    # limit makes the builder raise "reference rescue tokens=N exceed
    # max_sequence_length" before any assertion runs.
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
        max_sequence_length=9,
    )
    assert manifest["reference_rescue_count"] >= 1
    assert manifest["request"]["reference_rescue"] == "real_window_v2"
    # The count alone doesn't prove rescue covered this document -- confirm a
    # chunk actually recorded that mode.
    rescue_chunks = [chunk for chunk in manifest["chunks"] if chunk["mode"] == "reference_rescue"]
    assert rescue_chunks

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    rescue_user = rows[rescue_chunks[0]["output_row_index"]]["messages"][1]["content"]
    # These three together are what the strategy id cannot fake: the served
    # turn is real document text, it contains the span, and it is not merely
    # the span. That last one is the whole point -- source_text located the
    # passage and did not become it.
    assert rescue_user in body
    assert "w3 w4 w5" in rescue_user
    assert rescue_user != "w3 w4 w5"
    # Every user turn served for this document is real document text.
    assert all(row["messages"][1]["content"] in body for row in rows)
    # And the reference is genuinely taught, not merely counted.
    assert any(json.loads(row["messages"][2]["content"]) for row in rows)


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


def test_dense_replay_serves_the_real_passage_spanning_its_pack(
    tmp_path: Path, monkeypatch
) -> None:
    """Replay teaches many references at once, from continuous ruling prose.

    The pack used to be served as the annotator's spans joined by newlines:
    324 of the sealed view's 2022 rows were input the model will never meet.
    Rebuilt on real spanning passages, 2041 reference instances survive
    across 506 rows (median pack 4, median passage 1834 characters) -- 34
    fewer than an earlier count of 2075, because the native/windowed sizing
    fix now serves seven documents whole, and a native document is never
    replayed.
    """
    monkeypatch.setattr(context_windows, "_primary_window_geometry", lambda _: (4, 2))
    monkeypatch.setattr(context_windows, "FALLBACK_WINDOW_TOKENS", 3)
    monkeypatch.setattr(context_windows, "FALLBACK_OVERLAP_TOKENS", 1)
    body = "a0 madde bir a3 a4 a5 madde iki a8 a9"
    references = [
        {
            "kanun_no": "213",
            "kanun_ad": "Vergi Usul Kanunu",
            "madde": "1",
            "fikra": "",
            "bent": "",
            "source_text": "madde bir",
        },
        {
            "kanun_no": "213",
            "kanun_ad": "Vergi Usul Kanunu",
            "madde": "2",
            "fikra": "",
            "bent": "",
            "source_text": "madde iki",
        },
    ]
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(_row(body, references), ensure_ascii=False) + "\n", encoding="utf-8"
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
        max_sequence_length=14,
    )
    assert manifest["dense_replay_row_count"] >= 1
    assert manifest["dense_replay_reference_count"] == 2
    assert manifest["request"]["dense_replay"] == "real_passage_span_pack_max10_full_v6"

    replay = [
        json.loads(line)
        for line, mode in zip(
            output.read_text(encoding="utf-8").splitlines(),
            [chunk["mode"] for chunk in manifest["chunks"]],
            strict=True,
        )
        if mode == "reference_dense_replay"
    ]
    assert replay
    for row in replay:
        assert row["messages"][1]["content"] in body


def test_dense_replay_splits_into_multiple_rows_when_one_pack_cannot_hold_them_all(
    tmp_path: Path, monkeypatch
) -> None:
    """A document whose references cannot all fit one pack must still teach
    every one of them -- split across more than one dense-replay row, not
    truncated to whatever the first pack could hold.

    Budget arithmetic (the toy tokenizer splits on whitespace; the system
    message "extract" always costs 1 token; a packed reference's stripped
    JSON -- `kanun_ad` is "Vergi Usul Kanunu" -- costs a fixed 2 tokens plus
    1 more per reference in the pack, i.e. 2n+1 for n references):

      pack [ref1, ref2]:  user "madde bir a3 a4 a5 madde iki" = 7 words
                          -> 1 (system) + 7 (user) + 5 (assistant, n=2) = 13
      pack [ref1, ref2, ref3]: user extends to "...a8 a9 madde uc" = 10 words
                          -> 1 + 10 + 7 (assistant, n=3) = 18

    max_sequence_length=13 admits the first pack but not the second, so the
    loop must close [ref1, ref2] and start a fresh pack rather than stall:
      pack [ref3] alone: user "madde uc" = 2 words
                          -> 1 + 2 + 3 (assistant, n=1) = 6
    """
    monkeypatch.setattr(context_windows, "_primary_window_geometry", lambda _: (4, 2))
    monkeypatch.setattr(context_windows, "FALLBACK_WINDOW_TOKENS", 3)
    monkeypatch.setattr(context_windows, "FALLBACK_OVERLAP_TOKENS", 1)
    body = "a0 madde bir a3 a4 a5 madde iki a8 a9 madde uc a12 a13"
    references = [
        {
            "kanun_no": "213",
            "kanun_ad": "Vergi Usul Kanunu",
            "madde": "1",
            "fikra": "",
            "bent": "",
            "source_text": "madde bir",
        },
        {
            "kanun_no": "213",
            "kanun_ad": "Vergi Usul Kanunu",
            "madde": "2",
            "fikra": "",
            "bent": "",
            "source_text": "madde iki",
        },
        {
            "kanun_no": "213",
            "kanun_ad": "Vergi Usul Kanunu",
            "madde": "3",
            "fikra": "",
            "bent": "",
            "source_text": "madde uc",
        },
    ]
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(_row(body, references), ensure_ascii=False) + "\n", encoding="utf-8"
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
        max_sequence_length=13,
    )

    # Confirm the split is a dense-replay pack split, not the document
    # merely having been cut into unrelated windowed chunks -- filter on
    # the mode explicitly.
    replay_chunks = [
        chunk for chunk in manifest["chunks"] if chunk["mode"] == "reference_dense_replay"
    ]
    assert len(replay_chunks) >= 2
    replay_reference_counts = [chunk["reference_count"] for chunk in replay_chunks]
    # Observed: [2, 1] -- nothing lost when the pack splits.
    assert sum(replay_reference_counts) == len(references) == 3

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    for chunk in replay_chunks:
        user_turn = rows[chunk["output_row_index"]]["messages"][1]["content"]
        assert user_turn in body


def test_dense_replay_raises_when_a_single_reference_row_exceeds_max_sequence_length() -> None:
    """A reference dense replay cannot fit in one row must be a loud
    contract failure -- not silently dropped from the pack.

    Budget arithmetic (the toy tokenizer splits on whitespace): the system
    message "extract" costs 1 token; the stripped assistant target for one
    reference (`kanun_ad` "Vergi Usul Kanunu") costs a fixed 3 tokens; a
    lone reference's dense-replay passage is exactly its own `source_text`
    span, here "madde bir iki uc dort" = 5 words. Total: 1 + 5 + 3 = 9.
    max_sequence_length=8 is one below that, so the very first (and only)
    pack must raise rather than silently ship an oversized row.
    """
    reference = {
        "kanun_no": "213",
        "kanun_ad": "Vergi Usul Kanunu",
        "madde": "5",
        "fikra": "",
        "bent": "",
        "source_text": "madde bir iki uc dort",
    }
    body = "pad0 pad1 madde bir iki uc dort pad7 pad8"

    with pytest.raises(ContractError) as excinfo:
        _dense_replay_rows(
            tokenizer=_Tokenizer(),
            system_message={"role": "system", "content": "extract"},
            body=body,
            references=[reference],
            max_sequence_length=8,
        )
    assert "dense replay row" in str(excinfo.value)


def test_a_reference_cited_in_two_places_is_in_both_windows(tmp_path: Path, monkeypatch) -> None:
    """GT v7 (1 Oct 2026) keeps one row per law and article and every quote of
    the rows it merged under `source_texts`. A window holding any of them holds
    the reference: reading only the first quote would teach the model that the
    other window does not cite it. Neither quote field reaches the target."""
    monkeypatch.setattr(context_windows, "_primary_window_geometry", lambda _: (4, 0))
    monkeypatch.setattr(context_windows, "FALLBACK_WINDOW_TOKENS", 2)
    monkeypatch.setattr(context_windows, "FALLBACK_OVERLAP_TOKENS", 1)
    monkeypatch.setattr(context_windows, "_include_dense_replay", lambda _: False)
    source, doc_ids = tmp_path / "train.jsonl", tmp_path / "train_doc_ids.json"
    reference = {
        "kanun_no": "193",
        "kanun_ad": "",
        "madde": "94",
        "source_text": "w1 w2",
        "source_texts": ["w1 w2", "w5 w6"],
    }
    source.write_text(
        json.dumps(_row("w0 w1 w2 w3 w4 w5 w6 w7", [reference])) + "\n", encoding="utf-8"
    )
    doc_ids.write_text("[1]\n", encoding="utf-8")
    build_context_window_view(
        source_path=source,
        source_doc_ids_path=doc_ids,
        output_path=tmp_path / "out.jsonl",
        output_doc_ids_path=tmp_path / "out_ids.json",
        manifest_path=tmp_path / "manifest.json",
        tokenizer=_Tokenizer(),
        model_fingerprint="a" * 64,
        max_sequence_length=8,
    )
    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    targets = [json.loads(row["messages"][-1]["content"]) for row in rows]
    assert [row["messages"][1]["content"] for row in rows] == ["w0 w1 w2 w3", "w4 w5 w6 w7"]
    assert targets == [[{"kanun_no": "193", "kanun_ad": "", "madde": "94"}]] * 2
