from __future__ import annotations

import json
import re
from pathlib import Path

from data_quality_checker import context_windows
from data_quality_checker.context_windows import build_context_window_view


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
