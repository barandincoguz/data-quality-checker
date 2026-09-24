"""The GT750 source contract for the DQ-Loop.

`g0.validate_canonical_sources` pins GT v3 (500 documents, ids 1..500, its
manifest digest, G0's 394/50/50 split) and stays untouched: it is G0's
provenance. The GT750 loop gets its own contract, selected by the config's
`loop` block: 750 documents, a pinned manifest digest, and a split read
from a sealed SPLIT.json rather than regenerated here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import AppConfig
from .constants import EXAMPLE_BANK_SHA256, EXEMPLAR_DOC_IDS
from .contracts import validate_reference_list
from .errors import ContractError, IntegrityError
from .fingerprints import sha256_file
from .g0 import _example_doc_ids, repository_manifest_sha256

_SPLIT_KEYS = {"train": "train", "valid": "val", "test": "test"}


def has_loop_settings(config: AppConfig) -> bool:
    raw = getattr(config, "raw", None)
    return isinstance(raw, dict) and isinstance(raw.get("loop"), dict)


def _settings(config: AppConfig) -> dict[str, Any]:
    settings = config.raw.get("loop")
    if not isinstance(settings, dict):
        raise ContractError("config has no `loop` block")
    for key in ("split_path", "expected_documents", "sizes", "gt_manifest_sha256", "split_sha256"):
        if key not in settings:
            raise ContractError(f"loop settings are missing {key!r}")
    return settings


def loop_split_policy(config: AppConfig) -> str:
    sizes = _settings(config)["sizes"]
    return f"loop-gt750-{sizes['train']}-{sizes['valid']}-{sizes['test']}"


def _split_path(config: AppConfig, settings: dict[str, Any]) -> Path:
    path = Path(str(settings["split_path"])).expanduser()
    return path if path.is_absolute() else (config.source_path.parent / path).resolve()


def validate_loop_sources(config: AppConfig) -> dict[str, Any]:
    settings = _settings(config)
    gt_dir = config.canonical_gt_dir
    paths = sorted(gt_dir.glob("doc_*.json"))
    expected = int(settings["expected_documents"])
    if len(paths) != expected:
        raise ContractError(f"loop GT must contain {expected} doc_*.json files, found {len(paths)}")
    digest = repository_manifest_sha256(gt_dir)
    if digest != settings["gt_manifest_sha256"]:
        raise IntegrityError(f"loop GT manifest drift: expected {settings['gt_manifest_sha256']}, got {digest}")

    documents: dict[int, dict[str, Any]] = {}
    reference_count = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        doc_id = payload.get("doc_id")
        if not isinstance(doc_id, int) or path.name != f"doc_{doc_id}.json":
            raise ContractError(f"loop GT filename/doc_id mismatch in {path}")
        if not isinstance(payload.get("text"), str) or not payload["text"].strip():
            raise ContractError(f"loop GT document {doc_id} has no text")
        references = validate_reference_list(payload.get("references"))
        reference_count += len(references)
        documents[doc_id] = {**payload, "references": references}
    if set(documents) != set(range(1, expected + 1)):
        raise ContractError(f"loop GT doc_id coverage must be exactly 1..{expected}")

    split_path = _split_path(config, settings)
    if sha256_file(split_path) != settings["split_sha256"]:
        raise IntegrityError("loop split drift: SPLIT.json differs from its sealed sha256")
    raw_split = json.loads(split_path.read_text(encoding="utf-8"))
    split = {name: sorted(int(d) for d in raw_split[key]) for name, key in _SPLIT_KEYS.items()}
    everything = [d for ids in split.values() for d in ids]
    if len(everything) != len(set(everything)) or set(everything) != set(documents):
        raise ContractError("loop split sets overlap or do not cover the ground truth")
    counts = {name: len(ids) for name, ids in split.items()}
    if counts != settings["sizes"]:
        raise ContractError(f"loop split sizes {counts} differ from configured {settings['sizes']}")
    if not EXEMPLAR_DOC_IDS <= set(split["train"]):
        raise ContractError("a few-shot exemplar is outside train; it would leak into evaluation")

    if sha256_file(config.example_bank_path) != EXAMPLE_BANK_SHA256:
        raise IntegrityError("few-shot example bank checksum drift")
    examples = json.loads(config.example_bank_path.read_text(encoding="utf-8"))
    if _example_doc_ids(examples) != set(EXEMPLAR_DOC_IDS):
        raise ContractError("few-shot exemplar IDs drifted")

    return {
        "documents": documents,
        "split": split,
        "summary": {
            "canonical_gt_dir": str(gt_dir),
            "canonical_doc_count": len(documents),
            "canonical_reference_count": reference_count,
            "canonical_manifest_sha256": digest,
            "example_bank_path": str(config.example_bank_path),
            "example_bank_sha256": EXAMPLE_BANK_SHA256,
            "exemplar_doc_ids": sorted(EXEMPLAR_DOC_IDS),
            "split_manifest_path": str(split_path),
            "split_manifest_sha256": sha256_file(split_path),
            "split_counts": counts,
            "split": split,
        },
    }
