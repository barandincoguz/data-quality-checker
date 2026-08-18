"""Immutable v1 data and model contracts."""

from __future__ import annotations

SCHEMA_VERSION = 1

MODEL_ID = "mlx-community/Qwen3.5-9B-MLX-4bit"
MODEL_REVISION = "938d8919941c6e7efd3c7150eff7fe9d12afa631"
MINIMUM_MLX_LM = "0.31.0"

CANONICAL_GT_MANIFEST_SHA256 = "5367d50e3a9ff56f1253a54fe0d76d4ca2812939c183595142b99524a5afaf0c"
EXAMPLE_BANK_SHA256 = "89281d9b15bd522da835d8d30924323375ddd3f6a6207ea010eaa8f600b36f05"
EXEMPLAR_DOC_IDS = frozenset({1, 10, 16, 18, 36, 77})

REFERENCE_FIELDS = (
    "kanun_no",
    "kanun_ad",
    "madde",
    "fikra",
    "bent",
    "source_text",
)

ROUTER_BUCKETS = frozenset({"GREEN", "YELLOW", "RED", "QUARANTINE"})
REVIEW_STATUSES = frozenset({"pending", "deferred", "finalized"})
RELEASE_TRUST_LEVELS = frozenset(
    {"expert_adjudicated", "consensus_clean", "quarantine", "unresolved"}
)
EXPERT_ACTIONS = frozenset({"accept_human", "accept_model", "revise", "defer", "judge_override"})
