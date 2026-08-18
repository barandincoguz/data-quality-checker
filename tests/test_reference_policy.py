from __future__ import annotations

import pytest

from data_quality_checker.reference_policy import (
    DEFAULT_REFERENCE_POLICY_ID,
    NO_REFERENCE_FILTER_POLICY_ID,
    apply_reference_policy,
    reference_policy_spec,
)


def row(**updates):
    payload = {
        "kanun_no": "213",
        "kanun_ad": "Vergi Usul Kanunu",
        "madde": "413 üncü maddesi",
        "fikra": "1",
        "bent": "a",
        "source_text": "VUK 413 boilerplate",
    }
    payload.update(updates)
    return payload


def test_policy_normalizes_before_matching_and_preserves_raw_input() -> None:
    original = row()
    kept, audit = apply_reference_policy([original])
    assert kept == []
    assert audit["removed_reference_count"] == 1
    assert original["madde"] == "413 üncü maddesi"
    assert reference_policy_spec()["policy_id"] == DEFAULT_REFERENCE_POLICY_ID


def test_policy_none_is_an_auditable_identity_view() -> None:
    original = row()
    kept, audit = apply_reference_policy([original], policy_id=NO_REFERENCE_FILTER_POLICY_ID)
    assert kept == [original]
    assert audit["removed_reference_count"] == 0


def test_policy_does_not_remove_nonexact_core_identity() -> None:
    kept, audit = apply_reference_policy([row(madde="413/A"), row(kanun_no="3065", madde="413")])
    assert len(kept) == 2
    assert audit["removed_reference_count"] == 0


def test_unknown_policy_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown reference policy"):
        apply_reference_policy([row()], policy_id="unknown")
