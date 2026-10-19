"""One row per provision: a reference is a law and an article of it.

User decision (1 Oct 2026): nothing below the article exists. Rows that name
the same law and article are one claim -- whether they were written at two
depths before that day (`193/51` beside `193/51/1-8`, the shape that sent
d000017 to YELLOW) or spelled twice ("Geçici 21", "geçici 21"). A law-level row
still goes when an article of that law is named.
"""

from __future__ import annotations

import itertools

import pytest

from data_quality_checker.normalization import (
    compact_references,
    core_identity,
    normalize_reference,
)


def _reference(kanun_no: str, madde: str, **legacy: str) -> dict[str, str]:
    return {
        "kanun_no": kanun_no,
        "kanun_ad": "Gelir Vergisi Kanunu" if kanun_no == "193" else "",
        "madde": madde,
        "source_text": "",
        **legacy,
    }


def _identities(references: list[dict[str, str]]) -> set[tuple[str, ...]]:
    return {core_identity(reference) for reference in compact_references(references)}


class TestOneRowPerProvision:
    def test_a_stored_row_at_two_depths_is_one_provision(self) -> None:
        rows = [_reference("193", "51"), _reference("193", "51", fikra="1", bent="8")]
        assert len(compact_references(rows)) == 1

    def test_several_paragraphs_of_one_article_are_one_provision(self) -> None:
        rows = [
            _reference("3065", "17"),
            _reference("3065", "17", fikra="4", bent="l"),
            _reference("3065", "17/2-b"),
        ]
        assert _identities(rows) == {core_identity(normalize_reference(_reference("3065", "17")))}

    def test_one_reference_spelled_twice_is_one(self) -> None:
        rows = [_reference("4447", "Geçici 21"), _reference("4447", "geçici 21")]
        assert len(compact_references(rows)) == 1

    def test_different_articles_and_laws_stay_apart(self) -> None:
        rows = [
            _reference("193", "37"),
            _reference("193", "51"),
            _reference("6102", "51"),
            _reference("5520", "32/A"),
            _reference("5520", "32"),
        ]
        assert len(compact_references(rows)) == 5

    def test_the_law_level_rule_still_applies(self) -> None:
        """A bare law row goes when any article of that law is named."""
        rows = [_reference("193", ""), _reference("193", "51")]
        assert _identities(rows) == _identities([_reference("193", "51")])

    def test_a_bare_law_row_alone_stays(self) -> None:
        assert len(compact_references([_reference("193", ""), _reference("6102", "3")])) == 2


class TestRobustness:
    """Order independence and idempotence, the two properties a set-reduction
    rule has to have before any bucket may depend on it.
    """

    CASES = [
        [
            _reference("193", "51"),
            _reference("193", "51", fikra="1", bent="8"),
            _reference("193", "37"),
        ],
        [
            _reference("3065", "17", fikra="4", bent="l"),
            _reference("3065", "17"),
            _reference("3065", "17/2-b"),
            _reference("3065", ""),
        ],
        [
            _reference("193", ""),
            _reference("193", "51"),
            _reference("5520", "32/A"),
            _reference("5520", "32/a"),
        ],
    ]

    @pytest.mark.parametrize("rows", CASES)
    def test_the_result_does_not_depend_on_input_order(self, rows) -> None:
        expected = _identities(rows)
        for permutation in itertools.permutations(rows):
            assert _identities(list(permutation)) == expected

    @pytest.mark.parametrize("rows", CASES)
    def test_compaction_is_idempotent(self, rows) -> None:
        once = compact_references(rows)
        assert compact_references(once) == once

    @pytest.mark.parametrize("rows", CASES)
    def test_compaction_only_ever_removes(self, rows) -> None:
        """Every surviving identity is an identity of the input, and the result
        can never grow."""
        input_identities = {core_identity(normalize_reference(row)) for row in rows}
        assert _identities(rows) <= input_identities
        assert len(compact_references(rows)) <= len(rows)


def test_the_d000017_shape_is_an_exact_agreement() -> None:
    """The real document that once went YELLOW, reduced to its shape: five
    references identical on both sides; the model additionally names the bare
    article whose paragraph both sides named."""
    human = [
        _reference("193", "37"),
        _reference("6102", "3"),
        _reference("193", "46"),
        _reference("193", "47"),
        _reference("193", "48"),
        _reference("193", "51", fikra="1", bent="8"),
    ]
    model = [*human[:5], _reference("193", "51"), _reference("193", "51", fikra="1", bent="8")]
    assert _identities(human) == _identities(model)
