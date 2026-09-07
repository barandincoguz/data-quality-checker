"""A bare article is absorbed by its own paragraph, exactly as a bare law is.

`compact_references` has always dropped a law-level row when a specific article
of that law is also present. It did not do the same one level down, so a model
that named both `193/51` and `193/51/1-8` disagreed with an annotator who named
only the second, and the document was routed YELLOW -- to a judge call and an
expert -- over a row that adds nothing.

The measurement engine already disagreed with that verdict. `prepare_reference_set`
collapses rows sharing (kanun_no, kanun_ad, madde), so on d000017 it reduced the
model's seven rows to the annotator's exact six and scored a perfect match while
the router called the document a mismatch. This closes that gap.
"""

from __future__ import annotations

import itertools

import pytest

from data_quality_checker.normalization import compact_references, full_identity


def _reference(kanun_no: str, madde: str, fikra: str = "", bent: str = "") -> dict[str, str]:
    return {
        "kanun_no": kanun_no,
        "kanun_ad": "Gelir Vergisi Kanunu" if kanun_no == "193" else "",
        "madde": madde,
        "fikra": fikra,
        "bent": bent,
        "source_text": "",
    }


def _identities(references: list[dict[str, str]]) -> set[tuple[str, ...]]:
    return {full_identity(reference) for reference in compact_references(references)}


class TestAbsorption:
    def test_a_bare_article_is_absorbed_by_its_paragraph(self) -> None:
        rows = [_reference("193", "51"), _reference("193", "51", "1", "8")]
        assert _identities(rows) == _identities([_reference("193", "51", "1", "8")])

    def test_a_bare_article_is_absorbed_by_a_subparagraph_alone(self) -> None:
        rows = [_reference("3065", "13"), _reference("3065", "13", "", "a")]
        assert _identities(rows) == _identities([_reference("3065", "13", "", "a")])

    def test_several_paragraphs_all_absorb_the_same_bare_article(self) -> None:
        rows = [
            _reference("3065", "17"),
            _reference("3065", "17", "4", "l"),
            _reference("3065", "17", "2", "b"),
        ]
        assert _identities(rows) == {
            full_identity(_reference("3065", "17", "4", "l")),
            full_identity(_reference("3065", "17", "2", "b")),
        }

    def test_a_bare_article_of_a_different_article_survives(self) -> None:
        """Absorption is per article. Article 51 having a paragraph says
        nothing about article 37.
        """
        rows = [_reference("193", "37"), _reference("193", "51", "1", "8")]
        assert _identities(rows) == {
            full_identity(_reference("193", "37")),
            full_identity(_reference("193", "51", "1", "8")),
        }

    def test_a_bare_article_of_a_different_law_survives(self) -> None:
        rows = [_reference("6102", "3"), _reference("193", "3", "1", "")]
        assert len(compact_references(rows)) == 2

    def test_nothing_is_absorbed_when_no_row_is_specific(self) -> None:
        rows = [_reference("193", "37"), _reference("193", "51"), _reference("6102", "3")]
        assert len(compact_references(rows)) == 3

    def test_the_law_level_rule_still_applies(self) -> None:
        """The pre-existing rule must survive the new one: a bare law row goes
        when any article of that law is named.
        """
        rows = [_reference("193", ""), _reference("193", "51", "1", "8")]
        assert _identities(rows) == _identities([_reference("193", "51", "1", "8")])

    def test_a_law_named_only_by_a_paragraph_still_absorbs_the_bare_law_row(self) -> None:
        """Malformed but possible: a model may emit a paragraph with no article.
        The bare law row shares that row's core identity, so it is absorbed.
        Pinned so the behaviour on this shape is a decision, not an accident.
        """
        rows = [_reference("193", ""), _reference("193", "", "1", "")]
        assert _identities(rows) == {full_identity(_reference("193", "", "1", ""))}


class TestRobustness:
    """Order independence and idempotence, the two properties a set-reduction
    rule has to have before any bucket may depend on it.
    """

    CASES = [
        [_reference("193", "51"), _reference("193", "51", "1", "8"), _reference("193", "37")],
        [
            _reference("3065", "17", "4", "l"),
            _reference("3065", "17"),
            _reference("3065", "17", "2", "b"),
            _reference("3065", ""),
        ],
        [_reference("193", ""), _reference("193", "51"), _reference("193", "51", "1", "8")],
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
    def test_absorption_only_ever_removes(self, rows) -> None:
        """Compaction is a reduction. Every surviving identity must already be
        an identity of the input, and the result can never grow.
        """
        from data_quality_checker.normalization import normalize_reference

        input_identities = {full_identity(normalize_reference(row)) for row in rows}
        survivors = _identities(rows)
        assert survivors <= input_identities
        assert len(compact_references(rows)) <= len(rows)


def test_the_d000017_shape_becomes_an_exact_agreement() -> None:
    """The real document that motivated the change, reduced to its shape.

    Five references identical on both sides; the model additionally names the
    bare article whose paragraph it already named.
    """
    human = [
        _reference("193", "37"),
        _reference("6102", "3"),
        _reference("193", "46"),
        _reference("193", "47"),
        _reference("193", "48"),
        _reference("193", "51", "1", "8"),
    ]
    model = [*human[:5], _reference("193", "51"), _reference("193", "51", "1", "8")]
    assert _identities(human) == _identities(model)
