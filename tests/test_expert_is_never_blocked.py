"""The expert is the truth source; the system does not refuse their judgement.

Three checks used to reject an expert's submission outright over the *content*
of the references rather than their shape:

* a redundant row -- `193/51` written beside `193/51/1-8`, which the annotators
  themselves do in 177 of the 500 canonical documents;
* a "conflicting law identity", which fires on repealed/successor statute pairs
  that share a name (5422/5520 Kurumlar Vergisi Kanunu, 1475/4857 İş Kanunu) and
  is therefore present in the canonical ground truth itself;
* a row carrying no law number and no law name.

None of these are malformed input. Rejecting them told the expert their reading
of a ruling was inadmissible, which inverts who decides. They are normalised
now, and the record says what was normalised.

Structural validation stays: a payload that is not a list of objects is a UI
fault, not a judgement, and there is nothing to normalise.
"""

from __future__ import annotations

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.hitl import validate_final_references
from data_quality_checker.normalization import conflicting_law_identity


def ref(kanun_no="", kanun_ad="", madde="", fikra="", bent=""):
    return {
        "kanun_no": kanun_no,
        "kanun_ad": kanun_ad,
        "madde": madde,
        "fikra": fikra,
        "bent": bent,
        "source_text": "",
    }


class TestExpertSubmissionAccepted:
    def test_a_repealed_and_successor_statute_pair_is_accepted(self) -> None:
        """Present in the canonical ground truth: doc_234 and doc_285 both cite
        5422 and 5520 under the same name. The expert could not save them.
        """
        result = validate_final_references(
            [
                ref("5422", "Kurumlar Vergisi Kanunu", "8"),
                ref("5520", "Kurumlar Vergisi Kanunu", "5"),
            ]
        )
        assert len(result) == 2

    def test_the_labour_law_pair_is_accepted(self) -> None:
        result = validate_final_references(
            [ref("1475", "İş Kanunu", "14"), ref("4857", "İş Kanunu", "17")]
        )
        assert len(result) == 2

    def test_both_depths_of_the_same_article_are_accepted(self) -> None:
        result = validate_final_references(
            [
                ref("193", "Gelir Vergisi Kanunu", "51"),
                ref("193", "Gelir Vergisi Kanunu", "51", "1", "8"),
            ]
        )
        assert len(result) == 2

    def test_a_bare_law_beside_one_of_its_articles_is_accepted(self) -> None:
        result = validate_final_references(
            [ref("193", "Gelir Vergisi Kanunu"), ref("193", "Gelir Vergisi Kanunu", "51")]
        )
        assert len(result) == 2

    def test_an_exact_duplicate_is_accepted(self) -> None:
        rows = [ref("193", "Gelir Vergisi Kanunu", "51")] * 2
        assert len(validate_final_references(rows)) == 2

    def test_a_row_with_an_article_but_no_law_is_kept_not_refused(self) -> None:
        """It cannot be matched to anything, but it is still something the
        expert wrote. Losing it silently is as wrong as refusing the save.
        """
        result = validate_final_references(
            [ref("193", "Gelir Vergisi Kanunu", "51"), ref(madde="7")]
        )
        assert len(result) == 2
        assert result[1]["madde"] == "7"

    def test_a_wholly_empty_row_is_dropped_silently(self) -> None:
        """A blank form row carries no claim at all, so there is nothing to
        lose by dropping it and nothing to gain by refusing the save.
        """
        result = validate_final_references(
            [ref("193", "Gelir Vergisi Kanunu", "51"), ref()]
        )
        assert len(result) == 1

    def test_an_empty_submission_is_accepted(self) -> None:
        """"This ruling cites nothing" is a legitimate expert verdict."""
        assert validate_final_references([]) == []


class TestStructuralValidationStays:
    @pytest.mark.parametrize("payload", ["not a list", {"references": []}, [1, 2], [None]])
    def test_malformed_payloads_are_still_refused(self, payload) -> None:
        with pytest.raises(ContractError):
            validate_final_references(payload)


class TestConflictDetection:
    def test_a_shared_name_across_two_statute_numbers_is_not_a_conflict(self) -> None:
        """Measured: this direction fired six times across the canonical ground
        truth and M01. Five were legitimate repealed/successor pairs; the sixth
        (a model calling 7143 the Special Consumption Tax Law) was already RED
        on core-identity grounds. It caught nothing on its own and put four
        expert-validated documents in RED.
        """
        assert not conflicting_law_identity(
            [ref("5422", "Kurumlar Vergisi Kanunu", "8"), ref("5520", "Kurumlar Vergisi Kanunu", "5")]
        )

    def test_two_names_for_one_statute_number_is_still_a_conflict(self) -> None:
        """The direction that survives: one number cannot be two laws."""
        assert conflicting_law_identity(
            [ref("999", "Birinci Kanun"), ref("999", "İkinci Kanun", "2")]
        )

    def test_a_name_that_names_no_law_is_ignored(self) -> None:
        """d000072: the annotator wrote 2560's name once as the bare word
        "Kanun" and once in full. "Kanun" discriminates nothing, so it is not
        evidence that 2560 is two different laws.
        """
        assert not conflicting_law_identity(
            [
                ref("2560", "Kanun", "21"),
                ref("2560", "İstanbul Su ve Kanalizasyon İdaresi Kanunu", "1"),
            ]
        )

    @pytest.mark.parametrize("degenerate", ["Kanun", "kanunu", "Bu Kanun", "mezkur kanun", "  KANUN "])
    def test_every_shape_of_a_contentless_name_is_ignored(self, degenerate: str) -> None:
        assert not conflicting_law_identity(
            [ref("2560", degenerate, "21"), ref("2560", "İstanbul Su Kanalizasyon Kanunu", "1")]
        )

    def test_a_real_name_still_discriminates(self) -> None:
        assert conflicting_law_identity(
            [ref("2560", "Harçlar Kanunu", "21"), ref("2560", "Damga Vergisi Kanunu", "1")]
        )
