"""spaCy-based reference annotator using Matcher and PhraseMatcher."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from annotator.reference.regex import RegexReferenceAnnotator
    from annotator.reference.common import (
        CANONICAL_LAW_BY_NO,
        CORE_FIELDS,
        KNOWN_LAW_NAMES,
        LAW_ABBREVIATIONS,
        REFERENCE_TYPE,
        canonical_law_name_by_no,
        collapse_ws,
        deduplicate_references,
        extract_identifier_series,
        extract_ozelge_no,
        normalize_identifier,
        normalize_kanun_adi,
        normalize_reference_text,
        parse_madde_token,
        snippet,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from annotator.reference.regex import RegexReferenceAnnotator
    from annotator.reference.common import (
        CANONICAL_LAW_BY_NO,
        CORE_FIELDS,
        KNOWN_LAW_NAMES,
        LAW_ABBREVIATIONS,
        REFERENCE_TYPE,
        canonical_law_name_by_no,
        collapse_ws,
        deduplicate_references,
        extract_identifier_series,
        extract_ozelge_no,
        normalize_identifier,
        normalize_kanun_adi,
        normalize_reference_text,
        parse_madde_token,
        snippet,
    )

try:
    import spacy
    from spacy.matcher import Matcher, PhraseMatcher
except Exception:  # pragma: no cover - defensive path for environments without spaCy
    spacy = None
    Matcher = None
    PhraseMatcher = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "test_data.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark" / "predictions" / "spacy_reference_annotations.json"

LAW_NAME_TO_NO = {
    law_name: law_no
    for law_no, law_name in CANONICAL_LAW_BY_NO.items()
    if law_no.isdigit() and law_name not in {"Kanun", "Kanun Hükmünde Kararname"}
}

ORDINAL_SUFFIX_PATTERN = r"(?:'?(?:inci|ıncı|üncü|uncu|nci|ncı|ncu|ncü))"
IDENTIFIER_PREFIX_PATTERN = (
    r"(?:\((?P<par>[A-Za-z0-9]+)\)|"
    r"(?P<num>\d+)(?:\s*"
    + ORDINAL_SUFFIX_PATTERN
    + r")?|"
    r"(?P<ord>birinci|ikinci|ucuncu|dorduncu|besinci|altinci|yedinci|sekizinci|dokuzuncu|onuncu))"
)
BENT_IDENTIFIER_PREFIX_PATTERN = (
    r"(?:\((?P<par>[A-Za-z0-9]+)\)|"
    r"(?P<num>\d+)(?:\s*"
    + ORDINAL_SUFFIX_PATTERN
    + r")?|"
    r"(?P<ord>birinci|ikinci|ucuncu|dorduncu|besinci|altinci|yedinci|sekizinci|dokuzuncu|onuncu)|"
    r"\b(?P<let>[A-Za-zÇĞİÖŞÜçğıöşü])\b)"
)
RE_FIKRA = re.compile(
    IDENTIFIER_PREFIX_PATTERN + r"\s*(?:numaral[ıi]\s+)?f[ıi]kra\w*",
    re.IGNORECASE | re.UNICODE,
)
RE_BENT = re.compile(
    BENT_IDENTIFIER_PREFIX_PATTERN + r"\s*(?:numaral[ıi]\s+)?ben[dt]\w*",
    re.IGNORECASE | re.UNICODE,
)
RE_TABLE_REF = re.compile(r"\b(?:cetvel|tablo|liste)\w*\b", re.IGNORECASE | re.UNICODE)
RE_MADDE_TOKEN = re.compile(
    r"(?:(?:m[üu]kerrer|ge[çc]ici|ek)\s+)?\d+(?:\s*/\s*\d+(?:\s*-\s*[A-Za-zÇĞİÖŞÜçğıöşü])?|\s*/\s*[A-Za-zÇĞİÖŞÜçğıöşü]|\s*-\s*[A-Za-zÇĞİÖŞÜçğıöşü])?|[Ee]k\s+[Mm]adde\s+\d+",
    re.UNICODE,
)
RE_MADDE = re.compile(
    r"(?P<madde_token>(?:(?:m[üu]kerrer|ge[çc]ici|ek)\s+)?\d+(?:\s*/\s*\d+(?:\s*-\s*[A-Za-zÇĞİÖŞÜçğıöşü])?|\s*/\s*[A-Za-zÇĞİÖŞÜçğıöşü]|\s*-\s*[A-Za-zÇĞİÖŞÜçğıöşü])?|\(\s*\d+(?:\s*/\s*\d+(?:\s*-\s*[A-Za-zÇĞİÖŞÜçğıöşü])?|\s*/\s*[A-Za-zÇĞİÖŞÜçğıöşü]|\s*-\s*[A-Za-zÇĞİÖŞÜçğıöşü])?\s*\)|[Ee]k\s+[Mm]adde\s+\d+)"
    r"\s*" + ORDINAL_SUFFIX_PATTERN + r"?\s*\.?\s*madd\w*",
    re.IGNORECASE | re.UNICODE,
)
RE_CANONICAL_COMPOSITE_TOKEN = re.compile(
    r"^(?P<madde>(?:m[üu]kerrer|ge[çc]ici|ek)\s+\d+|\d+)\s*/\s*(?P<fikra>\d+)(?:\s*-\s*(?P<bent>[A-Za-zÇĞİÖŞÜçğıöşü]))?$",
    re.IGNORECASE | re.UNICODE,
)
RE_SAYILI_LEFT_CONTEXT = re.compile(r"(\d{2,5}(?:/\d{1,5})?)\s+say[ıi]l[ıi]\s*$", re.IGNORECASE | re.UNICODE)
RE_SLASH_LETTER_MADDE = re.compile(r"^\d+\s*/\s*[A-Za-zÇĞİÖŞÜçğıöşü]$", re.UNICODE)
RE_ARTICLE_ANAPHORA = re.compile(r"\b(?:bu|ayn[ıi])\s+madd\w*\b", re.IGNORECASE | re.UNICODE)
RE_PARAGRAPH_ANAPHORA = re.compile(r"\bayn[ıi]\s+f[ıi]kra\w*\b", re.IGNORECASE | re.UNICODE)
# Legacy extraction patterns kept for non-breaking baseline path.
RE_FIKRA_LEGACY = re.compile(
    r"(?:\((?P<par>[A-Za-z0-9]+)\)|(?P<num>\d+)|(?P<ord>birinci|ikinci|ucuncu|dorduncu|besinci|altinci|yedinci|sekizinci|dokuzuncu|onuncu))"
    r"\s*(?:numaral[ıi]\s+)?f[ıi]kra\w*",
    re.IGNORECASE | re.UNICODE,
)
RE_BENT_LEGACY = re.compile(
    BENT_IDENTIFIER_PREFIX_PATTERN + r"\s*(?:numaral[ıi]\s+)?ben[dt]\w*",
    re.IGNORECASE | re.UNICODE,
)
RE_BENT_SERIES = re.compile(
    r"(?P<items>(?:\(\s*[A-Za-z0-9]+\s*\)|\d+(?:\s*'?(?:inci|ıncı|üncü|uncu|nci|ncı|ncu|ncü))?|birinci|ikinci|ucuncu|dorduncu|besinci|altinci|yedinci|sekizinci|dokuzuncu|onuncu)"
    r"(?:\s*,\s*(?:\(\s*[A-Za-z0-9]+\s*\)|\d+(?:\s*'?(?:inci|ıncı|üncü|uncu|nci|ncı|ncu|ncü))?|birinci|ikinci|ucuncu|dorduncu|besinci|altinci|yedinci|sekizinci|dokuzuncu|onuncu))*"
    r"(?:\s*(?:ve|veya)\s*(?:\(\s*[A-Za-z0-9]+\s*\)|\d+(?:\s*'?(?:inci|ıncı|üncü|uncu|nci|ncı|ncu|ncü))?|birinci|ikinci|ucuncu|dorduncu|besinci|altinci|yedinci|sekizinci|dokuzuncu|onuncu))?)"
    r"\s*(?:numaral[ıi]\s+)?ben[dt]\w*",
    re.IGNORECASE | re.UNICODE,
)
RE_TABLE_FIKRA = re.compile(
    r"(?P<section>[IVXLCDM]+)\s*/\s*(?P<madde>\d+)(?:\s*-\s*(?P<bent>[A-Za-zÇĞİÖŞÜçğıöşü]))?\s*f[ıi]kra\w*",
    re.IGNORECASE | re.UNICODE,
)
RE_MADDE_TOKEN_LEGACY = re.compile(
    r"(?:(?:m[üu]kerrer|ge[çc]ici|ek)\s+)?\d+(?:/\d+(?:-[A-Za-zÇĞİÖŞÜçğıöşü])?|/[A-Za-zÇĞİÖŞÜçğıöşü]|-[A-Za-zÇĞİÖŞÜçğıöşü])?|[Ee]k\s+[Mm]adde\s+\d+",
    re.UNICODE,
)
RE_MADDE_LEGACY = re.compile(
    r"(?P<madde_token>(?:(?:m[üu]kerrer|ge[çc]ici|ek)\s+)?\d+(?:/\d+(?:-[A-Za-zÇĞİÖŞÜçğıöşü])?|/[A-Za-zÇĞİÖŞÜçğıöşü]|-[A-Za-zÇĞİÖŞÜçğıöşü])?|\(\d+(?:/\d+(?:-[A-Za-zÇĞİÖŞÜçğıöşü])?|/[A-Za-zÇĞİÖŞÜçğıöşü]|-[A-Za-zÇĞİÖŞÜçğıöşü])?\)|[Ee]k\s+[Mm]adde\s+\d+)"
    r"\s*(?:'?(?:inci|ıncı|üncü|uncu|nci|ncı|ncu|ncü))?\s*\.?\s*madd\w*",
    re.IGNORECASE | re.UNICODE,
)
REQUIRED_OUTPUT_KEYS = ("references", "ozelge_no", "status")

CANDIDATE_LABEL_LAW_NAME = "LAW_NAME"
CANDIDATE_LABEL_LAW_ALIAS = "LAW_ALIAS"
CANDIDATE_LABEL_ARTICLE = "ARTICLE"
CANDIDATE_LABEL_PARAGRAPH = "PARAGRAPH"
CANDIDATE_LABEL_CLAUSE = "CLAUSE"
CANDIDATE_LABEL_TABLE_REF = "TABLE_REF"
CANDIDATE_LABEL_TEMP_ARTICLE = "TEMP_ARTICLE"
CANDIDATE_LABEL_ANNEX_ARTICLE = "ANNEX_ARTICLE"
CANDIDATE_LABEL_LAW_ANAPHORA = "LAW_ANAPHORA"


@dataclass
class SpanCandidate:
    """Typed extraction candidate before final reference resolution."""

    label: str
    start_char: int
    end_char: int
    text: str
    sentence_index: int
    token_start: int
    token_end: int
    data: dict[str, str] = field(default_factory=dict)


@dataclass
class DocumentContextState:
    """Stateful context memory for cross-sentence legal anaphora resolution."""

    last_explicit_law_no: str = ""
    last_explicit_law_ad: str = ""
    last_explicit_article: str = ""
    last_explicit_table: str = ""
    last_resolved_paragraph: str = ""


class SpacyReferenceAnnotator:
    """spaCy matcher-based baseline annotator."""

    def __init__(self) -> None:
        if spacy is None:
            raise RuntimeError("spaCy is not installed in this environment.")

        try:
            self.nlp = spacy.blank("tr")
        except Exception:
            self.nlp = spacy.blank("xx")

        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

        self.law_matcher = Matcher(self.nlp.vocab)
        self.atif_matcher = Matcher(self.nlp.vocab)
        self.madde_matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.abbr_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        self._build_patterns()
        self.regex_fallback = RegexReferenceAnnotator()

    def _build_patterns(self) -> None:
        self.law_matcher.add(
            "LAW_EXPLICIT",
            [
                [
                    {"TEXT": {"REGEX": r"^\d{2,5}(?:/\d{1,5})?$"}},
                    {"LOWER": {"IN": ["sayılı", "sayili"]}},
                    {"IS_PUNCT": False, "OP": "*"},
                    {"LOWER": {"REGEX": r"^kanun.*$"}},
                ],
                [
                    {"TEXT": {"REGEX": r"^\d{2,5}(?:/\d{1,5})?$"}},
                    {"LOWER": {"IN": ["sayılı", "sayili"]}},
                    {"LOWER": {"REGEX": r"^kanun.*$"}},
                ],
            ],
        )

        self.atif_matcher.add(
            "LAW_ATIF",
            [
                [
                    {
                        "LOWER": {
                            "IN": [
                                "aynı",
                                "ayni",
                                "anılan",
                                "anilan",
                                "ilgili",
                                "bu",
                                "mezkur",
                                "söz",
                                "soz",
                                "bahse",
                            ]
                        }
                    },
                    {"LOWER": {"IN": ["konusu", "konu"]}, "OP": "?"},
                    {"LOWER": {"REGEX": r"^kanun.*$"}},
                ],
                [{"LOWER": {"IN": ["kanunun", "kanuna", "kanunda", "kanundan"]}}],
            ],
        )

        self.madde_matcher.add(
            "MADDE",
            [
                [
                    {"LOWER": {"IN": ["mükerrer", "geçici", "ek"]}, "OP": "?"},
                    {
                        "TEXT": {
                            "REGEX": r"^\d+(?:/\d+(?:-[A-Za-zÇĞİÖŞÜçğıöşü])?|/[A-Za-zÇĞİÖŞÜçğıöşü]|-[A-Za-zÇĞİÖŞÜçğıöşü])?\.?$"
                        }
                    },
                    {"IS_PUNCT": True, "OP": "?"},
                    {"LOWER": {"IN": ["inci", "ıncı", "üncü", "uncu", "nci", "ncı", "ncu", "ncü"]}, "OP": "?"},
                    {"LOWER": {"REGEX": r"^madd.*$"}},
                ],
                [
                    {"LOWER": {"IN": ["mükerrer", "geçici", "ek"]}, "OP": "?"},
                    {"TEXT": {"REGEX": r"^\d+$"}},
                    {"TEXT": "/"},
                    {"TEXT": {"REGEX": r"^\d+$"}},
                    {"TEXT": {"IN": ["-", "/"]}, "OP": "?"},
                    {"TEXT": {"REGEX": r"^[A-Za-zÇĞİÖŞÜçğıöşü]$"}, "OP": "?"},
                    {"LOWER": {"IN": ["inci", "ıncı", "üncü", "uncu", "nci", "ncı", "ncu", "ncü"]}, "OP": "?"},
                    {"LOWER": {"REGEX": r"^madd.*$"}},
                ],
                [
                    {"LOWER": {"REGEX": r"^madd.*$"}},
                    {
                        "TEXT": {
                            "REGEX": r"^\d+(?:/\d+(?:-[A-Za-zÇĞİÖŞÜçğıöşü])?|/[A-Za-zÇĞİÖŞÜçğıöşü]|-[A-Za-zÇĞİÖŞÜçğıöşü])?$"
                        }
                    },
                ],
                [
                    {"LOWER": "ek"},
                    {"LOWER": {"REGEX": r"^madd.*$"}},
                    {"TEXT": {"REGEX": r"^\d+$"}},
                ],
            ],
        )

        law_phrases = [self.nlp.make_doc(name) for name in KNOWN_LAW_NAMES]
        self.phrase_matcher.add("KNOWN_LAW", law_phrases)

        abbr_phrases = [self.nlp.make_doc(key) for key in LAW_ABBREVIATIONS.keys()]
        self.abbr_matcher.add("LAW_ABBR", abbr_phrases)

    def annotate_document(self, text: str) -> dict[str, Any]:
        """Annotate a single document in Gemini-compatible shape."""
        text = normalize_reference_text(text)
        doc = self.nlp(text)
        return self._annotate_parsed_doc(doc)

    def _annotate_parsed_doc(self, doc) -> dict[str, Any]:
        """Annotate an already parsed spaCy Doc."""
        candidates = self._extract_candidates(doc)
        resolved_candidates = self._resolve_candidates(doc, candidates)
        # Keep legacy output as primary non-breaking path while resolver matures.
        references = self._legacy_sentence_resolution(doc)
        cleaned = deduplicate_references(references)
        cleaned = self._dedupe_core_references(cleaned)
        for ref in cleaned:
            ref["type"] = REFERENCE_TYPE
        return {
            "references": cleaned,
            "ozelge_no": extract_ozelge_no(doc.text),
            "status": "success",
            "diagnostics": {
                "candidate_count": len(candidates),
                "resolver_reference_count": len(resolved_candidates),
            },
        }

    @staticmethod
    def _dedupe_core_references(references: list[dict[str, str]]) -> list[dict[str, str]]:
        """Drop repeated same-core references that differ only by source snippet."""
        unique: list[dict[str, str]] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        for ref in references:
            key = tuple(str(ref.get(field, "")).strip() for field in CORE_FIELDS)
            if key in seen:
                continue
            seen.add(key)
            unique.append(ref)
        return unique

    @staticmethod
    def _enforce_output_contract(result: dict[str, Any]) -> dict[str, Any]:
        """Guarantee stable output keys for downstream benchmark consumers."""
        for key in REQUIRED_OUTPUT_KEYS:
            if key not in result:
                if key == "references":
                    result[key] = []
                elif key == "ozelge_no":
                    result[key] = ""
                elif key == "status":
                    result[key] = "error"
        if not isinstance(result.get("references"), list):
            result["references"] = []
            result["status"] = "error"
        if not isinstance(result.get("ozelge_no"), str):
            result["ozelge_no"] = ""
        return result

    def _extract_candidates(self, doc) -> list[SpanCandidate]:
        """Extract typed span candidates from the document."""
        sent_spans = list(doc.sents)
        candidates: list[SpanCandidate] = []

        law_mentions = self._find_law_mentions(doc)
        for law in law_mentions:
            token_span = doc.char_span(law["start"], law["end"], alignment_mode="expand")
            if token_span is None:
                continue
            candidates.append(
                SpanCandidate(
                    label=CANDIDATE_LABEL_LAW_NAME,
                    start_char=law["start"],
                    end_char=law["end"],
                    text=doc.text[law["start"] : law["end"]],
                    sentence_index=self._sentence_index_for_char(sent_spans, law["start"]),
                    token_start=token_span.start,
                    token_end=token_span.end,
                    data={"kanun_no": law.get("kanun_no", ""), "kanun_ad": law.get("kanun_ad", "")},
                )
            )

        atif_mentions = self._find_atif_mentions(doc)
        for atif in atif_mentions:
            token_span = doc.char_span(atif["start"], atif["end"], alignment_mode="expand")
            if token_span is None:
                continue
            candidates.append(
                SpanCandidate(
                    label=CANDIDATE_LABEL_LAW_ANAPHORA,
                    start_char=atif["start"],
                    end_char=atif["end"],
                    text=doc.text[atif["start"] : atif["end"]],
                    sentence_index=self._sentence_index_for_char(sent_spans, atif["start"]),
                    token_start=token_span.start,
                    token_end=token_span.end,
                )
            )

        madde_mentions = self._find_madde_mentions(doc)
        for madde_item in madde_mentions:
            token_span = doc.char_span(madde_item["start"], madde_item["end"], alignment_mode="expand")
            if token_span is None:
                continue
            label = CANDIDATE_LABEL_ARTICLE
            maddetext = madde_item.get("madde", "").lower()
            if maddetext.startswith("geçici") or maddetext.startswith("gecici"):
                label = CANDIDATE_LABEL_TEMP_ARTICLE
            elif maddetext.startswith("ek "):
                label = CANDIDATE_LABEL_ANNEX_ARTICLE
            candidates.append(
                SpanCandidate(
                    label=label,
                    start_char=madde_item["start"],
                    end_char=madde_item["end"],
                    text=doc.text[madde_item["start"] : madde_item["end"]],
                    sentence_index=self._sentence_index_for_char(sent_spans, madde_item["start"]),
                    token_start=token_span.start,
                    token_end=token_span.end,
                    data={
                        "madde": madde_item.get("madde", ""),
                        "fikra": madde_item.get("fikra", ""),
                        "bent": madde_item.get("bent", ""),
                    },
                )
            )

        for match in RE_FIKRA.finditer(doc.text):
            token_span = doc.char_span(match.start(), match.end(), alignment_mode="expand")
            if token_span is None:
                continue
            value = self._identifier_from_match(match)
            candidates.append(
                SpanCandidate(
                    label=CANDIDATE_LABEL_PARAGRAPH,
                    start_char=match.start(),
                    end_char=match.end(),
                    text=match.group(0),
                    sentence_index=self._sentence_index_for_char(sent_spans, match.start()),
                    token_start=token_span.start,
                    token_end=token_span.end,
                    data={"fikra": normalize_identifier(value)},
                )
            )

        for match in RE_BENT.finditer(doc.text):
            token_span = doc.char_span(match.start(), match.end(), alignment_mode="expand")
            if token_span is None:
                continue
            value = self._identifier_from_match(match)
            candidates.append(
                SpanCandidate(
                    label=CANDIDATE_LABEL_CLAUSE,
                    start_char=match.start(),
                    end_char=match.end(),
                    text=match.group(0),
                    sentence_index=self._sentence_index_for_char(sent_spans, match.start()),
                    token_start=token_span.start,
                    token_end=token_span.end,
                    data={"bent": normalize_identifier(value)},
                )
            )

        for match in RE_TABLE_REF.finditer(doc.text):
            token_span = doc.char_span(match.start(), match.end(), alignment_mode="expand")
            if token_span is None:
                continue
            candidates.append(
                SpanCandidate(
                    label=CANDIDATE_LABEL_TABLE_REF,
                    start_char=match.start(),
                    end_char=match.end(),
                    text=match.group(0),
                    sentence_index=self._sentence_index_for_char(sent_spans, match.start()),
                    token_start=token_span.start,
                    token_end=token_span.end,
                )
            )

        candidates = self._merge_candidates_preserve_overlap(candidates)

        # SpanRuler-compatible storage: keep overlapping candidates in Doc.spans.
        span_store = []
        for candidate in candidates:
            span = doc[candidate.token_start : candidate.token_end]
            if span is not None:
                span_store.append(span)
        doc.spans["reference_candidates"] = span_store

        candidates.sort(key=lambda item: (item.start_char, item.end_char))
        return candidates

    def _resolve_candidates(self, doc, candidates: list[SpanCandidate]) -> list[dict[str, str]]:
        """Resolve typed candidates into canonical references using stateful context."""
        references: list[dict[str, str]] = []
        state = DocumentContextState()
        sent_spans = list(doc.sents)
        sent_count = len(sent_spans)
        resolved_articles: list[dict[str, str | int]] = []
        consumed_clause_ids: set[int] = set()
        consumed_paragraph_ids: set[int] = set()

        by_sentence: dict[int, list[SpanCandidate]] = {idx: [] for idx in range(sent_count)}
        for candidate in candidates:
            by_sentence.setdefault(candidate.sentence_index, []).append(candidate)

        for sent_idx, sent in enumerate(sent_spans):
            sent_candidates = by_sentence.get(sent_idx, [])
            explicit_laws = [
                c for c in sent_candidates if c.label in {CANDIDATE_LABEL_LAW_NAME, CANDIDATE_LABEL_LAW_ALIAS}
            ]
            if explicit_laws:
                law = explicit_laws[-1]
                state.last_explicit_law_no = law.data.get("kanun_no", "")
                state.last_explicit_law_ad = law.data.get("kanun_ad", "")

            table_refs = [c for c in sent_candidates if c.label == CANDIDATE_LABEL_TABLE_REF]
            if table_refs:
                state.last_explicit_table = table_refs[-1].text

            article_candidates = [
                c
                for c in sent_candidates
                if c.label in {CANDIDATE_LABEL_ARTICLE, CANDIDATE_LABEL_TEMP_ARTICLE, CANDIDATE_LABEL_ANNEX_ARTICLE}
            ]
            paragraph_candidates = [c for c in sent_candidates if c.label == CANDIDATE_LABEL_PARAGRAPH]
            clause_candidates = [c for c in sent_candidates if c.label == CANDIDATE_LABEL_CLAUSE]
            has_article_anaphora = bool(RE_ARTICLE_ANAPHORA.search(sent.text))
            has_paragraph_anaphora = bool(RE_PARAGRAPH_ANAPHORA.search(sent.text))

            for article in article_candidates:
                law_no, law_ad = self._resolve_law_for_candidate(
                    article=article,
                    sent_idx=sent_idx,
                    by_sentence=by_sentence,
                    state=state,
                )
                if law_ad:
                    state.last_explicit_law_no = law_no
                    state.last_explicit_law_ad = law_ad
                madde = article.data.get("madde", "")
                if madde:
                    state.last_explicit_article = madde

                local_tail = doc.text[article.end_char : min(len(doc.text), article.end_char + 220)]
                local_fikra = article.data.get("fikra", "") or self._extract_identifier(RE_FIKRA, local_tail)
                local_bent_values = self._extract_all_identifiers(RE_BENT, local_tail)
                if not local_bent_values:
                    local_bent_values = [article.data.get("bent", "")]
                if not local_bent_values:
                    local_bent_values = [""]

                for paragraph in paragraph_candidates:
                    if abs(paragraph.start_char - article.end_char) <= 220 and paragraph.data.get("fikra"):
                        local_fikra = paragraph.data["fikra"]
                        consumed_paragraph_ids.add(id(paragraph))

                for bent in local_bent_values:
                    ref = {
                        "type": REFERENCE_TYPE,
                        "kanun_no": law_no,
                        "kanun_ad": law_ad,
                        "madde": madde,
                        "fikra": local_fikra,
                        "bent": bent,
                        "source_text": snippet(doc.text, max(sent.start_char, article.start_char - 40), min(sent.end_char, article.end_char + 180)),
                    }
                    references.append(ref)
                    resolved_articles.append(
                        {
                            "sentence_index": sent_idx,
                            "kanun_no": law_no,
                            "kanun_ad": law_ad,
                            "madde": madde,
                            "fikra": local_fikra,
                        }
                    )
                    if local_fikra:
                        state.last_resolved_paragraph = local_fikra

            # Standalone paragraph/clause handling with nearest article context.
            for paragraph in paragraph_candidates:
                if id(paragraph) in consumed_paragraph_ids:
                    continue
                target = self._nearest_article_context(resolved_articles, sent_idx)
                if target is None:
                    if has_article_anaphora and state.last_explicit_article:
                        target = {
                            "sentence_index": sent_idx,
                            "kanun_no": state.last_explicit_law_no,
                            "kanun_ad": state.last_explicit_law_ad,
                            "madde": state.last_explicit_article,
                            "fikra": "",
                        }
                    else:
                        continue
                ref = {
                    "type": REFERENCE_TYPE,
                    "kanun_no": str(target.get("kanun_no", "")),
                    "kanun_ad": str(target.get("kanun_ad", "")),
                    "madde": str(target.get("madde", "")),
                    "fikra": paragraph.data.get("fikra", "") or str(target.get("fikra", "")),
                    "bent": "",
                    "source_text": snippet(doc.text, max(sent.start_char, paragraph.start_char - 40), min(sent.end_char, paragraph.end_char + 120)),
                }
                references.append(ref)
                state.last_resolved_paragraph = ref["fikra"]

            for clause in clause_candidates:
                if id(clause) in consumed_clause_ids:
                    continue
                target = self._nearest_article_context(resolved_articles, sent_idx)
                if target is None and has_article_anaphora and state.last_explicit_article:
                    target = {
                        "sentence_index": sent_idx,
                        "kanun_no": state.last_explicit_law_no,
                        "kanun_ad": state.last_explicit_law_ad,
                        "madde": state.last_explicit_article,
                        "fikra": state.last_resolved_paragraph if has_paragraph_anaphora else "",
                    }
                if target is None:
                    continue
                ref = {
                    "type": REFERENCE_TYPE,
                    "kanun_no": str(target.get("kanun_no", "")),
                    "kanun_ad": str(target.get("kanun_ad", "")),
                    "madde": str(target.get("madde", "")),
                    "fikra": str(target.get("fikra", "")),
                    "bent": clause.data.get("bent", ""),
                    "source_text": snippet(doc.text, max(sent.start_char, clause.start_char - 40), min(sent.end_char, clause.end_char + 120)),
                }
                references.append(ref)

        return references

    def _merge_candidates_preserve_overlap(self, candidates: list[SpanCandidate]) -> list[SpanCandidate]:
        """Merge duplicate/near-duplicate candidates without dropping valid overlaps."""
        if not candidates:
            return []

        merged: list[SpanCandidate] = []
        for candidate in sorted(candidates, key=lambda item: (item.start_char, item.end_char, item.label)):
            replaced = False
            for idx, current in enumerate(merged):
                if current.label != candidate.label:
                    continue
                if current.end_char <= candidate.start_char or candidate.end_char <= current.start_char:
                    continue
                if not self._same_candidate_payload(current, candidate):
                    continue
                merged[idx] = self._merge_candidate_pair(current, candidate)
                replaced = True
                break
            if not replaced:
                merged.append(candidate)
        merged.sort(key=lambda item: (item.start_char, item.end_char, item.label))
        return merged

    @staticmethod
    def _same_candidate_payload(left: SpanCandidate, right: SpanCandidate) -> bool:
        """Return True when two overlapping candidates encode the same semantic payload."""
        if left.label != right.label:
            return False
        if left.label in {CANDIDATE_LABEL_ARTICLE, CANDIDATE_LABEL_TEMP_ARTICLE, CANDIDATE_LABEL_ANNEX_ARTICLE}:
            left_madde = left.data.get("madde", "")
            right_madde = right.data.get("madde", "")
            if left_madde and right_madde and left_madde != right_madde:
                return False
            for key in ("fikra", "bent"):
                left_val = left.data.get(key, "")
                right_val = right.data.get(key, "")
                if left_val and right_val and left_val != right_val:
                    return False
            return True
        if left.label == CANDIDATE_LABEL_PARAGRAPH:
            return left.data.get("fikra", "") == right.data.get("fikra", "")
        if left.label == CANDIDATE_LABEL_CLAUSE:
            return left.data.get("bent", "") == right.data.get("bent", "")
        if left.label in {CANDIDATE_LABEL_LAW_NAME, CANDIDATE_LABEL_LAW_ALIAS}:
            left_key = (left.data.get("kanun_no", ""), left.data.get("kanun_ad", ""))
            right_key = (right.data.get("kanun_no", ""), right.data.get("kanun_ad", ""))
            if all(left_key) and all(right_key):
                return left_key == right_key
        return collapse_ws(left.text).lower() == collapse_ws(right.text).lower()

    @staticmethod
    def _candidate_quality(candidate: SpanCandidate) -> tuple[int, int]:
        non_empty_data = sum(1 for value in candidate.data.values() if value)
        span_len = candidate.end_char - candidate.start_char
        return non_empty_data, span_len

    def _merge_candidate_pair(self, left: SpanCandidate, right: SpanCandidate) -> SpanCandidate:
        """Prefer richer candidate while unioning data fields."""
        primary, secondary = left, right
        if self._candidate_quality(right) > self._candidate_quality(left):
            primary, secondary = right, left
        merged_data = dict(primary.data)
        for key, value in secondary.data.items():
            if value and not merged_data.get(key):
                merged_data[key] = value
        return SpanCandidate(
            label=primary.label,
            start_char=primary.start_char,
            end_char=primary.end_char,
            text=primary.text,
            sentence_index=primary.sentence_index,
            token_start=primary.token_start,
            token_end=primary.token_end,
            data=merged_data,
        )

    def _legacy_sentence_resolution(self, doc) -> list[dict[str, str]]:
        """Legacy sentence-based fallback to avoid recall regression."""
        references: list[dict[str, str]] = []
        law_mentions = self._find_law_mentions(doc)
        atif_mentions = self._find_atif_mentions(doc)
        madde_mentions = self._find_madde_mentions_legacy(doc)
        text = doc.text

        last_law_no = ""
        last_law_ad = ""
        for sent in doc.sents:
            sent_start = sent.start_char
            sent_end = sent.end_char
            sent_laws = [m for m in law_mentions if sent_start <= m["start"] < sent_end]
            if sent_laws:
                for index, law in enumerate(sent_laws):
                    last_law_no = law["kanun_no"]
                    last_law_ad = law["kanun_ad"]
                    region_start = law["start"]
                    region_end = sent_laws[index + 1]["start"] if index + 1 < len(sent_laws) else sent_end
                    references.extend(
                        self._extract_region_refs_legacy(
                            text=text,
                            region_start=region_start,
                            region_end=region_end,
                            kanun_no=last_law_no,
                            kanun_ad=last_law_ad,
                            madde_mentions=madde_mentions,
                        )
                    )

            if last_law_no or last_law_ad:
                sent_atifs = [m for m in atif_mentions if sent_start <= m["start"] < sent_end]
                for atif in sent_atifs:
                    if self._looks_like_explicit_chain(text, atif["start"]):
                        continue
                    references.extend(
                        self._extract_region_refs_legacy(
                            text=text,
                            region_start=atif["start"],
                            region_end=sent_end,
                            kanun_no=last_law_no,
                            kanun_ad=last_law_ad,
                            madde_mentions=madde_mentions,
                        )
                    )
                if not sent_laws and self._has_reference_signal(text[sent_start:sent_end]):
                    references.extend(
                        self._extract_region_refs_legacy(
                            text=text,
                            region_start=sent_start,
                            region_end=sent_end,
                            kanun_no=last_law_no,
                            kanun_ad=last_law_ad,
                            madde_mentions=madde_mentions,
                        )
                    )
        return references

    def _resolve_law_for_candidate(
        self,
        article: SpanCandidate,
        sent_idx: int,
        by_sentence: dict[int, list[SpanCandidate]],
        state: DocumentContextState,
    ) -> tuple[str, str]:
        """Resolve law context in legal-window priority."""
        # 1) same span neighborhood
        same_sentence = by_sentence.get(sent_idx, [])
        overlap_laws = [
            c
            for c in same_sentence
            if c.label in {CANDIDATE_LABEL_LAW_NAME, CANDIDATE_LABEL_LAW_ALIAS}
            and not (c.end_char < article.start_char or c.start_char > article.end_char + 80)
        ]
        if overlap_laws:
            law = overlap_laws[-1]
            return law.data.get("kanun_no", ""), law.data.get("kanun_ad", "")

        # 2) same sentence fallback
        same_sent_laws = [c for c in same_sentence if c.label in {CANDIDATE_LABEL_LAW_NAME, CANDIDATE_LABEL_LAW_ALIAS}]
        if same_sent_laws:
            law = same_sent_laws[-1]
            return law.data.get("kanun_no", ""), law.data.get("kanun_ad", "")

        # 3) previous 1-2 sentence window
        for prev in range(max(0, sent_idx - 2), sent_idx):
            prev_laws = [
                c for c in by_sentence.get(prev, []) if c.label in {CANDIDATE_LABEL_LAW_NAME, CANDIDATE_LABEL_LAW_ALIAS}
            ]
            if prev_laws:
                law = prev_laws[-1]
                return law.data.get("kanun_no", ""), law.data.get("kanun_ad", "")

        # 4) document memory
        return state.last_explicit_law_no, state.last_explicit_law_ad

    @staticmethod
    def _nearest_article_context(
        resolved_articles: list[dict[str, str | int]],
        sent_idx: int,
    ) -> dict[str, str | int] | None:
        """Get nearest valid article context from current/previous sentence window."""
        if not resolved_articles:
            return None
        for item in reversed(resolved_articles):
            item_sent = int(item.get("sentence_index", -999))
            if sent_idx - item_sent <= 2:
                return item
        return resolved_articles[-1]

    @staticmethod
    def _sentence_index_for_char(sent_spans, char_pos: int) -> int:
        for idx, sent in enumerate(sent_spans):
            if sent.start_char <= char_pos < sent.end_char:
                return idx
        return max(0, len(sent_spans) - 1)

    def _find_law_mentions(self, doc) -> list[dict[str, str]]:
        mentions: list[dict[str, str]] = []

        for _, start, end in self.law_matcher(doc):
            span = doc[start:end]
            if len(span) < 3 or len(span) > 14:
                continue
            if sum(1 for token in span[1:] if any(ch.isdigit() for ch in token.text)) > 0:
                continue

            kanun_no = collapse_ws(span[0].text)
            kanun_ad_text = collapse_ws(" ".join(token.text for token in span[2:]))
            kanun_ad = normalize_kanun_adi(kanun_ad_text, kanun_no=kanun_no)

            mentions.append(
                {
                    "start": span.start_char,
                    "end": span.end_char,
                    "kanun_no": kanun_no,
                    "kanun_ad": kanun_ad,
                }
            )

        for _, start, end in self.phrase_matcher(doc):
            span = doc[start:end]
            kanun_ad = normalize_kanun_adi(span.text)
            kanun_no = self._infer_law_no_nearby(doc.text, span.start_char) or LAW_NAME_TO_NO.get(kanun_ad, "")

            mentions.append(
                {
                    "start": span.start_char,
                    "end": span.end_char,
                    "kanun_no": kanun_no,
                    "kanun_ad": kanun_ad,
                }
            )

        mentions.extend(self._find_abbreviation_mentions(doc))
        mentions.extend(self._find_regex_explicit_law_mentions(doc.text))

        return self._dedupe_mentions(mentions)

    def _find_regex_explicit_law_mentions(self, text: str) -> list[dict[str, str]]:
        """Use regex explicit law detection as additional high-recall hint source."""
        mentions: list[dict[str, str]] = []
        for mention in self.regex_fallback._find_explicit_mentions(text):
            mentions.append(
                {
                    "start": int(mention["start"]),
                    "end": int(mention["end"]),
                    "kanun_no": str(mention.get("kanun_no", "")),
                    "kanun_ad": str(mention.get("kanun_ad", "")),
                }
            )
        return mentions

    def _find_abbreviation_mentions(self, doc) -> list[dict[str, str]]:
        mentions: list[dict[str, str]] = []
        for _, start, end in self.abbr_matcher(doc):
            span = doc[start:end]
            abbr = re.sub(r"[^A-Za-z]", "", span.text.upper())
            kanun_ad = LAW_ABBREVIATIONS.get(abbr, "")
            if not kanun_ad:
                continue
            kanun_no = LAW_NAME_TO_NO.get(kanun_ad, "")
            mentions.append(
                {
                    "start": span.start_char,
                    "end": span.end_char,
                    "kanun_no": kanun_no,
                    "kanun_ad": kanun_ad,
                }
            )
        return mentions

    def _find_atif_mentions(self, doc) -> list[dict[str, int]]:
        mentions: list[dict[str, int]] = []
        for _, start, end in self.atif_matcher(doc):
            span = doc[start:end]
            mentions.append({"start": span.start_char, "end": span.end_char})
        mentions.sort(key=lambda item: (item["start"], item["end"]))
        return mentions

    def _find_madde_mentions(self, doc) -> list[dict[str, str]]:
        mentions: list[dict[str, str]] = []
        for _, start, end in self.madde_matcher(doc):
            span = doc[start:end]
            token_match = RE_MADDE_TOKEN.search(span.text)
            if not token_match:
                continue

            madde, slash_fikra, slash_bent = self._parse_composite_madde_token(token_match.group(0))
            if not madde:
                continue

            mentions.append(
                {
                    "start": span.start_char,
                    "end": span.end_char,
                    "madde": madde,
                    "fikra": slash_fikra,
                    "bent": slash_bent,
                }
            )

        for match in RE_MADDE.finditer(doc.text):
            token = match.group("madde_token").strip("()")
            madde, slash_fikra, slash_bent = self._parse_composite_madde_token(token)
            if not madde:
                continue
            mentions.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "madde": madde,
                    "fikra": slash_fikra,
                    "bent": slash_bent,
                }
            )

        mentions.sort(key=lambda item: (item["start"], item["end"]))
        return self._merge_madde_mentions(mentions)

    def _find_madde_mentions_legacy(self, doc) -> list[dict[str, str]]:
        """Legacy madde detection kept for stable sentence-resolution output."""
        mentions: list[dict[str, str]] = []
        for _, start, end in self.madde_matcher(doc):
            span = doc[start:end]
            token_match = RE_MADDE_TOKEN_LEGACY.search(span.text)
            if not token_match:
                continue

            madde, slash_fikra, slash_bent = parse_madde_token(token_match.group(0))
            if not madde:
                continue

            mentions.append(
                {
                    "start": span.start_char,
                    "end": span.end_char,
                    "madde": madde,
                    "fikra": slash_fikra,
                    "bent": slash_bent,
                }
            )

        for match in RE_MADDE_LEGACY.finditer(doc.text):
            token = match.group("madde_token").strip("()")
            madde, slash_fikra, slash_bent = parse_madde_token(token)
            if not madde:
                continue
            mentions.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "madde": madde,
                    "fikra": slash_fikra,
                    "bent": slash_bent,
                }
            )

        mentions.sort(key=lambda item: (item["start"], item["end"]))

        deduped: list[dict[str, str]] = []
        for mention in mentions:
            if deduped and mention["start"] < deduped[-1]["end"]:
                continue
            deduped.append(mention)
        return deduped

    def _extract_region_refs(
        self,
        text: str,
        region_start: int,
        region_end: int,
        kanun_no: str,
        kanun_ad: str,
        madde_mentions: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        refs: list[dict[str, str]] = []
        region_madde = [m for m in madde_mentions if region_start <= m["start"] < region_end]

        for idx, madde_item in enumerate(region_madde):
            next_madde_start = region_madde[idx + 1]["start"] if idx + 1 < len(region_madde) else region_end
            tail_end = min(next_madde_start, madde_item["end"] + 220)
            tail = text[madde_item["end"] : tail_end]
            fikra_values, bent_values = self._parse_verbose_components(tail)
            fikra = madde_item["fikra"] or (fikra_values[0] if fikra_values else "")
            bent_candidates = [madde_item["bent"]] if madde_item["bent"] else (bent_values or [""])

            for bent in bent_candidates:
                refs.append(
                    {
                        "type": REFERENCE_TYPE,
                        "kanun_no": kanun_no,
                        "kanun_ad": kanun_ad,
                        "madde": madde_item["madde"],
                        "fikra": fikra,
                        "bent": bent,
                        "source_text": snippet(text, max(region_start, madde_item["start"] - 40), min(region_end, madde_item["end"] + 140)),
                    }
                )

        if not region_madde:
            table_match = RE_TABLE_REF.search(text[region_start:region_end])
            if table_match:
                segment = text[region_start:region_end]
                left_ctx = segment[max(0, table_match.start() - 28) : table_match.start()].lower()
                if "sayılı" not in left_ctx and "sayili" not in left_ctx:
                    return refs
                abs_start = region_start + table_match.start()
                abs_end = region_start + table_match.end()
                refs.append(
                    {
                        "type": REFERENCE_TYPE,
                        "kanun_no": kanun_no,
                        "kanun_ad": kanun_ad,
                        "madde": "",
                        "fikra": "",
                        "bent": "",
                        "source_text": snippet(text, max(region_start, abs_start - 40), min(region_end, abs_end + 80)),
                    }
                )

        return refs

    def _extract_region_refs_legacy(
        self,
        text: str,
        region_start: int,
        region_end: int,
        kanun_no: str,
        kanun_ad: str,
        madde_mentions: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        refs: list[dict[str, str]] = []
        region_madde = [m for m in madde_mentions if region_start <= m["start"] < region_end]

        for idx, madde_item in enumerate(region_madde):
            next_start = region_madde[idx + 1]["start"] if idx + 1 < len(region_madde) else region_end
            tail_end = min(next_start, madde_item["end"] + 220)
            tail = text[madde_item["end"] : tail_end]
            fikra_values = self._extract_all_identifiers(RE_FIKRA_LEGACY, tail)
            bent_values = self._extract_all_identifiers(RE_BENT_LEGACY, tail, series_pattern=RE_BENT_SERIES)
            fikra = madde_item["fikra"] or (fikra_values[0] if fikra_values else "")
            bent_candidates = [madde_item["bent"]] if madde_item["bent"] else (bent_values or [""])

            for bent in bent_candidates:
                refs.append(
                    {
                        "type": REFERENCE_TYPE,
                        "kanun_no": kanun_no,
                        "kanun_ad": kanun_ad,
                        "madde": madde_item["madde"],
                        "fikra": fikra,
                        "bent": bent,
                        "source_text": snippet(text, max(region_start, madde_item["start"] - 40), min(region_end, madde_item["end"] + 140)),
                    }
                )

        if not region_madde:
            table_clause_match = RE_TABLE_FIKRA.search(text[region_start:region_end])
            if table_clause_match:
                abs_start = region_start + table_clause_match.start()
                abs_end = region_start + table_clause_match.end()
                refs.append(
                    {
                        "type": REFERENCE_TYPE,
                        "kanun_no": kanun_no,
                        "kanun_ad": kanun_ad,
                        "madde": normalize_identifier(table_clause_match.group("madde") or ""),
                        "fikra": "",
                        "bent": normalize_identifier(table_clause_match.group("bent") or ""),
                        "source_text": snippet(text, max(region_start, abs_start - 40), min(region_end, abs_end + 120)),
                    }
                )
            table_match = RE_TABLE_REF.search(text[region_start:region_end])
            if table_match:
                abs_start = region_start + table_match.start()
                abs_end = region_start + table_match.end()
                refs.append(
                    {
                        "type": REFERENCE_TYPE,
                        "kanun_no": kanun_no,
                        "kanun_ad": kanun_ad,
                        "madde": "",
                        "fikra": "",
                        "bent": "",
                        "source_text": snippet(text, max(region_start, abs_start - 40), min(region_end, abs_end + 80)),
                    }
                )

        return refs

    def _parse_composite_madde_token(self, token: str) -> tuple[str, str, str]:
        """Parse canonical composite forms such as `16/1-a` into components."""
        cleaned = collapse_ws(token).strip("()[]{}.,;:")
        cleaned = re.sub(r"\s*/\s*", "/", cleaned)
        cleaned = re.sub(r"\s*-\s*", "-", cleaned)
        cleaned = collapse_ws(cleaned)

        madde, fikra, bent = parse_madde_token(cleaned)
        if madde:
            return madde, fikra, bent

        canonical_match = RE_CANONICAL_COMPOSITE_TOKEN.match(cleaned)
        if not canonical_match:
            return "", "", ""
        madde = canonical_match.group("madde") or ""
        fikra = normalize_identifier(canonical_match.group("fikra") or "")
        bent = normalize_identifier(canonical_match.group("bent") or "")
        return madde, fikra, bent

    def _parse_verbose_components(self, text: str) -> tuple[list[str], list[str]]:
        """Parse verbose legal form tails (e.g. `1 inci fıkra (a) bendi`)."""
        fikra_values = self._extract_all_identifiers(RE_FIKRA, text)
        bent_values = self._extract_all_identifiers(RE_BENT, text)

        if not bent_values:
            return fikra_values, bent_values

        # If a paragraph exists, prioritize clause candidates that appear after it.
        fikra_match = RE_FIKRA.search(text)
        if fikra_match:
            tail = text[fikra_match.end() :]
            trailing_bents = self._extract_all_identifiers(RE_BENT, tail)
            if trailing_bents:
                bent_values = trailing_bents
        return fikra_values, bent_values

    def _merge_madde_mentions(self, mentions: list[dict[str, str]]) -> list[dict[str, str]]:
        """Merge duplicate mentions while preserving valid overlapping article spans."""
        if not mentions:
            return []
        merged: list[dict[str, str]] = []
        for mention in mentions:
            if not merged:
                merged.append(dict(mention))
                continue
            prev = merged[-1]
            overlaps = mention["start"] < prev["end"]
            same_article = mention.get("madde", "") == prev.get("madde", "")
            same_fikra = not (mention.get("fikra") and prev.get("fikra")) or mention.get("fikra", "") == prev.get("fikra", "")
            same_bent = not (mention.get("bent") and prev.get("bent")) or mention.get("bent", "") == prev.get("bent", "")

            if overlaps and same_article and same_fikra and same_bent:
                prev["start"] = min(prev["start"], mention["start"])
                prev["end"] = max(prev["end"], mention["end"])
                if mention.get("fikra") and not prev.get("fikra"):
                    prev["fikra"] = mention["fikra"]
                if mention.get("bent") and not prev.get("bent"):
                    prev["bent"] = mention["bent"]
                continue

            merged.append(dict(mention))
        return merged

    @staticmethod
    def _extract_identifier(pattern: re.Pattern[str], text: str) -> str:
        match = pattern.search(text)
        if not match:
            return ""
        value = SpacyReferenceAnnotator._identifier_from_match(match)
        return normalize_identifier(value)

    @staticmethod
    def _extract_all_identifiers(
        pattern: re.Pattern[str],
        text: str,
        series_pattern: re.Pattern[str] | None = None,
    ) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()

        if series_pattern is not None:
            for match in series_pattern.finditer(text):
                for item in extract_identifier_series(match.group("items")):
                    if item in seen:
                        continue
                    seen.add(item)
                    values.append(item)

        for match in pattern.finditer(text):
            value = SpacyReferenceAnnotator._identifier_from_match(match)
            normalized = normalize_identifier(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            values.append(normalized)
        return values

    @staticmethod
    def _identifier_from_match(match: re.Match[str]) -> str:
        return (
            match.groupdict().get("par")
            or match.groupdict().get("num")
            or match.groupdict().get("ord")
            or match.groupdict().get("let")
            or ""
        )

    @staticmethod
    def _looks_like_explicit_chain(text: str, start: int) -> bool:
        left_ctx = text[max(0, start - 40) : start].lower()
        return bool(re.search(r"\d{2,5}\s+say[ıi]l[ıi]", left_ctx))

    @staticmethod
    def _has_reference_signal(text: str) -> bool:
        lowered = text.lower()
        return "madd" in lowered or "fıkra" in lowered or "fikra" in lowered or "bent" in lowered or bool(RE_TABLE_REF.search(text))

    @staticmethod
    def _infer_law_no_nearby(text: str, span_start: int) -> str:
        left = text[max(0, span_start - 40) : span_start]
        match = RE_SAYILI_LEFT_CONTEXT.search(left)
        if not match:
            return ""
        return collapse_ws(match.group(1))

    @staticmethod
    def _dedupe_mentions(mentions: list[dict[str, str]]) -> list[dict[str, str]]:
        ordered = sorted(mentions, key=lambda item: (item["start"], -(item["end"] - item["start"])))
        unique: list[dict[str, str]] = []

        for mention in ordered:
            if not mention.get("kanun_ad"):
                continue
            if unique and mention["start"] < unique[-1]["end"]:
                prev = unique[-1]
                prev_score = (1 if prev.get("kanun_no") else 0, prev["end"] - prev["start"])
                new_score = (1 if mention.get("kanun_no") else 0, mention["end"] - mention["start"])
                if new_score > prev_score:
                    unique[-1] = mention
                continue
            unique.append(mention)

        for mention in unique:
            mention["kanun_ad"] = normalize_kanun_adi(mention.get("kanun_ad", ""), mention.get("kanun_no", ""))
            if not mention.get("kanun_no"):
                mention["kanun_no"] = LAW_NAME_TO_NO.get(mention["kanun_ad"], "")

        return unique


def annotate_documents(
    documents: list[dict[str, Any]],
    batch_size: int = 256,
    n_process: int = 1,
) -> list[dict[str, Any]]:
    """Annotate a list of input documents."""
    annotator = SpacyReferenceAnnotator()

    results: list[dict[str, Any]] = []

    stream = (
        (
            normalize_reference_text(str(document.get("text", ""))),
            {"doc_id": document.get("doc_id")},
        )
        for document in documents
    )

    try:
        for doc, ctx in annotator.nlp.pipe(stream, as_tuples=True, batch_size=batch_size, n_process=n_process):
            doc_id = ctx.get("doc_id")
            try:
                result = annotator._annotate_parsed_doc(doc)
                result = annotator._enforce_output_contract(result)
                result["doc_id"] = doc_id
            except Exception as exc:  # pragma: no cover - defensive path
                result = {
                    "doc_id": doc_id,
                    "references": [],
                    "ozelge_no": "",
                    "status": "error",
                    "error": str(exc),
                }
            results.append(result)
    except Exception:
        # Defensive recovery: if pipe-level failure occurs, process per-document with spaCy path.
        results = []
        for document in documents:
            doc_id = document.get("doc_id")
            text = str(document.get("text", ""))
            try:
                result = annotator.annotate_document(text)
                result = annotator._enforce_output_contract(result)
                result["doc_id"] = doc_id
                result["engine"] = "spacy_recovery"
            except Exception as exc:
                result = {
                    "doc_id": doc_id,
                    "references": [],
                    "ozelge_no": "",
                    "status": "error",
                    "error": str(exc),
                }
            results.append(result)

    return results


def load_documents(path: Path) -> list[dict[str, Any]]:
    """Load input documents from a JSON array file."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Input JSON must be an array: {path}")
    return data


def save_results(results: list[dict[str, Any]], path: Path) -> None:
    """Persist annotations to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


def main() -> None:
    """CLI entrypoint for spaCy baseline annotation."""
    parser = argparse.ArgumentParser(description="spaCy baseline reference annotator")
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT, help="Input documents JSON")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT, help="Output annotations JSON")
    parser.add_argument("--batch-size", type=int, default=256, help="spaCy nlp.pipe batch size")
    parser.add_argument("--n-process", type=int, default=1, help="spaCy nlp.pipe process count")
    args = parser.parse_args()

    documents = load_documents(args.input)
    results = annotate_documents(documents, batch_size=max(1, args.batch_size), n_process=max(1, args.n_process))
    save_results(results, args.output)

    success_count = sum(1 for item in results if item.get("status") == "success")
    total_refs = sum(len(item.get("references", [])) for item in results)

    print(f"Input docs: {len(documents)}")
    print(f"Successful annotations: {success_count}")
    print(f"Total references: {total_refs}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
