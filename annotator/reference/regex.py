"""Regex baseline annotator that emits Gemini-compatible reference output."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from annotator.reference.common import (
        CANONICAL_LAW_BY_NO,
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

    from annotator.reference.common import (
        CANONICAL_LAW_BY_NO,
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "test_data.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark" / "predictions" / "regex_reference_annotations.json"

RE_SENTENCE = re.compile(r"[^.!?;\n]+(?:[.!?;](?=\s|$)|\n)?", re.UNICODE)
RE_TABLE_REF = re.compile(r"\b(?:cetvel|tablo|liste)\w*\b", re.IGNORECASE | re.UNICODE)

RE_LAW_EXPLICIT = re.compile(
    r"(?P<kanun_no>\d{2,5}(?:/\d{1,5})?)\s+say[ıi]l[ıi]\s+"
    r"(?P<kanun_ad>(?:[A-Za-z0-9ÇĞİÖŞÜçğıöşü()'’`\-]+\s+){0,16}?Kanun(?:u|un|unun|una|unda|undan|la|daki|da|a)?)",
    re.IGNORECASE | re.UNICODE,
)

RE_LAW_NUMBER_ONLY = re.compile(
    r"(?P<kanun_no>\d{2,5}(?:/\d{1,5})?)\s+say[ıi]l[ıi]\s+kanun(?:u|un|unun|una|unda|undan)?",
    re.IGNORECASE | re.UNICODE,
)

RE_LAW_ANAPHORA = re.compile(
    r"(?P<atif>(?:ayn[ıi]|an[ıi]lan|ilgili|bu|mezkur|bahse\s+konu|s[öo]z\s+konusu|adı\s+ge[çc]en)\s+kanun\w*|kanunun|kanuna|kanunda|kanundan)",
    re.IGNORECASE | re.UNICODE,
)

RE_LAW_ABBR = re.compile(
    r"\b(?P<abbr>VUK|GVK|KDVK|KDV|KVK|ÖTVK|OTVK|ÖTV|OTV|DVK)(?:['’`][A-Za-zÇĞİÖŞÜçğıöşü]+)?\b",
    re.IGNORECASE | re.UNICODE,
)

RE_MADDE = re.compile(
    r"(?P<madde_token>(?:(?:m[üu]kerrer|ge[çc]ici|ek)\s+)?\d+(?:/\d+(?:-[A-Za-zÇĞİÖŞÜçğıöşü])?|/[A-Za-zÇĞİÖŞÜçğıöşü]|-[A-Za-zÇĞİÖŞÜçğıöşü])?|\(\d+(?:/\d+(?:-[A-Za-zÇĞİÖŞÜçğıöşü])?|/[A-Za-zÇĞİÖŞÜçğıöşü]|-[A-Za-zÇĞİÖŞÜçğıöşü])?\)|[Ee]k\s+[Mm]adde\s+\d+)"
    r"\s*(?:'?(?:inci|ıncı|üncü|uncu|nci|ncı|ncu|ncü))?\s*\.?\s*madd\w*",
    re.IGNORECASE | re.UNICODE,
)

RE_FIKRA = re.compile(
    r"(?:\((?P<par>[A-Za-z0-9]+)\)|(?P<num>\d+)|(?P<ord>birinci|ikinci|ucuncu|dorduncu|besinci|altinci|yedinci|sekizinci|dokuzuncu|onuncu))"
    r"\s*(?:numaral[ıi]\s+)?f[ıi]kra\w*",
    re.IGNORECASE | re.UNICODE,
)

RE_BENT = re.compile(
    r"(?:\((?P<par>[A-Za-z0-9]+)\)|(?P<num>\d+)|(?P<ord>birinci|ikinci|ucuncu|dorduncu|besinci|altinci|yedinci|sekizinci|dokuzuncu|onuncu)|\b(?P<let>[A-Za-zÇĞİÖŞÜçğıöşü])\b)"
    r"\s*(?:numaral[ıi]\s+)?ben[dt]\w*",
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

LAW_NAME_TO_NO = {
    law_name: law_no
    for law_no, law_name in CANONICAL_LAW_BY_NO.items()
    if law_no.isdigit() and law_name not in {"Kanun", "Kanun Hükmünde Kararname"}
}


class RegexReferenceAnnotator:
    """Rule-based baseline annotator for legal references."""

    def annotate_document(self, text: str) -> dict[str, Any]:
        """Annotate a single document in Gemini-compatible shape."""
        text = normalize_reference_text(text)
        references: list[dict[str, str]] = []
        last_kanun_no = ""
        last_kanun_ad = ""

        for sentence_match in RE_SENTENCE.finditer(text):
            sentence = sentence_match.group(0)
            explicit_mentions = self._find_explicit_mentions(sentence)

            if explicit_mentions:
                for idx, mention in enumerate(explicit_mentions):
                    last_kanun_no = mention["kanun_no"]
                    last_kanun_ad = mention["kanun_ad"]

                    region_start = mention["start"]
                    region_end = explicit_mentions[idx + 1]["start"] if idx + 1 < len(explicit_mentions) else len(sentence)
                    refs = self._extract_region_refs(
                        sentence=sentence,
                        region_start=region_start,
                        region_end=region_end,
                        kanun_no=last_kanun_no,
                        kanun_ad=last_kanun_ad,
                    )
                    references.extend(refs)

            if last_kanun_no:
                for match in RE_LAW_ANAPHORA.finditer(sentence):
                    if self._looks_like_explicit_chain(sentence, match.start()):
                        continue
                    refs = self._extract_region_refs(
                        sentence=sentence,
                        region_start=match.start(),
                        region_end=len(sentence),
                        kanun_no=last_kanun_no,
                        kanun_ad=last_kanun_ad,
                    )
                    references.extend(refs)

                if not explicit_mentions and self._has_reference_signal(sentence):
                    refs = self._extract_region_refs(
                        sentence=sentence,
                        region_start=0,
                        region_end=len(sentence),
                        kanun_no=last_kanun_no,
                        kanun_ad=last_kanun_ad,
                    )
                    references.extend(refs)

        cleaned = deduplicate_references(references)
        for ref in cleaned:
            ref["type"] = REFERENCE_TYPE

        return {
            "references": cleaned,
            "ozelge_no": extract_ozelge_no(text),
            "status": "success",
        }

    def _find_explicit_mentions(self, sentence: str) -> list[dict[str, str | int]]:
        mentions: list[dict[str, str | int]] = []

        for match in RE_LAW_EXPLICIT.finditer(sentence):
            kanun_no = collapse_ws(match.group("kanun_no"))
            kanun_ad = normalize_kanun_adi(match.group("kanun_ad"), kanun_no=kanun_no)
            mentions.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "kanun_no": kanun_no,
                    "kanun_ad": kanun_ad,
                }
            )

        for match in RE_LAW_NUMBER_ONLY.finditer(sentence):
            kanun_no = collapse_ws(match.group("kanun_no"))
            kanun_ad = canonical_law_name_by_no(kanun_no) or "Kanun"
            mentions.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "kanun_no": kanun_no,
                    "kanun_ad": kanun_ad,
                }
            )

        for match in RE_LAW_ABBR.finditer(sentence):
            abbr_raw = match.group("abbr")
            abbr = re.sub(r"[^A-Za-z]", "", abbr_raw.upper())
            kanun_ad = LAW_ABBREVIATIONS.get(abbr, "")
            if not kanun_ad:
                continue
            kanun_no = LAW_NAME_TO_NO.get(kanun_ad, "")
            mentions.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "kanun_no": kanun_no,
                    "kanun_ad": kanun_ad,
                }
            )

        mentions.sort(key=lambda item: (int(item["start"]), -(int(item["end"]) - int(item["start"]))))

        filtered: list[dict[str, str | int]] = []
        for mention in mentions:
            if filtered and int(mention["start"]) < int(filtered[-1]["end"]):
                continue
            filtered.append(mention)

        return filtered

    def _extract_region_refs(
        self,
        sentence: str,
        region_start: int,
        region_end: int,
        kanun_no: str,
        kanun_ad: str,
    ) -> list[dict[str, str]]:
        """Extract references from a sentence segment bound to one law context."""
        region = sentence[region_start:region_end]
        refs: list[dict[str, str]] = []

        madde_matches = list(RE_MADDE.finditer(region))
        if madde_matches:
            for idx, madde_match in enumerate(madde_matches):
                madde_token = madde_match.group("madde_token").strip("()")
                madde, slash_fikra, slash_bent = parse_madde_token(madde_token)
                if not madde:
                    continue

                next_start = madde_matches[idx + 1].start() if idx + 1 < len(madde_matches) else len(region)
                tail_end = min(next_start, madde_match.end() + 220)
                tail = region[madde_match.end() : tail_end]
                fikra = slash_fikra or self._extract_identifier(RE_FIKRA, tail)
                bent_values = [slash_bent] if slash_bent else self._extract_all_identifiers(RE_BENT, tail, series_pattern=RE_BENT_SERIES)
                if not bent_values:
                    bent_values = [""]

                local_start = max(0, madde_match.start() - 40)
                local_end = min(len(region), madde_match.end() + 160)
                source_text = snippet(region, local_start, local_end)

                for bent in bent_values:
                    refs.append(
                        {
                            "type": REFERENCE_TYPE,
                            "kanun_no": kanun_no,
                            "kanun_ad": kanun_ad,
                            "madde": madde,
                            "fikra": fikra,
                            "bent": bent,
                            "source_text": source_text,
                        }
                    )

        if not madde_matches:
            table_clause_match = RE_TABLE_FIKRA.search(region)
            if table_clause_match:
                refs.append(
                    {
                        "type": REFERENCE_TYPE,
                        "kanun_no": kanun_no,
                        "kanun_ad": kanun_ad,
                        "madde": normalize_identifier(table_clause_match.group("madde") or ""),
                        "fikra": "",
                        "bent": normalize_identifier(table_clause_match.group("bent") or ""),
                        "source_text": snippet(region, max(0, table_clause_match.start() - 40), min(len(region), table_clause_match.end() + 120)),
                    }
                )

        if not madde_matches:
            table_match = RE_TABLE_REF.search(region)
            if table_match:
                refs.append(
                    {
                        "type": REFERENCE_TYPE,
                        "kanun_no": kanun_no,
                        "kanun_ad": kanun_ad,
                        "madde": "",
                        "fikra": "",
                        "bent": "",
                        "source_text": snippet(region, max(0, table_match.start() - 40), table_match.end() + 80),
                    }
                )

        return refs

    @staticmethod
    def _extract_identifier(pattern: re.Pattern[str], text: str) -> str:
        match = pattern.search(text)
        if not match:
            return ""
        value = match.group("par") or match.group("num") or match.group("ord") or match.groupdict().get("let") or ""
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
                    if item not in seen:
                        seen.add(item)
                        values.append(item)

        for match in pattern.finditer(text):
            value = match.group("par") or match.group("num") or match.group("ord") or match.groupdict().get("let") or ""
            normalized = normalize_identifier(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            values.append(normalized)

        return values

    @staticmethod
    def _looks_like_explicit_chain(sentence: str, start: int) -> bool:
        left_ctx = sentence[max(0, start - 40) : start].lower()
        return bool(re.search(r"\d{2,5}\s+say[ıi]l[ıi]", left_ctx))

    @staticmethod
    def _has_reference_signal(sentence: str) -> bool:
        lowered = sentence.lower()
        return "madd" in lowered or "fıkra" in lowered or "fikra" in lowered or "bent" in lowered or bool(RE_TABLE_REF.search(sentence))


def annotate_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate a list of input documents."""
    annotator = RegexReferenceAnnotator()
    results: list[dict[str, Any]] = []

    for document in documents:
        doc_id = document.get("doc_id")
        text = document.get("text", "")
        try:
            result = annotator.annotate_document(text)
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
    """CLI entrypoint for regex baseline annotation."""
    parser = argparse.ArgumentParser(description="Regex baseline reference annotator")
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT, help="Input documents JSON")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT, help="Output annotations JSON")
    args = parser.parse_args()

    documents = load_documents(args.input)
    results = annotate_documents(documents)
    save_results(results, args.output)

    success_count = sum(1 for item in results if item.get("status") == "success")
    total_refs = sum(len(item.get("references", [])) for item in results)

    print(f"Input docs: {len(documents)}")
    print(f"Successful annotations: {success_count}")
    print(f"Total references: {total_refs}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
