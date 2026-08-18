"""Legal-reference normalization and document-level compaction."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from .contracts import validate_reference_list
from .text import normalize_text

CANONICAL_LAW_BY_NO = {
    "193": "Gelir Vergisi Kanunu",
    "213": "Vergi Usul Kanunu",
    "488": "Damga Vergisi Kanunu",
    "5520": "Kurumlar Vergisi Kanunu",
    "3065": "Katma Değer Vergisi Kanunu",
    "4760": "Özel Tüketim Vergisi Kanunu",
    "5018": "Kamu Mali Yönetimi ve Kontrol Kanunu",
    "6183": "Amme Alacaklarının Tahsil Usulü Hakkında Kanun",
}
CANONICAL_NO_BY_NAME = {name: number for number, name in CANONICAL_LAW_BY_NO.items()}

LAW_NAME_ALIASES = {
    "vuk": "Vergi Usul Kanunu",
    "vergi usul kanunu": "Vergi Usul Kanunu",
    "gvk": "Gelir Vergisi Kanunu",
    "gelir vergisi kanunu": "Gelir Vergisi Kanunu",
    "kvk": "Kurumlar Vergisi Kanunu",
    "kurumlar vergisi kanunu": "Kurumlar Vergisi Kanunu",
    "kdvk": "Katma Değer Vergisi Kanunu",
    "kdv kanunu": "Katma Değer Vergisi Kanunu",
    "katma deger vergisi kanunu": "Katma Değer Vergisi Kanunu",
    "katma değer vergisi kanunu": "Katma Değer Vergisi Kanunu",
    "otvk": "Özel Tüketim Vergisi Kanunu",
    "ötvk": "Özel Tüketim Vergisi Kanunu",
    "otv kanunu": "Özel Tüketim Vergisi Kanunu",
    "ötv kanunu": "Özel Tüketim Vergisi Kanunu",
    "ozel tuketim vergisi kanunu": "Özel Tüketim Vergisi Kanunu",
    "özel tüketim vergisi kanunu": "Özel Tüketim Vergisi Kanunu",
    "dvk": "Damga Vergisi Kanunu",
    "damga vergisi kanunu": "Damga Vergisi Kanunu",
}

_SPACE_RE = re.compile(r"\s+")
_ARTICLE_SUFFIX_RE = re.compile(
    r"\s*(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü)?\s*(?:madde(?:si|sinin|sinde|sinden)?|maddesi|maddesinde)?\s*$",
    re.IGNORECASE,
)
_PARENS_RE = re.compile(r"^[\[(]\s*(.*?)\s*[\])]$")
_ORDINALS = {
    "birinci": "1",
    "ikinci": "2",
    "üçüncü": "3",
    "ucuncu": "3",
    "dördüncü": "4",
    "dorduncu": "4",
    "beşinci": "5",
    "besinci": "5",
    "altıncı": "6",
    "altinci": "6",
    "yedinci": "7",
    "sekizinci": "8",
    "dokuzuncu": "9",
    "onuncu": "10",
}


def _ascii_fold(value: str) -> str:
    translation = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    return value.translate(translation)


def normalize_law_number(value: Any) -> str:
    text = normalize_text(value)
    match = re.search(r"\d+", text.replace(".", ""))
    return match.group(0) if match else text.casefold()


def normalize_law_name(value: Any, *, law_number: str = "") -> str:
    text = normalize_text(value)
    folded = _SPACE_RE.sub(" ", text.casefold()).strip(" .,:;-'\"")
    folded_ascii = _ascii_fold(folded)
    alias = LAW_NAME_ALIASES.get(folded) or LAW_NAME_ALIASES.get(folded_ascii)
    if alias:
        return alias
    if law_number in CANONICAL_LAW_BY_NO and (
        not text or folded in {"kanun", "kanunu", "sayılı kanun", "sayili kanun"}
    ):
        return CANONICAL_LAW_BY_NO[law_number]
    if law_number in CANONICAL_LAW_BY_NO:
        canonical = CANONICAL_LAW_BY_NO[law_number]
        if _ascii_fold(canonical.casefold()) == folded_ascii:
            return canonical
    text = re.sub(r"\bKanunun(?:un|da|dan)?\b", "Kanunu", text, flags=re.IGNORECASE)
    return _SPACE_RE.sub(" ", text).strip()


def normalize_article(value: Any) -> str:
    text = normalize_text(value).strip(" .,:;\"'")
    text = _ARTICLE_SUFFIX_RE.sub("", text).strip()
    text = re.sub(r"^(mukerrer|mükerrer)\s*", "mükerrer ", text, flags=re.IGNORECASE)
    text = re.sub(r"^gecici\s*", "geçici ", text, flags=re.IGNORECASE)
    text = re.sub(r"^ek\s*", "ek ", text, flags=re.IGNORECASE)
    return _SPACE_RE.sub(" ", text).strip().casefold()


def normalize_extension(value: Any) -> str:
    text = normalize_text(value).strip(" .,:;\"'")
    match = _PARENS_RE.match(text)
    if match:
        text = match.group(1).strip()
    folded = text.casefold()
    return _ORDINALS.get(folded, folded)


def normalize_reference(raw: dict[str, Any]) -> dict[str, str]:
    validated = validate_reference_list([raw])[0]
    law_number = normalize_law_number(validated["kanun_no"])
    law_name = normalize_law_name(validated["kanun_ad"], law_number=law_number)
    if not law_number and law_name in CANONICAL_NO_BY_NAME:
        law_number = CANONICAL_NO_BY_NAME[law_name]
    return {
        "kanun_no": law_number,
        "kanun_ad": normalize_law_name(law_name, law_number=law_number),
        "madde": normalize_article(validated["madde"]),
        "fikra": normalize_extension(validated["fikra"]),
        "bent": normalize_extension(validated["bent"]),
        "source_text": normalize_text(validated["source_text"]),
    }


def law_identity(reference: dict[str, str]) -> str:
    if reference["kanun_no"]:
        return f"no:{reference['kanun_no']}"
    if reference["kanun_ad"]:
        return f"name:{_ascii_fold(reference['kanun_ad'].casefold())}"
    return "unknown"


def core_identity(reference: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        law_identity(reference),
        reference["kanun_no"],
        _ascii_fold(reference["kanun_ad"].casefold()),
        reference["madde"],
    )


def full_identity(reference: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (*core_identity(reference), reference["fikra"], reference["bent"])


def compact_references(references: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = [normalize_reference(reference) for reference in references]
    specific_laws = {law_identity(reference) for reference in normalized if reference["madde"]}
    compacted: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for reference in normalized:
        if not reference["madde"] and law_identity(reference) in specific_laws:
            continue
        identity = full_identity(reference)
        if identity in seen:
            continue
        seen.add(identity)
        compacted.append(reference)
    return compacted


def conflicting_law_identity(references: Iterable[dict[str, str]]) -> bool:
    names_by_number: defaultdict[str, set[str]] = defaultdict(set)
    numbers_by_name: defaultdict[str, set[str]] = defaultdict(set)
    for reference in references:
        number = reference["kanun_no"]
        name = _ascii_fold(reference["kanun_ad"].casefold())
        if number and name:
            names_by_number[number].add(name)
            numbers_by_name[name].add(number)
    return any(len(values) > 1 for values in names_by_number.values()) or any(
        len(values) > 1 for values in numbers_by_name.values()
    )
