"""Legal-reference normalization and document-level compaction."""

from __future__ import annotations

import re
import unicodedata
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
# "229 ve müteakip maddelerinde" cites article 229 onwards; the article is 229.
# The same prose comes as "ve izleyen", "ve devamı", "ve takip eden", and with
# the typo "mütakip" -- all found in the GT750 data (27 Sep 2026).
_MUTEAKIP_RE = re.compile(
    r"\s+ve\s+(?:m[uü]teakip|m[uü]takip|izleyen|devam\w*|takip\s+eden|sonraki)"
    r"(?:\s+madde\w*)?\s*$",
    re.IGNORECASE,
)
_ARTICLE_RANGE_RE = re.compile(r"^(\d+)\s*(?:-|–|ila|ilâ)\s*(\d+)$", re.IGNORECASE)
_LEADING_WORD_RE = re.compile(r"^(?P<word>[^\W\d_]+)\.?\s*(?P<rest>.*)$")
# An article's qualifier names a different article ("geçici 3" is not 3), so
# it is kept -- but spelled one way. Compared folded and allowing one typing
# slip ("geçiçi", "mükrrer"): the number carries the identity, so folding the
# qualifier cannot merge two articles. "ek" is too short to allow a slip.
_QUALIFIERS = {"mukerrer": "mükerrer", "gecici": "geçici"}
# A trailing "/" plus a *lowercase* letter is a sub-part written inline
# ("6/b"); a trailing "/" plus an *uppercase* letter is part of the
# article's own name inserted by amendment (e.g. 5520 art. "32/A") and must
# never be split out. Case is therefore load-bearing in this class, not
# folded away.
_EMBEDDED_BENT_RE = re.compile(r"^(?P<madde>.*\S)\s*/\s*(?P<bent>[a-zçğıöşü])$")
_LAW_NAME_PUNCT = " .,:;-'\""
_LAW_NUMBER_PREFIX_RE = re.compile(r"^\d+\s*(?:sayılı|sayili)\s*", re.IGNORECASE)
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
    "onbirinci": "11",
    "on birinci": "11",
    "onikinci": "12",
    "on ikinci": "12",
    "onüçüncü": "13",
    "on üçüncü": "13",
    "onucuncu": "13",
    "on ucuncu": "13",
    "ondördüncü": "14",
    "on dördüncü": "14",
    "ondorduncu": "14",
    "on dorduncu": "14",
    "onbeşinci": "15",
    "on beşinci": "15",
    "onbesinci": "15",
    "on besinci": "15",
    "onaltıncı": "16",
    "on altıncı": "16",
    "onaltinci": "16",
    "on altinci": "16",
    "onyedinci": "17",
    "on yedinci": "17",
    "onsekizinci": "18",
    "on sekizinci": "18",
    "ondokuzuncu": "19",
    "on dokuzuncu": "19",
    "yirminci": "20",
}


def _ascii_fold(value: str) -> str:
    translation = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    return value.translate(translation)


def _fold(value: str) -> str:
    """Turkish-aware case fold: also strips combining marks left by NFKD.

    ``str.casefold()`` on Turkish uppercase ``İ`` yields ``i`` followed by
    U+0307 COMBINING DOT ABOVE, which ``_ascii_fold`` alone does not remove.
    Decomposing first and dropping combining characters neutralizes that
    before the existing ASCII-fold/casefold pipeline runs.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _ascii_fold(without_marks).casefold()


def normalize_law_number(value: Any) -> str:
    text = normalize_text(value)
    match = re.search(r"\d+", text.replace(".", ""))
    if not match:
        return text.casefold()
    # Strip leading zeros ("0193" == "193"), but a run of all zeros must
    # collapse to "0", never to the empty string.
    return match.group(0).lstrip("0") or "0"


def normalize_law_name(value: Any, *, law_number: str = "") -> str:
    text = normalize_text(value)
    # A leading "N sayılı" ("numbered N") restates the law number already
    # carried by the kanun_no field; strip it before alias lookup so it
    # cannot defeat matching, and strip it from the returned text too since
    # it is the same redundant prefix either way.
    text = _LAW_NUMBER_PREFIX_RE.sub("", text).strip(_LAW_NAME_PUNCT)
    folded = _SPACE_RE.sub(" ", text.casefold()).strip(_LAW_NAME_PUNCT)
    folded_ascii = _fold(folded)
    alias = LAW_NAME_ALIASES.get(folded) or LAW_NAME_ALIASES.get(folded_ascii)
    if alias:
        return alias
    if law_number in CANONICAL_LAW_BY_NO and (
        not text or folded in {"kanun", "kanunu", "sayılı kanun", "sayili kanun"}
    ):
        return CANONICAL_LAW_BY_NO[law_number]
    if law_number in CANONICAL_LAW_BY_NO:
        canonical = CANONICAL_LAW_BY_NO[law_number]
        if _fold(canonical) == folded_ascii:
            return canonical
    text = re.sub(r"\bKanunun(?:un|da|dan)?\b", "Kanunu", text, flags=re.IGNORECASE)
    return _SPACE_RE.sub(" ", text).strip(_LAW_NAME_PUNCT)


def _within_one_edit(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1
    short, long_ = sorted((left, right), key=len)
    return any(long_[:i] + long_[i + 1 :] == short for i in range(len(long_)))


def article_qualifier(word: str) -> str | None:
    """The canonical qualifier ("geçici", "mükerrer", "ek") a word spells, if any."""
    folded = _fold(word)
    if folded == "ek":
        return "ek"
    for plain, canonical in _QUALIFIERS.items():
        if len(folded) >= 5 and _within_one_edit(folded, plain):
            return canonical
    return None


def normalize_article(value: Any) -> str:
    text = normalize_text(value).strip(" .,:;\"'")
    text = _MUTEAKIP_RE.sub("", text).strip()
    text = _ARTICLE_SUFFIX_RE.sub("", text).strip()
    leading = _LEADING_WORD_RE.match(text)
    if leading and _fold(leading.group("word")) == "degisik":
        # "Değişik 3" is article 3 as amended, not a different article.
        text = leading.group("rest").strip()
        leading = _LEADING_WORD_RE.match(text)
    if leading and (qualifier := article_qualifier(leading.group("word"))):
        rest = re.sub(r"^madde\w*\s+", "", leading.group("rest"), flags=re.IGNORECASE)
        text = f"{qualifier} {rest}"
    text = _SPACE_RE.sub(" ", text).strip().casefold()
    article_range = _ARTICLE_RANGE_RE.match(text)
    if article_range:
        text = f"{article_range.group(1)} ila {article_range.group(2)}"
    return text


def normalize_extension(value: Any) -> str:
    text = normalize_text(value).strip(" .,:;\"'")
    match = _PARENS_RE.match(text)
    if match:
        text = match.group(1).strip()
    folded = text.casefold()
    return _ORDINALS.get(folded, folded)


def normalize_reference(raw: dict[str, Any]) -> dict[str, str]:
    validated = validate_reference_list([raw])[0]
    madde_value = validated["madde"]
    bent_value = validated["bent"]
    if not normalize_text(bent_value):
        # Only when bent is empty: an explicit annotation is never overwritten.
        embedded = _EMBEDDED_BENT_RE.match(normalize_text(madde_value))
        if embedded:
            madde_value = embedded.group("madde")
            bent_value = embedded.group("bent")
    law_number = normalize_law_number(validated["kanun_no"])
    law_name = normalize_law_name(validated["kanun_ad"], law_number=law_number)
    if not law_number and law_name in CANONICAL_NO_BY_NAME:
        law_number = CANONICAL_NO_BY_NAME[law_name]
    return {
        "kanun_no": law_number,
        "kanun_ad": normalize_law_name(law_name, law_number=law_number),
        "madde": normalize_article(madde_value),
        "fikra": normalize_extension(validated["fikra"]),
        "bent": normalize_extension(bent_value),
        "source_text": normalize_text(validated["source_text"]),
    }


def law_identity(reference: dict[str, str]) -> str:
    if reference["kanun_no"]:
        return f"no:{reference['kanun_no']}"
    if reference["kanun_ad"]:
        return f"name:{_fold(reference['kanun_ad'])}"
    return "unknown"


def core_identity(reference: dict[str, str]) -> tuple[str, str, str, str]:
    """Identify a provision: which law, and which article of it.

    The law name is deliberately absent whenever ``kanun_no`` is present. The
    number already names the law, and only eight statutes carry a canonical
    name in ``CANONICAL_LAW_BY_NO`` -- for every other law an empty name on one
    side and a written one on the other made two references to the same
    provision compare as different. ``law_identity`` still falls back to the
    folded name when there is no number, so an unnumbered law keeps its
    identity.
    """

    return (
        law_identity(reference),
        reference["kanun_no"],
        "" if reference["kanun_no"] else _fold(reference["kanun_ad"]),
        reference["madde"],
    )


def full_identity(reference: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (*core_identity(reference), reference["fikra"], reference["bent"])


def compact_references(references: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Drop rows that another row in the same set already accounts for.

    Two levels of the same rule. A row naming only a law goes when any article
    of that law is named; a row naming only an article goes when any paragraph
    or subparagraph of that article is named. `193/51` beside `193/51/1-8` is
    not a second claim about the ruling -- it is the same claim, written twice
    at two depths.

    Only the law-level half existed before, so a model that named both depths
    disagreed with an annotator who named one, and the document was routed
    YELLOW: a judge call and expert attention spent on a row that adds nothing.
    The measurement engine had already settled the question the other way --
    `benchmark.reference.matching.prepare_reference_set` collapses rows sharing
    (kanun_no, kanun_ad, madde), so on d000017 it reduced the model's seven rows
    to the annotator's exact six and scored a perfect match while the router
    called the same document a mismatch.

    Both decisions are computed against the whole input set rather than a
    running survivor, so the result cannot depend on row order -- a bucket
    depends on this, and a bucket that moved when its input was shuffled would
    not be a measurement.
    """
    normalized = [normalize_reference(reference) for reference in references]
    specific_laws = {law_identity(reference) for reference in normalized if reference["madde"]}
    specific_articles = {
        core_identity(reference)
        for reference in normalized
        if reference["fikra"] or reference["bent"]
    }
    compacted: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for reference in normalized:
        if not reference["madde"] and law_identity(reference) in specific_laws:
            continue
        if (
            not reference["fikra"]
            and not reference["bent"]
            and core_identity(reference) in specific_articles
        ):
            continue
        identity = full_identity(reference)
        if identity in seen:
            continue
        seen.add(identity)
        compacted.append(reference)
    return compacted


# Words that name the concept "statute" rather than any particular statute. A
# label built only from these discriminates nothing: an annotator who clipped
# "... İstanbul Su ve Kanalizasyon İdaresi Kanununun 21 inci maddesi" down to
# "Kanun" has not claimed that law 2560 is some other law.
_CONTENTLESS_LAW_WORDS = frozenset(
    {"kanun", "kanunu", "kanunun", "kanununun", "sayili", "bu", "mezkur", "ilgili", "adi", "gecen"}
)


def names_a_particular_law(kanun_ad: str) -> bool:
    """True when a law name carries something that distinguishes one law."""

    folded = _fold(kanun_ad)
    return any(word not in _CONTENTLESS_LAW_WORDS for word in folded.split())


def conflicting_law_identity(references: Iterable[dict[str, str]]) -> bool:
    """True when one statute number is given two different names.

    Only that direction. The mirror -- one name given to two numbers -- used to
    count as well, and it is wrong for Turkish tax rulings: a ruling may cite a
    repealed statute alongside the one that replaced it, and the two share a
    name. 1475 and 4857 are both the İş Kanunu, 5422 and 5520 both the Kurumlar
    Vergisi Kanunu, 4628 and 6446 both the Elektrik Piyasası Kanunu.

    Measured before removing it: across the 500 canonical ground-truth
    documents and M01's 100, that direction fired six times. Five were
    legitimate repealed/successor pairs -- four of them inside the
    expert-validated ground truth, which the router would therefore have called
    self-contradictory -- and the sixth, a model naming 7143 the Special
    Consumption Tax Law, was already RED because two of the annotator's core
    references were missing from it. It caught nothing on its own, it hid the
    real reason on the documents it did fire on, and it blocked the expert from
    saving a pair the ground truth already contains.

    Names that name no particular law are ignored entirely; see
    `names_a_particular_law`.
    """
    names_by_number: defaultdict[str, set[str]] = defaultdict(set)
    for reference in references:
        number = reference["kanun_no"]
        name = _fold(reference["kanun_ad"])
        if number and name and names_a_particular_law(reference["kanun_ad"]):
            names_by_number[number].add(name)
    return any(len(values) > 1 for values in names_by_number.values())
