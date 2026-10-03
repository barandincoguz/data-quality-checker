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
    "197": "Motorlu Taşıtlar Vergisi Kanunu",
    "213": "Vergi Usul Kanunu",
    "488": "Damga Vergisi Kanunu",
    "492": "Harçlar Kanunu",
    "1319": "Emlak Vergisi Kanunu",
    "2464": "Belediye Gelirleri Kanunu",
    "3065": "Katma Değer Vergisi Kanunu",
    "3194": "İmar Kanunu",
    "3218": "Serbest Bölgeler Kanunu",
    "4458": "Gümrük Kanunu",
    "4721": "Türk Medeni Kanunu",
    "4760": "Özel Tüketim Vergisi Kanunu",
    "4857": "İş Kanunu",
    "5018": "Kamu Mali Yönetimi ve Kontrol Kanunu",
    "5510": "Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu",
    "5520": "Kurumlar Vergisi Kanunu",
    "6098": "Türk Borçlar Kanunu",
    "6102": "Türk Ticaret Kanunu",
    "6183": "Amme Alacaklarının Tahsil Usulü Hakkında Kanun",
    "6802": "Gider Vergileri Kanunu",
}
# Names that more than one law has carried (a repealed law and its successor):
# a name alone never decides between them.
SHARED_LAW_NAMES = {"İş Kanunu", "Türk Ticaret Kanunu", "Gümrük Kanunu", "İmar Kanunu", "Kurumlar Vergisi Kanunu"}
CANONICAL_NO_BY_NAME = {
    **{name: number for number, name in CANONICAL_LAW_BY_NO.items() if name not in SHARED_LAW_NAMES},
    # One law carries this name; the number 7338 is shared (the 2021 amending law), so it
    # is absent from CANONICAL_LAW_BY_NO (number -> name) but the name decides the number.
    "Veraset ve İntikal Vergisi Kanunu": "7338",
}
# Every number a law name can stand for. A row without a number takes a number only
# from this set; a number outside it (a model's slip) never resolves the name.
VALID_LAW_NUMBERS_BY_NAME = {
    **{name: {number} for number, name in CANONICAL_LAW_BY_NO.items()},
    "Kurumlar Vergisi Kanunu": {"5422", "5520"},
    "İş Kanunu": {"931", "1475", "3008", "4857"},
    "Türk Ticaret Kanunu": {"6762", "6102"},
    "Gümrük Kanunu": {"1615", "4458"},
    "İmar Kanunu": {"6785", "3194"},
    "Medeni Kanun": {"743", "4721"},
    "Borçlar Kanunu": {"818", "6098"},
    "Veraset ve İntikal Vergisi Kanunu": {"7338"},
}

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
    "harclar kanunu": "Harçlar Kanunu",
    "harçlar kanunu": "Harçlar Kanunu",
    "gider vergileri kanunu": "Gider Vergileri Kanunu",
    "gider vergisi kanunu": "Gider Vergileri Kanunu",
    "emlak vergisi kanunu": "Emlak Vergisi Kanunu",
    "emlak kanunu": "Emlak Vergisi Kanunu",
    "belediye gelirleri kanunu": "Belediye Gelirleri Kanunu",
    "veraset ve intikal vergisi kanunu": "Veraset ve İntikal Vergisi Kanunu",
    "veraset ve intikal kanunu": "Veraset ve İntikal Vergisi Kanunu",
    "turk ticaret kanunu": "Türk Ticaret Kanunu",
    "türk ticaret kanunu": "Türk Ticaret Kanunu",
    "ttk": "Türk Ticaret Kanunu",
    "sosyal sigortalar ve genel saglik sigortasi kanunu": "Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu",
    "sosyal sigortalar ve genel sağlık sigortası kanunu": "Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu",
    "ssgsk": "Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu",
    "turk medeni kanunu": "Türk Medeni Kanunu",
    "türk medeni kanunu": "Türk Medeni Kanunu",
    "tmk": "Türk Medeni Kanunu",
    "motorlu tasitlar vergisi kanunu": "Motorlu Taşıtlar Vergisi Kanunu",
    "motorlu taşıtlar vergisi kanunu": "Motorlu Taşıtlar Vergisi Kanunu",
    "mtvk": "Motorlu Taşıtlar Vergisi Kanunu",
    "mtv kanunu": "Motorlu Taşıtlar Vergisi Kanunu",
    "serbest bolgeler kanunu": "Serbest Bölgeler Kanunu",
    "serbest bölgeler kanunu": "Serbest Bölgeler Kanunu",
    "sbk": "Serbest Bölgeler Kanunu",
    "turk borclar kanunu": "Türk Borçlar Kanunu",
    "türk borçlar kanunu": "Türk Borçlar Kanunu",
    "tbk": "Türk Borçlar Kanunu",
    "gumruk kanunu": "Gümrük Kanunu",
    "gümrük kanunu": "Gümrük Kanunu",
    "imar kanunu": "İmar Kanunu",
    "is kanunu": "İş Kanunu",
    "iş kanunu": "İş Kanunu",
    "aatuhk": "Amme Alacaklarının Tahsil Usulü Hakkında Kanun",
    "aatk": "Amme Alacaklarının Tahsil Usulü Hakkında Kanun",
    "amme alacaklarinin tahsil usulu hakkinda kanun": "Amme Alacaklarının Tahsil Usulü Hakkında Kanun",
    "amme alacaklarının tahsil usulü hakkında kanun": "Amme Alacaklarının Tahsil Usulü Hakkında Kanun",
    "amme alacaklari kanunu": "Amme Alacaklarının Tahsil Usulü Hakkında Kanun",
    "amme alacakları kanunu": "Amme Alacaklarının Tahsil Usulü Hakkında Kanun",
    "kmykk": "Kamu Mali Yönetimi ve Kontrol Kanunu",
    "kamu mali yonetimi ve kontrol kanunu": "Kamu Mali Yönetimi ve Kontrol Kanunu",
    "kamu mali yönetimi ve kontrol kanunu": "Kamu Mali Yönetimi ve Kontrol Kanunu",
}

_SPACE_RE = re.compile(r"\s+")
_ARTICLE_SUFFIX_RE = re.compile(
    r"\s*(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü)?\s*(?:madde(?:si|sinin|sinde|sinden)?|maddesi|maddesinde)?\s*$",
    re.IGNORECASE,
)
# "229 ve müteakip maddelerinde" cites article 229 onwards; the article is 229.
# The same prose comes as "ve izleyen", "ve devamı", "ve takip eden", and with
# the typo "mütakip" -- all found in the GT750 data (27 Sep 2026).
_MUTEAKIP_RE = re.compile(
    r"\s+ve\s+(?:m[uü]teakip|m[uü]takip|izleyen|devam\w*|takip\s+eden|sonraki)"
    r"(?:\s+madde\w*)?\s*$",
    re.IGNORECASE,
)
_ARTICLE_RANGE_RE = re.compile(r"^(\d+)\s*(?:-|–|—|ila|ilâ)\s*(\d+)$", re.IGNORECASE)
_LEADING_WORD_RE = re.compile(r"^(?P<word>[^\W\d_]+)\.?\s*(?P<rest>.*)$")
# An article's qualifier names a different article ("geçici 3" is not 3), so
# it is kept -- but spelled one way. Compared folded and allowing one typing
# slip ("geçiçi", "mükrrer"): the number carries the identity, so folding the
# qualifier cannot merge two articles. "ek" is too short to allow a slip.
_QUALIFIERS = {"mukerrer": "mükerrer", "gecici": "geçici"}
_QUALIFIER_ABBREVIATIONS = {"muk": "mükerrer", "gec": "geçici"}
# A reference names a law and an article -- nothing below the article (user
# decision, 1 Oct 2026). A part of the article written into `madde` itself is
# therefore cut off: "13/a" and "4-a" (a lowercase letter: a sub-part, never an
# article), "17/4" and "17/4-a" (a number after the slash), "10 (2)". A
# trailing "/" plus an *uppercase* letter is part of the article's own name,
# inserted by amendment (5520 art. "32/A"), and is kept -- in upper case, so
# that normalizing an already normalized value changes nothing. Case is
# load-bearing in this class, not folded away.
_INSERTED_ARTICLE_RE = re.compile(
    r"^(?P<head>.*\S)\s*/\s*(?P<letter>[A-ZÇĞİÖŞÜ](?:\s*-\s*[A-ZÇĞİÖŞÜ])?)$"
)
_SUBPART_TAILS = (
    re.compile(r"\s*\(\s*[0-9a-zçğıöşü]{1,3}\s*\)$"),
    re.compile(r"\s*/\s*\d[^/]*$"),
    re.compile(r"\s*/\s*[a-zçğıöşü](?:\s*-\s*\S+)?$"),
    re.compile(r"\s*-\s*[a-zçğıöşü]$"),
    re.compile(r"(?<=/[A-ZÇĞİÖŞÜ])\s*-\s*\d+(?:\.\d+)*$"),
)
_LAW_NAME_PUNCT = " .,:;-'\""
_LAW_NUMBER_PREFIX_RE = re.compile(r"^\d+\s*(?:sayılı|sayili)\s*", re.IGNORECASE)


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


# A decree, circular, ruling or date number is not a law number: "2007/13033"
# and "89/14391" (BKK), "2013/6", "2006-32/32", "24/12/2012", "24.12.2012" --
# any value with "/" or "-" between digit groups, and a dotted date. Reading
# its first run of digits made two decrees one identity and a year a law
# (engine audit, 2 Oct 2026); it is kept whole, as the evaluator keeps it
# (`annotator.reference.common.normalize_kanun_no`, the same rule).
NON_LAW_NUMBER_RE = re.compile(r"\d\s*[/-]\s*\d|(?<!\d)\d{1,2}\s*\.\s*\d{1,2}\s*\.\s*\d{2,4}(?!\d)")
_DOTTED_DATE_RE = re.compile(r"(?<!\d)\d{1,2}\s*\.\s*\d{1,2}\s*\.\s*\d{2,4}(?!\d)")
#: Written for "no number": not a number, and not a name either.
EMPTY_LAW_NUMBER_WORDS = {"null", "none", "nan", "yok", "-", "—", "–", "n/a"}


def normalize_law_number(value: Any) -> str:
    text = normalize_text(value)
    if text.strip().casefold() in EMPTY_LAW_NUMBER_WORDS:
        return ""
    if NON_LAW_NUMBER_RE.search(text):
        text = _DOTTED_DATE_RE.sub(lambda m: re.sub(r"\s*\.\s*", "/", m.group(0)), text)
        compact = re.sub(r"[^0-9/-]", "", text).strip("/-")
        return re.sub(r"(?<!\d)0+(?=\d)", "", compact)
    cleaned_float = re.sub(r"\.0+$", "", text.strip())
    match = re.search(r"\d+", cleaned_float.replace(".", ""))
    if not match:
        # No digits: not a number. A law *name* written into the number field
        # is read as the name by `normalize_reference`.
        return ""
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
    # A dotted abbreviation ("G.V.K.", "V.U.K.") is the abbreviation.
    undotted = folded.replace(".", "") if re.fullmatch(r"(?:[^\W\d_]\.\s*)+[^\W\d_]?", folded) else folded
    alias = (LAW_NAME_ALIASES.get(folded) or LAW_NAME_ALIASES.get(folded_ascii)
             or LAW_NAME_ALIASES.get(undotted) or LAW_NAME_ALIASES.get(_fold(undotted)))
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
    # The written abbreviations "Mük." and "Geç." (engine audit, 2 Oct 2026).
    if folded in _QUALIFIER_ABBREVIATIONS:
        return _QUALIFIER_ABBREVIATIONS[folded]
    for plain, canonical in _QUALIFIERS.items():
        if len(folded) >= 5 and _within_one_edit(folded, plain):
            return canonical
    return None


def strip_article_subpart(text: str) -> str:
    """The article a `madde` value names, with any part of it cut off."""
    previous = None
    while text != previous:
        previous = text
        for pattern in _SUBPART_TAILS:
            text = pattern.sub("", text).strip()
    return text


# A list mark written before the article ("-1/1 inci maddesinde", KDV rulings)
# is not part of it. No article starts with one; GT750 and the annotators never
# write one; M000 v2 copied it from the text at update 25 (1 Oct 2026).
_LEADING_LIST_MARK_RE = re.compile(r"^[-–—•·*]+\s*")


# Engine audit (2 Oct 2026): spellings a model or an annotator writes that name
# the article before them, read the same way by the evaluator
# (`annotator.reference.common.normalize_madde`), pinned by
# tests/test_engine_agreement.py.
#: "94'üncü", "41’inci", "geçici 9'uncu": an apostrophe and the ordinal after a number.
APOSTROPHE_ORDINAL_RE = re.compile(
    r"(?<=\d)\s*['’`´]\s*(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü)?", re.IGNORECASE
)
#: "madde 94", "md. 94", "m.94", "madde: 94", "md: 94", "m: 94" before the number; "94 md." after it.
_ARTICLE_WORD_BEFORE_RE = re.compile(
    r"^(?:madde[^\W\d_]*|md\.?|m\.?)[:\s-]+(?=\d)|^(?:madde[^\W\d_]*|md\.|m\.)(?=\d)",
    re.IGNORECASE,
)
_ARTICLE_ABBREVIATION_AFTER_RE = re.compile(r"\s+(?:md|mad)\.?$", re.IGNORECASE)
#: "(17/4-r)", KDV rulings' way of writing article 17's part: the brackets
#: enclose the whole value. A bare "(17)" is a paragraph or item number, not an
#: article, and is left to the unreadable-article rule below.
ENCLOSED_ARTICLE_RE = re.compile(r"^\(\s*(?P<inner>[^()]*?/[^()]*?|[^\W\d_]+\s+\d[^()]*?)\s*\)$")
#: "32 / A" is "32/A".
_SLASH_SPACING_RE = re.compile(r"\s*/\s*")


def _normalize_range_span(span_text: str) -> str:
    article_range = _ARTICLE_RANGE_RE.match(span_text)
    if article_range:
        start, end = article_range.group(1), article_range.group(2)
        if int(end) > int(start):
            return f"{start} ila {end}"
        if int(end) == int(start):
            return start
        if "ila" not in span_text and "ilâ" not in span_text:
            # "17-4" (from "17-4-r") is article 17 and its part, not a span.
            return start
    return span_text


def _normalize_article_once(value: Any) -> str:
    text = normalize_text(value).strip(" .,:;\"'")
    text = _LEADING_LIST_MARK_RE.sub("", text)
    text = APOSTROPHE_ORDINAL_RE.sub("", text)
    text = _ARTICLE_WORD_BEFORE_RE.sub("", text)
    text = _MUTEAKIP_RE.sub("", text).strip()
    text = _ARTICLE_SUFFIX_RE.sub("", text).strip()
    text = _ARTICLE_ABBREVIATION_AFTER_RE.sub("", text).strip(" .")
    enclosed = ENCLOSED_ARTICLE_RE.match(text)
    if enclosed:
        text = enclosed.group("inner").strip()
    text = _SLASH_SPACING_RE.sub("/", text)
    text = strip_article_subpart(text)
    leading = _LEADING_WORD_RE.match(text)
    if leading and _fold(leading.group("word")) == "degisik":
        # "Değişik 3" is article 3 as amended, not a different article.
        text = leading.group("rest").strip()
        leading = _LEADING_WORD_RE.match(text)
    if leading and (qualifier := article_qualifier(leading.group("word"))):
        rest = _ARTICLE_WORD_BEFORE_RE.sub("", leading.group("rest")).strip()
        rest = _normalize_range_span(rest)
        text = f"{qualifier} {rest}"
    text = _SPACE_RE.sub(" ", text).strip()
    inserted = _INSERTED_ARTICLE_RE.match(text)
    if inserted:
        letter = re.sub(r"\s+", "", inserted.group("letter"))
        return f"{inserted.group('head').casefold()}/{letter}"
    text = text.casefold()
    return _normalize_range_span(text)


def normalize_article(value: Any) -> str:
    """The article a `madde` value names, in one spelling.

    Applying it to its own output changes nothing (a fixed point is taken: a
    value such as "II/B-5.1" once read "ii/b-5.1" and, normalized again, "ii").
    A value that names something but no article this can read ("(17)", a
    paragraph number) is kept as written, folded, so that it can match no
    article and is never mistaken for a law-level row (engine audit,
    2 Oct 2026).
    """
    text = _normalize_article_once(value)
    for _ in range(4):
        again = _normalize_article_once(text)
        if again == text:
            break
        text = again
    if not text:
        written = _SPACE_RE.sub(" ", normalize_text(value)).strip()
        if re.search(r"[^\W_]", written):
            return written.casefold()
    return text


def normalize_reference(raw: dict[str, Any], *, resolve_number_from_name: bool = True) -> dict[str, str]:
    """One reference in one spelling.

    `resolve_number_from_name=False` keeps the number as written (a number the
    name itself states, "6183 sayılı Kanun", counts as written): the matcher
    (`reference_matching`) compares written numbers, so that a name shared by a
    repealed law and its successor ("Kurumlar Vergisi Kanunu": 5422 and 5520)
    is matched by name and never resolved to the wrong one.
    """
    validated = validate_reference_list([raw])[0]
    law_number = normalize_law_number(validated["kanun_no"])
    raw_name = validated["kanun_ad"]
    if (not law_number and validated["kanun_no"].strip() and not raw_name.strip()
            and validated["kanun_no"].strip().casefold() not in EMPTY_LAW_NUMBER_WORDS):
        # A name written into the number field ("Gelir Vergisi Kanunu") is the name.
        raw_name = validated["kanun_no"]
    if not law_number:
        # "6183 sayılı Kanun" in the name field carries the number.
        numbered = re.match(r"^\s*(\d{1,4})\s*(?:sayılı|sayili)\b", normalize_text(raw_name), re.IGNORECASE)
        if numbered:
            law_number = numbered.group(1).lstrip("0") or "0"
    law_name = normalize_law_name(raw_name, law_number=law_number)
    if resolve_number_from_name and not law_number and law_name in CANONICAL_NO_BY_NAME:
        law_number = CANONICAL_NO_BY_NAME[law_name]
    return {
        "kanun_no": law_number,
        "kanun_ad": normalize_law_name(law_name, law_number=law_number),
        "madde": normalize_article(validated["madde"]),
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


def compact_references(references: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """One row per provision: drop rows another row in the same set accounts for.

    A row naming only a law goes when any article of that law is named, and a
    second row naming the same law and article goes ("Geçici 21" beside
    "geçici 21"; before 1 Oct 2026, "193/51" beside "193/51/1-8").

    The law-level decision is computed against the whole input set rather than
    a running survivor, and the survivor of a repeated provision is its first
    row, so the identities kept cannot depend on row order -- a bucket depends
    on this, and a bucket that moved when its input was shuffled would not be a
    measurement.
    """
    normalized = [normalize_reference(reference) for reference in references]
    specific_laws = {law_identity(reference) for reference in normalized if reference["madde"]}
    compacted: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for reference in normalized:
        if not reference["madde"] and law_identity(reference) in specific_laws:
            continue
        identity = core_identity(reference)
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
