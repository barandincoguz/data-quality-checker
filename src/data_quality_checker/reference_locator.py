"""Locate a known reference in raw document text -- without its quote.

Not extraction. The reference's identity (law, article, paragraph,
subparagraph) is already decided -- by the expert, or by annotator and model
agreeing -- and this module only finds *where in the document* that provision
is cited, so `context_windows.py` can put the reference in the training window
that actually shows it. `context_windows.py` needs a verbatim substring of the
body (`body.find`), so every span returned here is one.

The annotator's `source_text` is never read (user rule, 27 September 2026:
the annotators did not use the quote field as intended, so it decides
nothing). Everything is found from the reference's own fields.

How:

1. **Every article mention is parsed**, with the citation grammar built for
   GT750's quotes (`ner-project/scripts/gt750/quotes.py`): series ("148, 149
   ve mükerrer 257 nci maddeleri"), ranges ("638-642 inci madde"), "ve
   müteakip/izleyen", qualifiers (geçici, mükerrer, ek), subdivisions
   ("150/e", "17/4-k") and the paragraph/subparagraph chain after the article
   word ("maddesinin (1) numaralı fıkrasının (e) bendinde").
2. **Articles are compared the way the two scoring engines compare them**:
   each cited item goes through `normalization.normalize_reference`, so
   "Geçici 3", "GEÇİCİ 3", "geçiçi 3" and "Geçici Madde 3" are one article and
   an uppercase "32/A" stays apart from bent "32/a" (`test_engine_agreement.py`).
3. **Each mention gets a law** from its context: the law named immediately
   before it ("213 sayılı Vergi Usul Kanununun 153/A maddesi"), an anaphor
   ("aynı Kanunun", "anılan Kanunun") or, failing both, the last law the text
   named. A law named by title is tied to its number through the document's
   own "N sayılı X Kanunu" and a small registry.
4. **A reference is placed only on a mention of its article whose law does not
   contradict its own.** A mention explicitly of another law is never used.
   Among the rest: a mention whose paragraph chain does not contradict the
   reference before one that does; explicit law before inferred; the chain
   that says most about the reference; then the earliest.

An article the text does not cite at all is refused (`None`): that is evidence
the reference is wrong, and inventing a span would teach the model that an
article claim is supported by a passage about something else.

Measured on GT750 (27 Sep 2026; 4,356 references, their verified quotes used
only as the answer key): 92.7% placed on the gold quote itself; the other 7.2%
on another citation of the same provision (audited: the span names the law and
the article, or a range containing it); 3 refused (two table sections, one
Roman-numbered article). Re-spelling the article (case, "geçiçi", "Geçici
Madde", "ve izleyen") moves no placement: 3,861/3,861. The locator it replaced:
88.1%, 29 refused, 154/3,861 unmoved. On R01's live references every refusal
but one was a wrong or malformed reference ("80" for "mükerrer 80", law "758"
for "6758", a communiqué serial number read as an article).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .normalization import (
    CANONICAL_LAW_BY_NO,
    CANONICAL_NO_BY_NAME,
    LAW_NAME_ALIASES,
    _fold,
    article_qualifier,
    normalize_law_number,
    normalize_reference,
)

_LO = "a-zçğıöşü"
_UP = "A-ZÇĞİÖŞÜ"
_ORD_SUFFIX = r"(?:inci|ıncı|nci|ncı|üncü|uncu|ncu|ncü)"
_UNITS = {
    "birinci": 1,
    "ikinci": 2,
    "üçüncü": 3,
    "dördüncü": 4,
    "beşinci": 5,
    "altıncı": 6,
    "yedinci": 7,
    "sekizinci": 8,
    "dokuzuncu": 9,
}
_TENS_ORDINAL = {"onuncu": 10, "yirminci": 20, "otuzuncu": 30}
_TENS = {"on": 10, "yirmi": 20, "otuz": 30}
_ORD_WORD = rf"(?i:(?:(?:on|yirmi|otuz)\s*)?(?:{'|'.join(_UNITS)})|{'|'.join(_TENS_ORDINAL)}|son)"

# Paragraph / subparagraph chain -------------------------------------------
_PAREN = rf"\(\s*[0-9{_LO}{_UP}/\-]+\s*\)"
_NUMV = rf"\d+(?:\s*/\s*[0-9{_LO}{_UP}]+)?(?:\s*{_ORD_SUFFIX})?"
_LETTER = rf"[{_LO}]\)"
_ONE = rf"(?:{_PAREN}|{_ORD_WORD}|{_NUMV}|{_LETTER})"
_VALUES = rf"{_ONE}(?:\s*(?:,|ve|veya|ile|ila|-)\s*{_ONE})*"
_SUBWORD = rf"(?i:(?:alt\s*)?(?:fıkra|bend|bent|paragraf|cümle))[{_LO}]*"
_ELEMENT = rf"{_VALUES}\s*(?:numaralı\s+|nolu\s+|no\.?lu\s+)?{_SUBWORD}"
_CHAIN = rf"(?:\s+{_ELEMENT})*"
_ELEMENT_PARTS = re.compile(
    rf"(?P<vals>{_VALUES})\s*(?:numaralı\s+|nolu\s+|no\.?lu\s+)?(?P<word>{_SUBWORD})"
)

# Article mention ----------------------------------------------------------
# Qualifier spellings seen in rulings and annotations, typos included; every
# one is read through `normalization.article_qualifier`.
_QUALIFIER_WORD = (
    r"(?i:geçici|gecici|geçiçi|gecıci|mükerrer|mukerrer|mükrrer|mükerer|mukerer|ek|"
    r"değişik|degisik)"
)
_PREFIX = rf"(?:{_QUALIFIER_WORD}\s+(?:(?i:madde)\s+)?)?"
_NUM = rf"\d+(?:\s*/\s*[0-9{_LO}{_UP}]+)*(?:\s*-\s*[0-9{_LO}{_UP}]+)*"
_ORD = rf"(?:\s*['’`]?\s*{_ORD_SUFFIX}(?![{_LO}])|\.(?=\s)|['’`][{_LO}]+)?"
_ITEM = rf"\(?{_PREFIX}{_NUM}\)?{_ORD}"
_JOIN = r"\s*(?:,|ve|veya|ile|ila|ilâ|–|-)\s*"
_FOLLOWING = rf"(?:\s+ve\s+(?:müteakip|mütakip|izleyen|takip\s+eden|devam[{_LO}]*|sonraki))?"
# A title may sit between the number and the word: '3 üncü "Tanımlar" başlıklı maddesinde'.
_ARTICLE_WORD = rf"(?:\s*[“\"‘][^”\"’]{{1,80}}[”\"’]\s*başlıklı)?\s*(?:(?i:madde)[{_LO}]*|(?i:md)\.?(?![{_LO}]))"
# A mention starts at a word boundary -- or glued to the law word before it,
# as extraction sometimes leaves it: "Vergi Usul Kanununun267 inci maddesi".
_START = r"(?:(?<![\w/.])|(?<=[Kk]anunun)|(?<=[Kk]anununun)|(?<=KANUNUN)|(?<=KANUNUNUN))"

ARTICLE_PHRASE = re.compile(
    rf"{_START}{_ITEM}(?:{_JOIN}{_ITEM})*{_FOLLOWING}{_ARTICLE_WORD}{_CHAIN}"
)
# "Madde 5", "Ek Madde 6'da" -- the article word first.
WORD_FIRST_ARTICLE = re.compile(
    rf"(?<![\w])(?:{_QUALIFIER_WORD}\s+)?(?i:madde)\s+{_PREFIX}{_NUM}(?![\d])"
)
NUMBERED_ARTICLE = re.compile(
    rf"{_START}\(?{_PREFIX}{_NUM}\)?(?:\s*(?:,|ve|veya)\s*[0-9{_LO}{_UP}]{{1,3}}(?![\w]))*"
    rf"\s*(?:numaralı\s+)?{_SUBWORD}{_CHAIN}"
)
# "birinci maddesinde": the article number written as a word.
ORDINAL_ARTICLE = re.compile(rf"(?<![\w]){_ORD_WORD}\s+(?i:madde)[{_LO}]*{_CHAIN}")
# "Kanununun l inci maddesinde": text extraction read the digit 1 as a lowercase L.
L_FOR_ONE_ARTICLE = re.compile(rf"{_START}l\s*inci\s+(?i:madde)[{_LO}]*{_CHAIN}")
# The article word that closes the cited items (not the "Madde" of "Ek Madde 6").
_CLOSING_ARTICLE_WORD = re.compile(rf"(?i:madde)[{_LO}]*(?!\s+\d)|(?i:md)\.?(?![{_LO}\d])")
_ITEM_TOKEN = re.compile(
    rf"(?P<q>{_QUALIFIER_WORD}\s+(?:(?i:madde)\s+)?)?(?P<num>\d+)"
    rf"(?P<rest>(?:\s*/\s*[0-9{_LO}{_UP}]+)*(?:\s*-\s*[0-9{_LO}{_UP}]+)*)"
)

# Law names -----------------------------------------------------------------
_LAW_WORD = re.compile(rf"^[“\"'‘(]*(?i:kanun|yasa)[{_LO}{_UP}’']*[”\"'’)]*$")
_TITLE = r"(?:[“\"‘]|'')[^“”\"]{1,120}?(?:[”\"’]|'')\s*başlıklı\s+"
_LAW_BEFORE = re.compile(
    rf"(?P<law>(?:Kanun|KANUN|Yasa|YASA)[{_LO}{_UP}]*(?:['’][{_LO}]+)?)\s*(?:{_TITLE})?$"
)
_ACRONYM = re.compile(rf"^\([{_UP}]+\)[,;]?$")
_DETERMINERS = {"aynı", "anılan", "mezkur", "mezkûr", "bu", "ilgili"}
_SENTENCE_OPENERS = {
    "ayrıca",
    "ancak",
    "buna",
    "bununla",
    "diğer",
    "öte",
    "yine",
    "nitekim",
    "dolayısıyla",
    "öncelikle",
    "fakat",
    "ise",
    "hem",
    "gerek",
    "söz",
}
_JOINERS = {"ve", "ile", "Hakkında", "hakkında"}
_MAX_NAME_TOKENS = 16
_NUMBERED_LAW = re.compile(r"(?<![\d/.])(?P<no>\d{2,5})\s+(?:[Ss]ayılı|SAYILI)\s+")
_ANAPHOR_LAW = re.compile(
    rf"(?i:aynı|anılan|mezkur|mezkûr|söz\s+konusu|adı\s+geçen|bu)\s+(?:Kanun|kanun|KANUN)[{_LO}’']*"
)
_ACRONYM_NUMBER = {
    "VUK": "213",
    "GVK": "193",
    "KVK": "5520",
    "KDVK": "3065",
    "ÖTVK": "4760",
    "DVK": "488",
}
_ACRONYM_LAW = re.compile(rf"(?<![\w])(?P<acr>{'|'.join(_ACRONYM_NUMBER)})(?![\w])")
_NAMED_LAW = re.compile(
    rf"(?:(?!Kanun|KANUN)[{_UP}][{_LO}{_UP}'’.]*,?\s+(?:(?:ve|ile|Hakkında|hakkında)\s+)?){{1,14}}"
    rf"(?:Kanun|KANUN)[{_LO}{_UP}’']*"
)
_NON_TITLE_STEMS = {"", "bu", "ayni", "anilan", "soz konusu", "ilgili", "mezkur", "adi gecen"}


def _capitalised(token: str) -> bool:
    bare = token.strip("“”\"'‘’(),;:.")
    return bool(bare) and (bare[0].isupper() or bool(_ACRONYM.match(token)))


def _lower(value: str) -> str:
    return value.replace("İ", "i").replace("I", "ı").lower()


def _norm(value: Any) -> str:
    return re.sub(r"\s+", "", _lower(str(value or "")))


def _name_stem(name: str) -> str:
    """A law's title folded, without its trailing "Kanun…" word and case endings."""
    folded = _fold(name)
    folded = re.sub(r"^\d+\s*sayili\s*", "", folded)
    folded = re.sub(r"\b(kanun|yasa)\w*\b.*$", "", folded)
    folded = re.sub(r"[^a-z0-9 ]+", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


_REGISTRY_STEMS = {_name_stem(name): number for number, name in CANONICAL_LAW_BY_NO.items()}
# "KDV Kanunu", "ÖTV Kanunu", "GVK": the aliases the loop's own normalization knows.
_REGISTRY_STEMS.update(
    {
        _name_stem(alias if "kanun" in alias else f"{alias} kanunu"): CANONICAL_NO_BY_NAME[name]
        for alias, name in LAW_NAME_ALIASES.items()
        if name in CANONICAL_NO_BY_NAME
    }
)


# --- where the law name before a mention starts ---------------------------


def _law_start(text: str, start: int) -> int:
    """Start of the law name immediately before a mention; `start` if none."""
    before = text[max(0, start - 300) : start]
    law = _LAW_BEFORE.search(before)
    if not law:
        return start
    begin = start - len(before) + law.start("law")
    tokens = text[:begin].split(" ")
    if tokens and tokens[-1] == "":
        tokens.pop()
    taken: list[str] = []
    for token in reversed(tokens):
        if len(taken) >= _MAX_NAME_TOKENS or not token:
            break
        bare = token.strip("“”\"'‘’(),;:.")
        lowered = _lower(bare)
        if taken and _lower(taken[-1]) == "sayılı":
            if re.fullmatch(r"\d+", bare) and token[-1] not in ",;:.":
                taken.append(token)
            else:
                taken.pop()
            break
        if lowered in _SENTENCE_OPENERS or _LAW_WORD.match(token.rstrip(",;:.")):
            break
        if token[-1] in ",;:." and not (
            token[-1] == "," and _capitalised(token) and taken and _capitalised(taken[-1])
        ):
            break
        if lowered == "sayılı":
            taken.append(token)
            continue
        if lowered in _DETERMINERS:
            taken.append(token)
            break
        if _capitalised(token):
            taken.append(token)
            continue
        if token in _JOINERS and taken:
            taken.append(token)
            continue
        break
    while taken and (taken[-1] in _JOINERS or _lower(taken[-1]) == "sayılı"):
        taken.pop()
    name = " ".join(reversed(taken))
    return begin - (len(name) + 1 if name else 0)


def _law_end(text: str, start: int) -> int | None:
    """End of the last "Kanun…" word of the law title starting at `start`."""
    tokens = text[start : start + 400].split(" ")
    offset = start
    last = None
    for index, token in enumerate(tokens):
        core = token.rstrip(",;:.")
        if _LAW_WORD.match(core):
            last = offset + len(core)
            if token != core:
                break
        elif index == 0 and not _capitalised(token):
            break
        elif not (_capitalised(token) or token in _JOINERS):
            break
        elif token != core and not (
            token.endswith(",") and index + 1 < len(tokens) and _capitalised(tokens[index + 1])
        ):
            break
        offset += len(token) + 1
    return last


# --- which law each position of the text is talking about ------------------


@dataclass(frozen=True)
class _LawMention:
    start: int
    end: int
    number: str  # "" when the title is not tied to a number
    stem: str  # folded title, "" when only the number is written
    written: bool = True  # False for an anaphor ("aynı Kanunun"), which is only a guess
    amending: bool = False  # "2772 sayılı Kanunla değişik": names a law, owns no article here


_AMENDING = re.compile(r"\s*(?:ile\s+)?(?i:değişik|değiştiril\w*|eklen\w*|ihdas\w*)")


def _law_mentions(text: str) -> list[_LawMention]:
    """Every law the text names, in order; anaphors resolved to the previous one.

    A law that amends ("2772 sayılı Kanunla değişik 24/2 nci maddesi") is kept
    for law-level references but marked: it is not the article's law, so the
    article falls to the law before it.
    """
    found: list[tuple[int, int, str, str, str]] = []
    aliases: dict[str, str] = {}
    for match in _NUMBERED_LAW.finditer(text):
        end = _law_end(text, match.end()) or match.end()
        stem = _name_stem(text[match.end() : end])
        number = match.group("no").lstrip("0")
        if stem:
            aliases.setdefault(stem, number)
        found.append(
            (match.start(), end, "amending" if _AMENDING.match(text, end) else "law", number, stem)
        )
    covered = [(start, end) for start, end, *_ in found]
    covered += [
        (match.start(), (_law_end(text, match.end()) or match.end()))
        for match in _NUMBERED_LAW.finditer(text)
    ]
    for match in _NAMED_LAW.finditer(text):
        if any(
            start <= match.start() < end or start < match.end() <= end for start, end in covered
        ):
            continue
        stem = _name_stem(match.group(0))
        if stem in _NON_TITLE_STEMS:
            continue
        kind = "amending" if _AMENDING.match(text, match.end()) else "law"
        found.append((match.start(), match.end(), kind, "", stem))
    for match in _ACRONYM_LAW.finditer(text):
        found.append((match.start(), match.end(), "law", _ACRONYM_NUMBER[match.group("acr")], ""))
    for match in _ANAPHOR_LAW.finditer(text):
        found.append((match.start(), match.end(), "anaphor", "", ""))
    mentions: list[_LawMention] = []
    for start, end, kind, number, stem in sorted(found):
        if kind == "anaphor":
            owners = [m for m in mentions if not m.amending]
            if owners and not any(m.start <= start < m.end for m in mentions):
                previous = owners[-1]
                mentions.append(_LawMention(start, end, previous.number, previous.stem, False))
            continue
        number = number or aliases.get(stem, "") or _REGISTRY_STEMS.get(stem, "")
        mentions.append(_LawMention(start, end, number, stem, True, kind == "amending"))
    return mentions


def _reference_law(reference: Mapping[str, Any]) -> tuple[str, set[str]]:
    raw_number = str(reference.get("kanun_no") or "")
    number = normalize_law_number(raw_number) if raw_number else ""
    stems = {_name_stem(str(reference.get("kanun_ad") or ""))}
    if number in CANONICAL_LAW_BY_NO:
        stems.add(_name_stem(CANONICAL_LAW_BY_NO[number]))
    return number, {stem for stem in stems if stem not in _NON_TITLE_STEMS}


def _law_agrees(mention: _LawMention, number: str, stems: set[str]) -> bool | None:
    """True when the mention is the reference's law, False when it is known to be
    another, None when the text does not say (a title tied to no number that does
    not match the reference's own title is unknown, not a contradiction)."""
    if mention.number and number:
        return mention.number == number
    if mention.stem and stems:
        if any(
            mention.stem == stem
            or mention.stem.endswith(" " + stem)
            or stem.endswith(" " + mention.stem)
            for stem in stems
        ):
            return True
    return None


# --- which article a mention cites, and what it says about sub-parts ------


def _canonical_madde(value: str) -> str:
    return normalize_reference(
        {"kanun_no": "", "kanun_ad": "", "madde": value, "fikra": "", "bent": "", "source_text": ""}
    )["madde"]


def _item_rest(match: re.Match[str], madde: str) -> str | None:
    """The part of a cited item after the reference's article ("/1-a"), or None."""
    word = match.group("q").split()[0] if match.group("q") else ""
    qualifier = article_qualifier(word) if word else None
    number = match.group("num").lstrip("0") or "0"
    head = f"{qualifier} {number}" if qualifier else number
    rest = re.sub(r"\s+", "", match.group("rest"))
    candidates = [(head, rest)]
    letter = re.match(r"^/([A-ZÇĞİÖŞÜ])(?![0-9A-Za-zÇĞİÖŞÜçğıöşü])", rest)
    if letter:
        candidates.insert(0, (f"{head}/{letter.group(1)}", rest[len(letter.group(0)) :]))
    for candidate, remainder in candidates:
        if _canonical_madde(candidate) == madde:
            return remainder
    return None


def _cited_items(phrase: str, *, numbered: bool = False) -> str:
    """The part of a mention that lists articles -- never its paragraph chain.

    "23 üncü maddesinin 1 inci fıkrasının (a) bendi" cites article 23; the 1 is
    a paragraph. Reading numbers from the chain placed article-1 references on
    article 23 (GT750 benchmark, 27 Sep 2026).
    """
    closing = re.search(_SUBWORD, phrase) if numbered else _CLOSING_ARTICLE_WORD.search(phrase)
    return phrase[: closing.start()] if closing else phrase


def _named_rest(phrase: str, madde: str) -> str | None:
    """Does this mention cite `madde`? Returns what follows it in the item."""
    article_range = re.fullmatch(r"(.+?) ila (.+)", madde)
    if article_range:
        # "466 ila 469", "229 ila mükerrer 242": each end may carry a qualifier.
        ends = []
        for end in article_range.groups():
            qualified = re.fullmatch(r"(\D+?)\s+(\d+)", end)
            ends.append(
                rf"{_QUALIFIER_WORD}\s+{qualified.group(2)}" if qualified else re.escape(end)
            )
        pattern = rf"(?<!\d){ends[0]}\s*(?:ila|ilâ|-|–)\s*{ends[1]}(?!\d)"
        return "" if re.search(pattern, phrase) else None
    for match in _ITEM_TOKEN.finditer(phrase):
        rest = _item_rest(match, madde)
        if rest is not None:
            return rest
    return None


def _in_cited_range(phrase: str, madde: str) -> bool:
    """ "638-642 inci madde" names every article from 638 to 642."""
    if not madde.isdigit():
        return False
    return any(
        int(low) < int(high) and int(low) <= int(madde) <= int(high)
        for low, high in re.findall(r"(?<![\d/])(\d+)\s*(?:-|–|ila|ilâ)\s*(\d+)(?![\d/])", phrase)
    )


def _ordinal_value(word: str) -> str | None:
    parts = _lower(word).split()
    if len(parts) == 1:
        joined = re.match(rf"^({'|'.join(_TENS)})({'|'.join(_UNITS)})$", parts[0])
        if joined:
            parts = [joined.group(1), joined.group(2)]
    if len(parts) == 1:
        if parts[0] in _UNITS:
            return str(_UNITS[parts[0]])
        if parts[0] in _TENS_ORDINAL:
            return str(_TENS_ORDINAL[parts[0]])
        return None
    if len(parts) == 2 and parts[0] in _TENS and parts[1] in _UNITS:
        return str(_TENS[parts[0]] + _UNITS[parts[1]])
    return None


def _values(text: str) -> set[str]:
    found: set[str] = set()
    for match in re.finditer(rf"{_PAREN}|{_ORD_WORD}|{_NUMV}|{_LETTER}", text):
        token = match.group(0)
        if token.startswith("("):
            found.add(_norm(token.strip("()")))
        elif token.endswith(")"):
            found.add(_lower(token[0]))
        elif token[0].isdigit():
            found.add(_norm(re.sub(rf"\s*{_ORD_SUFFIX}$", "", token)))
        else:
            value = _ordinal_value(re.sub(r"\s+", " ", token))
            if value:
                found.add(value)
    return found


def _levels(phrase: str, rest: str | None, numbered: bool) -> tuple[list[set[str]], int]:
    """The sub-parts a mention names, top-down: [{"1"}, {"6"}, {"b"}]."""
    levels: list[set[str]] = []
    if rest:
        levels.extend({part} for part in re.split(r"[/-]", _norm(rest)) if part)
    elements = list(_ELEMENT_PARTS.finditer(phrase))
    if numbered:
        word = re.search(_SUBWORD, phrase)
        head = phrase[: word.start()] if word else phrase
        extra = {
            _norm(v) for v in re.findall(r"(?:,|ve|veya)\s*([0-9A-Za-zÇĞİÖŞÜ]{1,3})(?=\s|$)", head)
        }
        if levels:
            levels[-1] |= extra
        elements = [e for e in elements if word and e.start() >= word.end()]
    optional = 0
    for element in elements:
        values = _values(element.group("vals"))
        if not values:
            continue
        if len(values) == 1 and re.search(r"[/-]", next(iter(values))):
            atoms = [{a} for a in re.split(r"[/-]", next(iter(values))) if a]
        else:
            atoms = [values]
        levels.extend(atoms)
        is_alt = element.group("word").lower().startswith("alt")
        optional = optional + len(atoms) if is_alt else 0
    return levels, optional


def _row_parts(reference: Mapping[str, Any]) -> list[str]:
    parts: list[str] = []
    for field in ("fikra", "bent"):
        value = _norm(reference.get(field))
        if value:
            parts.extend(p for p in re.split(r"[/-]", value) if p)
    return parts


def _fit(parts: list[str], levels: list[set[str]], optional: int) -> tuple[bool, int]:
    """(does the mention's chain agree with the row's sub-parts, levels it leaves unsaid)."""
    best: tuple[bool, int] | None = None
    for depth in range(len(levels), len(levels) - optional - 1, -1):
        head = levels[:depth]
        if all(i < len(parts) and parts[i] in level for i, level in enumerate(head)):
            candidate = (True, max(len(parts) - len(head), 0))
            if best is None or candidate[1] < best[1]:
                best = candidate
    return best or (False, 0)


# --- the locator ----------------------------------------------------------

_PAIRS = {"(": ")", "“": "”", '"': '"', "‘": "’"}


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end:
        char = text[start]
        if char in ".…-–, " or (char in _PAIRS and _PAIRS[char] not in text[start + 1 : end]):
            start += 1
        else:
            break
    while end > start and text[end - 1] in " ,;:":
        end -= 1
    return start, end


def _law_rank(
    mentions: list[_LawMention], law_begin: int, start: int, number: str, stems: set[str]
) -> int | None:
    """0 explicit agreeing law, 1 inferred agreeing, 2 unknown, 3 inferred other; None explicit other."""
    mentions = [m for m in mentions if not m.amending]
    if law_begin < start:
        named = [m for m in mentions if law_begin <= m.start < start]
        if named:
            verdict = _law_agrees(named[-1], number, stems)
            if verdict is False:
                # Only a law actually written before the mention excludes it; an
                # anaphor ("bu Kanunun") is often the quoted law itself.
                return None if named[-1].written else 3
            if verdict:
                return 0 if named[-1].written else 1
    preceding = [m for m in mentions if m.start < start]
    if not preceding:
        return 2
    verdict = _law_agrees(preceding[-1], number, stems)
    if verdict is None:
        return 2
    return 1 if verdict else 3


@dataclass(frozen=True)
class Placement:
    """Where a reference was placed, and on what kind of evidence."""

    start: int
    end: int
    law: int  # 0 law written before it, 1 inferred, 2 unknown, 3 inferred other, 5 no law
    contradicts_subparts: bool  # the mention's paragraph chain disagrees with the row's


def locate_reference_offsets(
    document_text: str, reference: Mapping[str, Any]
) -> tuple[int, int] | None:
    """(start, end) of the citation in `document_text` that supports `reference`."""
    placement = place_reference(document_text, reference)
    return None if placement is None else (placement.start, placement.end)


def place_reference(document_text: str, reference: Mapping[str, Any]) -> Placement | None:
    """The best citation of `reference` in `document_text`, with its evidence kind."""
    normalized = normalize_reference(
        {
            field: str(reference.get(field) or "")
            for field in ("kanun_no", "kanun_ad", "madde", "fikra", "bent", "source_text")
        }
    )
    madde = normalized["madde"]
    number, stems = _reference_law(normalized)
    mentions = _law_mentions(document_text)
    if not madde:
        for mention in mentions:
            if _law_agrees(mention, number, stems):
                start, end = _trim(document_text, mention.start, mention.end)
                return Placement(start, end, 0, False)
        return None

    parts = _row_parts(normalized)
    ranked: list[tuple[int, int, int, int, int, int]] = []

    def consider(
        start: int,
        end: int,
        phrase: str,
        rest: str,
        numbered: bool,
        lawless: bool = False,
        in_range: bool = False,
    ) -> None:
        law_begin = _law_start(document_text, start)
        law = _law_rank(mentions, law_begin, start, number, stems)
        if law is None:
            return
        if lawless:
            law = 5
        levels, optional = _levels(phrase, rest, numbered)
        fits, missing = _fit(parts, levels, optional)
        # The law comes first: the right article of the wrong law is the wrong
        # provision, a different paragraph of the right article is not. A chain
        # that contradicts the row's sub-parts still cites the article -- kept,
        # but behind every mention of the same law standing that does not.
        # Among mentions of the agreeing law, one that names the article itself
        # comes before a range that merely contains it ("620-645 inci maddeleri").
        agreeing = 0 if law in (0, 1) else 1
        ranked.append(
            (
                agreeing,
                int(in_range),
                law,
                0 if fits else 1,
                missing if fits else 9,
                start,
                law_begin,
                end,
            )
        )

    for match in ARTICLE_PHRASE.finditer(document_text):
        items = _cited_items(match.group(0))
        rest = _named_rest(items, madde)
        if rest is not None:
            consider(match.start(), match.end(), match.group(0), rest, False)
        elif _in_cited_range(items, madde):
            consider(match.start(), match.end(), match.group(0), "", False, in_range=True)
    for match in ORDINAL_ARTICLE.finditer(document_text):
        word = re.match(_ORD_WORD, match.group(0))
        if word and _ordinal_value(re.sub(r"\s+", " ", word.group(0))) == madde:
            consider(match.start(), match.end(), match.group(0), "", False)
    if madde == "1":
        for match in L_FOR_ONE_ARTICLE.finditer(document_text):
            consider(match.start(), match.end(), match.group(0), "", False)
    for match in WORD_FIRST_ARTICLE.finditer(document_text):
        rest = _named_rest(match.group(0), madde)
        if rest is not None:
            consider(match.start(), match.end(), match.group(0), rest, False)
    for match in NUMBERED_ARTICLE.finditer(document_text):
        if ARTICLE_PHRASE.match(document_text, match.start()):
            continue
        rest = _named_rest(_cited_items(match.group(0), numbered=True), madde)
        if rest is None:
            continue
        lawless = _law_start(document_text, match.start()) == match.start()
        # With no law before it, "(6) numaralı bendine" is a subparagraph, not an
        # article. Only a subdivided number ("20/d fıkrasında") is kept, and only
        # as the last resort.
        if lawless and not re.match(r"\(?\s*\d+\s*/", match.group(0)):
            continue
        consider(match.start(), match.end(), match.group(0), rest, True, lawless)
    if not ranked:
        return None
    _, _, law, contradiction, _, start, law_begin, end = min(ranked)
    start, end = _trim(document_text, law_begin if law_begin < start else start, end)
    return Placement(start, end, law, bool(contradiction))


def locate_reference_span(document_text: str, reference: Mapping[str, str]) -> str | None:
    """A verbatim span of `document_text` that supports `reference`, or None."""
    offsets = locate_reference_offsets(document_text, reference)
    if offsets is None:
        return None
    return document_text[offsets[0] : offsets[1]] or None


def attach_missing_spans(
    documents: Mapping[Any, dict[str, Any]],
) -> tuple[dict[Any, dict[str, Any]], list[dict[str, str]]]:
    """Fill every empty `source_text` in `documents` with a located span.

    `documents` maps a document id to `{"text": ..., "references": [...]}`, in
    the shape `round_training_documents` reads off a release. Neither the
    mapping nor any reference dict inside it is mutated; a new structure is
    returned with each empty `source_text` replaced by a located span where
    one could be found.

    Also returns a refusal list of `{"doc_id", "kanun_no", "kanun_ad",
    "madde", "fikra", "bent"}`, one entry per reference the locator could not
    place -- this function only reports refusals, it never raises for them.
    `round_training_documents` is the layer that turns a non-empty refusal
    list into one `ContractError` naming all of them at once.
    """
    located: dict[Any, dict[str, Any]] = {}
    refusals: list[dict[str, str]] = []
    for doc_id, document in documents.items():
        text = document["text"]
        new_references: list[Any] = []
        for reference in document["references"]:
            source_text = reference.get("source_text") if isinstance(reference, dict) else None
            if isinstance(source_text, str) and source_text.strip():
                new_references.append(reference)
                continue
            span = locate_reference_span(text, reference)
            if span is None:
                refusals.append(
                    {
                        "doc_id": str(doc_id),
                        "kanun_no": str(reference.get("kanun_no", "")),
                        "kanun_ad": str(reference.get("kanun_ad", "")),
                        "madde": str(reference.get("madde", "")),
                        "fikra": str(reference.get("fikra", "")),
                        "bent": str(reference.get("bent", "")),
                    }
                )
                new_references.append(reference)
                continue
            new_references.append({**reference, "source_text": span})
        located[doc_id] = {**document, "references": new_references}
    return located, refusals
