"""Shared helpers for reference-style legal annotation outputs."""

from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict
from typing import Iterable

REFERENCE_TYPE = "kanun_madde_referansi"
# A reference is a law and an article of it (user decision, 1 Oct 2026); no
# field below the article exists. The archived five-field schema is at git tag
# archive/gt750-v1-fikra-bent.
REFERENCE_FIELDS = ("type", "kanun_no", "kanun_ad", "madde", "source_text")
CORE_FIELDS = ("kanun_no", "kanun_ad", "madde")

# A part of an article written into `madde` itself is cut off -- "13/a",
# "4-a" (a lowercase letter is a sub-part, never an article), "17/4",
# "17/4-a", "10 (2)" -- while an inserted article's uppercase letter ("32/A",
# "20/A-B") is part of its name and stays. Mirrors
# `data_quality_checker.normalization.strip_article_subpart` (pinned by
# tests/test_engine_agreement.py).
_SUBPART_TAILS = (
    re.compile(r"\s*\(\s*[0-9a-zçğıöşü]{1,3}\s*\)$"),
    re.compile(r"\s*/\s*\d[^/]*$"),
    re.compile(r"\s*/\s*[a-zçğıöşü](?:\s*-\s*\S+)?$"),
    re.compile(r"\s*-\s*[a-zçğıöşü]$"),
    re.compile(r"(?<=/[A-ZÇĞİÖŞÜ])\s*-\s*\d+(?:\.\d+)*$"),
)


def strip_article_subpart(text: str) -> str:
    """The article a `madde` value names, with any part of it cut off."""
    previous = None
    while text != previous:
        previous = text
        for pattern in _SUBPART_TAILS:
            text = pattern.sub("", text).strip()
    return text

ORDINAL_SUFFIX_RE = r"(?:[i\u0131u\u00fc]nc[i\u0131u\u00fc]|nc[i\u0131u\u00fc])"

RE_OZELGE_NO = re.compile(
    r"Say\u0131\s*:\s*(?P<ozelge_no>[^\n]{5,120}?)\s+(?:\d{1,2}[./]\d{1,2}[./]\d{2,4})",
    re.UNICODE,
)
RE_MULTI_SPACE = re.compile(r"\s+")
RE_NON_NO_CHARS = re.compile(r"[^0-9A-Za-z/-]+")
RE_INLINE_MADDE_DOT = re.compile(r"(?<=\d)\.(?=\s*madd)", re.IGNORECASE)

GENERIC_LAW_NAMES = {"", "kanun", "kanunu", "kanunun", "kanuna", "kanunda", "kanundan"}

CANONICAL_LAW_BY_NO = {
    "90": "Kanun Hükmünde Kararname",
    "193": "Gelir Vergisi Kanunu",
    "197": "Motorlu Taşıtlar Vergisi Kanunu",
    "213": "Vergi Usul Kanunu",
    "375": "Kanun Hükmünde Kararname",
    "488": "Damga Vergisi Kanunu",
    "492": "Harçlar Kanunu",
    "1219": "Tababet ve Şuabatı San'atlarının Tarzı İcrasına Dair Kanun",
    "1319": "Emlak Vergisi Kanunu",
    "2464": "Belediye Gelirleri Kanunu",
    "2547": "Yükseköğretim Kanunu",
    "2634": "Turizmi Teşvik Kanunu",
    "2872": "Çevre Kanunu",
    "2914": "Yüksek Öğretim Personel Kanunu",
    "3065": "Katma Değer Vergisi Kanunu",
    "3226": "Finansal Kiralama Kanunu",
    "3308": "Çıraklık ve Mesleki Eğitim Kanunu",
    "3713": "Terörle Mücadele Kanunu",
    "4389": "Bankalar Kanunu",
    "4572": "Tarım Satış Kooperatif ve Birlikleri Hakkında Kanun",
    "4628": "Elektrik Piyasası Kanunu",
    "4721": "Türk Medeni Kanunu",
    "4737": "Endüstri Bölgeleri Kanunu",
    "4760": "Özel Tüketim Vergisi Kanunu",
    "4958": "Sosyal Sigortalar Kanunu",
    "5018": "Kamu Mali Yönetimi ve Kontrol Kanunu",
    "5070": "Elektronik İmza Kanunu",
    "5300": "Tarım Ürünleri Lisanslı Depoculuk Kanunu",
    "5434": "Türkiye Cumhuriyeti Emekli Sandığı Kanunu",
    "5502": "Sosyal Güvenlik Kurumu Kanunu",
    "5510": "Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu",
    "5520": "Kurumlar Vergisi Kanunu",
    "5684": "Sigortacılık Kanunu",
    "5746": "Araştırma, Geliştirme ve Tasarım Faaliyetlerinin Desteklenmesi Hakkında Kanun",
    "5811": "Bazı Varlıkların Milli Ekonomiye Kazandırılması Hakkında Kanun",
    "5901": "Türk Vatandaşlığı Kanunu",
    "6458": "Yabancılar ve Uluslararası Koruma Kanunu",
    "6502": "Tüketicinin Korunması Hakkında Kanun",
    "6802": "Gider Vergileri Kanunu",
    "7033": "Sanayinin Geliştirilmesi ve Üretimin Desteklenmesi Amacıyla Bazı Kanun ve Kanun Hükmünde Kararnamelerde Değişiklik Yapılmasına Dair Kanun",
    "7061": "Bazı Vergi Kanunları ile Diğer Bazı Kanunlarda Değişiklik Yapılmasına Dair Kanun",
    "7104": "Katma Değer Vergisi Kanunu ve Bazı Kanunlar ile 178 Sayılı Kanun Hükmünde Kararnamede Değişiklik Yapılmasına Dair Kanun",
    "7326": "Bazı Alacakların Yeniden Yapılandırılması ile Bazı Kanunlarda Değişiklik Yapılmasına İlişkin Kanun",
    "7338": "Veraset ve İntikal Vergisi Kanunu",
    "7456": "6/2/2023 Tarihinde Meydana Gelen Depremlerin Yol Açtığı Ekonomik Kayıpların Telafisi İçin Ek Motorlu Taşıtlar Vergisi İhdası ile Bazı Kanunlarda ve 375 Sayılı Kanun Hükmünde Kararnamede Değişiklik Yapılması Hakkında Kanun",
    "2012/3305": "Bakanlar Kurulu Kararı",
    "2012/1": "Yatırımlarda Devlet Yardımları Hakkında Kararın Uygulamasına İlişkin Tebliğ",
}

LAW_ABBREVIATIONS = {
    "VUK": "Vergi Usul Kanunu",
    "GVK": "Gelir Vergisi Kanunu",
    "KDVK": "Katma Değer Vergisi Kanunu",
    "KDV": "Katma Değer Vergisi Kanunu",
    "KVK": "Kurumlar Vergisi Kanunu",
    "OTVK": "Özel Tüketim Vergisi Kanunu",
    "OTV": "Özel Tüketim Vergisi Kanunu",
    "DVK": "Damga Vergisi Kanunu",
}

LAW_NAME_ALIASES = {
    "VERGIUSULKANUNU": "Vergi Usul Kanunu",
    "GELIRVERGISIKANUNU": "Gelir Vergisi Kanunu",
    "KURUMLARVERGISIKANUNU": "Kurumlar Vergisi Kanunu",
    "KATMADEGERVERGISIKANUNU": "Katma Değer Vergisi Kanunu",
    "KATMADEGERVERGISIKDVKANUNU": "Katma Değer Vergisi Kanunu",
    "KDVKANUNU": "Katma Değer Vergisi Kanunu",
    "OZELTUKETIMVERGISIKANUNU": "Özel Tüketim Vergisi Kanunu",
    "OTVKANUNU": "Özel Tüketim Vergisi Kanunu",
    "DAMGAVERGISIKANUNU": "Damga Vergisi Kanunu",
    "HARCLARKANUNU": "Harçlar Kanunu",
    "BELEDIYEGELIRLERIKANUNU": "Belediye Gelirleri Kanunu",
    "EMLAKVERGISIKANUNU": "Emlak Vergisi Kanunu",
    "ELEKTRONIKIMZAKANUNU": "Elektronik İmza Kanunu",
    "TURKMEDENIKANUNU": "Türk Medeni Kanunu",
    "TURKVATANDASLIGIKANUNU": "Türk Vatandaşlığı Kanunu",
    "TURIZMTESVIKKANUNU": "Turizmi Teşvik Kanunu",
    "ENDUSTRIBOLGELERIKANUNU": "Endüstri Bölgeleri Kanunu",
    "GIDERVERGILERIKANUNU": "Gider Vergileri Kanunu",
    "VERASETVEINTIKALVERGISIKANUNU": "Veraset ve İntikal Vergisi Kanunu",
    "MOTORLUTASITLARVERGISIKANUNU": "Motorlu Taşıtlar Vergisi Kanunu",
    "KAMUMALIYONETIMIVEKONTROLKANUNU": "Kamu Mali Yönetimi ve Kontrol Kanunu",
    "SOSYALSIGORTALARVEGENELSAGLIKSIGORTASIKANUNU": "Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu",
    "SOSYALGUVENLIKKURUMUKANUNU": "Sosyal Güvenlik Kurumu Kanunu",
    "YABANCILARVEULUSLARARASIKORUMAKANUNU": "Yabancılar ve Uluslararası Koruma Kanunu",
    "TUKETICININKORUNMASIHAKKINDAKANUN": "Tüketicinin Korunması Hakkında Kanun",
    "TABABETVESUABATISANATLARININTARZIICRASINADAIRKANUN": "Tababet ve Şuabatı San'atlarının Tarzı İcrasına Dair Kanun",
    "ARASTIRMAVEGELISTIRMEFAALIYETLERININDESTEKLENMESIHAKKINDAKANUN": "Araştırma, Geliştirme ve Tasarım Faaliyetlerinin Desteklenmesi Hakkında Kanun",
    "ARASTIRMAGELISTIRMEVETASARIMFAALIYETLERININDESTEKLENMESIHAKKINDAKANUN": "Araştırma, Geliştirme ve Tasarım Faaliyetlerinin Desteklenmesi Hakkında Kanun",
    "ARASTIRMAVEGELISTIRMEFAALIYETLERININDESTEKLENMESIHAKKINDAKANUNU": "Araştırma, Geliştirme ve Tasarım Faaliyetlerinin Desteklenmesi Hakkında Kanun",
    "VERGIUSULKANUNUNUN": "Vergi Usul Kanunu",
    "GELIRVERGISIKANUNUNUN": "Gelir Vergisi Kanunu",
    "KURUMLARVERGISIKANUNUNUN": "Kurumlar Vergisi Kanunu",
    "KATMADEGERVERGISIKANUNUNUN": "Katma Değer Vergisi Kanunu",
    "DAMGAVERGISIKANUNUNUN": "Damga Vergisi Kanunu",
}


KNOWN_LAW_NAMES = sorted(
    set(CANONICAL_LAW_BY_NO.values())
    | set(LAW_NAME_ALIASES.values())
    | {
        "Türkiye Cumhuriyeti ile Almanya Federal Cumhuriyeti Arasında Gelir Üzerinden Alınan Vergilerde Çifte Vergilendirmeyi ve Vergi Kaçakçılığını Önleme Anlaşması",
        "Kanun Hükmünde Kararname",
    }
)


def collapse_ws(text: str) -> str:
    """Collapse repeated whitespace to a single space."""
    return RE_MULTI_SPACE.sub(" ", text).strip()


def normalize_reference_text(text: str) -> str:
    """Normalize punctuation variants that break legal-reference parsing."""
    if not text:
        return ""
    return RE_INLINE_MADDE_DOT.sub(" ", text)


def extract_ozelge_no(text: str) -> str:
    """Extract `ozelge_no` from the document header if possible."""
    match = RE_OZELGE_NO.search(text)
    if match:
        return collapse_ws(match.group("ozelge_no"))
    fallback = re.search(r"Say\u0131\s*:\s*(\S+)", text)
    return fallback.group(1).strip() if fallback else ""


# A decree, circular, ruling or date number is not a law number ("2007/13033"
# and "89/14391" BKK, "2013/6", "2006-32/32", "24/12/2012", "24.12.2012"): kept
# whole. Any other value is the law number it starts with: "193 sayılı",
# "No:193", "193 Sayılı Kanun" all name law 193 -- before the engine audit
# (2 Oct 2026) they read "193sayl", "No193", "193SaylKanun" here while the
# loop's engine read "193", so a model writing the number that way was charged
# a miss and a wrong row. Mirrors `data_quality_checker.normalization.normalize_law_number`.
RE_NON_LAW_NUMBER = re.compile(r"\d\s*[/-]\s*\d|(?<!\d)\d{1,2}\s*\.\s*\d{1,2}\s*\.\s*\d{2,4}(?!\d)")
RE_DOTTED_DATE = re.compile(r"(?<!\d)\d{1,2}\s*\.\s*\d{1,2}\s*\.\s*\d{2,4}(?!\d)")
EMPTY_LAW_NUMBER_WORDS = {"null", "none", "nan", "yok", "-", "\u2014", "\u2013", "n/a"}


def normalize_kanun_no(value: str) -> str:
    """The law number a `kanun_no` value names ("" when it names none)."""
    if not value:
        return ""
    text = collapse_ws(str(value))
    if text.casefold() in EMPTY_LAW_NUMBER_WORDS:
        return ""
    if RE_NON_LAW_NUMBER.search(text):
        text = RE_DOTTED_DATE.sub(lambda m: re.sub(r"\s*\.\s*", "/", m.group(0)), text)
        compact = re.sub(r"[^0-9/-]", "", text).strip("/-")
        return re.sub(r"(?<!\d)0+(?=\d)", "", compact)
    match = re.search(r"\d+", text.replace(".", ""))
    if not match:
        return ""
    return match.group(0).lstrip("0") or "0"


def canonical_law_name_by_no(kanun_no: str) -> str:
    """Resolve canonical law name by law number."""
    normalized_no = normalize_kanun_no(kanun_no)
    if not normalized_no:
        return ""
    return CANONICAL_LAW_BY_NO.get(normalized_no, "")


def _normalize_turkish_key(text: str) -> str:
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^A-Za-z0-9]+", "", value).upper()
    return value


def normalize_kanun_adi(text: str, kanun_no: str = "") -> str:
    """Normalize case/suffix variants of law names."""
    raw = collapse_ws(text)
    fallback_by_no = canonical_law_name_by_no(kanun_no)

    if not raw:
        return fallback_by_no

    value = re.sub(r"[\u2018\u2019`´]", "'", raw)
    value = re.sub(r"\((?:kdv|ötv|otv)\)", "", value, flags=re.IGNORECASE)
    # "193 sayılı Gelir Vergisi Kanunu" names the same law as "Gelir Vergisi
    # Kanunu"; the prefix only restates kanun_no and otherwise defeats the
    # alias lookup below.
    value = re.sub(r"^\d+\s*say[ıi]l[ıi]\s+", "", value, flags=re.IGNORECASE)
    value = collapse_ws(value.strip(" ,;:.()[]{}\"'«»“”"))
    value = re.sub(
        r"Kanun(?:unun|unda|una|undan|uyla|udaki|la|daki|da|a)(?:'?(?:nun|n[u\u00fc]n|na|nda))?$",
        "Kanunu",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"Kanunu'?(?:nun|n[u\u00fc]n|na|nda)$", "Kanunu", value, flags=re.IGNORECASE)
    value = collapse_ws(value)

    upper_key = _normalize_turkish_key(value)
    if upper_key in LAW_ABBREVIATIONS:
        return LAW_ABBREVIATIONS[upper_key]

    if upper_key in LAW_NAME_ALIASES:
        return LAW_NAME_ALIASES[upper_key]

    if value.lower() in GENERIC_LAW_NAMES:
        return fallback_by_no or "Kanun"

    if fallback_by_no and upper_key in {"KANUN", "KANUNU"}:
        return fallback_by_no

    return value


#: Engine audit (2 Oct 2026): "94'üncü", "geçici 9'uncu" -- an apostrophe and the
#: ordinal after a number; "md. 94", "m.94", "94 md."; "(17/4-r)" with the whole
#: value in brackets; "32 / A". Mirrors `data_quality_checker.normalization`.
RE_APOSTROPHE_ORDINAL = re.compile(
    r"(?<=\d)\s*['\u2019`\u00b4]\s*(?:inci|\u0131nc\u0131|uncu|\u00fcnc\u00fc|nci|nc\u0131|ncu|nc\u00fc)?",
    re.IGNORECASE,
)
RE_ARTICLE_ABBREVIATION_BEFORE = re.compile(r"^(?:md\.?|m\.)\s*(?=\d)", re.IGNORECASE)
RE_ARTICLE_ABBREVIATION_AFTER = re.compile(r"\s+(?:md|mad)\.?$", re.IGNORECASE)
RE_ENCLOSED_ARTICLE = re.compile(r"^\(\s*(?P<inner>[^()]*?/[^()]*?|[^\W\d_]+\s+\d[^()]*?)\s*\)$")


def _normalize_madde_once(value: str) -> str:
    raw = collapse_ws(value)
    # A list mark before the article ("-1/1 inci maddesinde") is not part of it;
    # mirrors data_quality_checker.normalization (1 Oct 2026).
    raw = re.sub(r"^[-–—•·*]+\s*", "", raw)
    raw = re.sub(r"[\u2018\u2019`´]", "'", raw)
    raw = RE_APOSTROPHE_ORDINAL.sub("", raw)
    raw = RE_ARTICLE_ABBREVIATION_BEFORE.sub("", raw)
    raw = re.sub(r"^madd\w*\s+", "", raw, flags=re.IGNORECASE)
    # "229 ve müteakip maddelerinde" cites article 229 onwards; the article
    # is 229 and the trailing phrase is prose.
    raw = re.sub(r"\s+ve\s+m[üu]teakip(\s+madd\w*)?$", "", raw, flags=re.IGNORECASE)
    # The article word must go first. "37 nci maddesinde" ends in
    # "maddesinde", so an ordinal strip anchored to the end could never fire
    # and what survived was "37 nci" -- an article that matches nothing.
    raw = re.sub(r"\s*madd\w*$", "", raw, flags=re.IGNORECASE)
    raw = re.sub(rf"\s+{ORDINAL_SUFFIX_RE}$", "", raw, flags=re.IGNORECASE)
    raw = RE_ARTICLE_ABBREVIATION_AFTER.sub("", raw).strip(" .")
    enclosed = RE_ENCLOSED_ARTICLE.match(raw)
    if enclosed:
        raw = enclosed.group("inner").strip()
    raw = re.sub(r"\s*/\s*", "/", raw)
    raw = strip_article_subpart(raw)
    raw = raw.strip(" .()")
    raw = re.sub(r"\s+", " ", raw)
    if re.match(r"(?i)^ek\s+madde\s+\d+$", raw):
        number = raw.split()[-1]
        raw = f"Ek {number}"
    return collapse_ws(raw)


def normalize_madde(value: str) -> str:
    """Normalize article text to canonical `madde` style.

    A fixed point: normalizing the output again changes nothing. A value that
    names something but no article this can read ("(17)", a paragraph number)
    is kept as written, folded, so it matches no article and is never read as
    a law-level row (engine audit, 2 Oct 2026; the loop's engine does the same).
    """
    if not value:
        return ""
    text = _normalize_madde_once(value)
    for _ in range(4):
        again = _normalize_madde_once(text) if text else text
        if again == text:
            break
        text = again
    if not text:
        written = collapse_ws(str(value))
        if re.search(r"[^\W_]", written):
            return written.casefold()
    return text


def normalize_reference(reference: dict) -> dict:
    """Return a safe, schema-compatible reference object.

    One normalizer for the project (user decision, 2 Oct 2026): the fields are
    `data_quality_checker.normalization`'s, with the number as written, exactly
    as the matcher (`data_quality_checker.reference_matching`) reads them.
    """
    from data_quality_checker.reference_matching import canonical_reference

    canonical = canonical_reference(reference)
    return {
        "type": REFERENCE_TYPE,
        "kanun_no": canonical["kanun_no"],
        "kanun_ad": canonical["kanun_ad"],
        "madde": canonical["madde"],
        "source_text": collapse_ws(str(reference.get("source_text", ""))),
    }


def deduplicate_references(references: Iterable[dict]) -> list[dict]:
    """Deduplicate references while preserving insertion order."""
    unique = OrderedDict()
    for raw_ref in references:
        ref = normalize_reference(raw_ref)
        key = tuple(ref[field] for field in REFERENCE_FIELDS)
        if key not in unique:
            unique[key] = ref
    return list(unique.values())


def reference_tuple_key(reference: dict) -> tuple[str, str, str, str]:
    """Return a document-level identity key that ignores `source_text`."""
    ref = normalize_reference(reference)
    return (
        ref["type"],
        ref["kanun_no"],
        ref["kanun_ad"],
        ref["madde"],
    )


def law_family_key(reference: dict) -> str:
    """Return a best-effort grouping key for rows that point to the same law."""
    ref = normalize_reference(reference)
    if ref["kanun_no"]:
        return f"no:{ref['kanun_no']}"
    if ref["kanun_ad"] and ref["kanun_ad"].lower() not in GENERIC_LAW_NAMES:
        return f"name:{_normalize_turkish_key(ref['kanun_ad'])}"
    return ""


def is_specific_reference(reference: dict) -> bool:
    """Return whether a reference names an article, not only a law."""
    ref = normalize_reference(reference)
    return bool(ref["madde"])


def compact_law_references(references: Iterable[dict]) -> tuple[list[dict], dict[str, int]]:
    """Apply document-level generic suppression and tuple-level deduplication."""
    normalized_rows: list[dict] = []
    for order, raw_ref in enumerate(references):
        merged = dict(raw_ref)
        merged.update(normalize_reference(raw_ref))
        merged["_assist_order"] = int(raw_ref.get("_assist_order", order))
        merged["_assist_origin"] = str(raw_ref.get("_assist_origin", "")).strip() or "new"
        normalized_rows.append(merged)

    groups: OrderedDict[str, list[dict]] = OrderedDict()
    ungrouped: list[dict] = []
    for row in normalized_rows:
        family_key = law_family_key(row)
        if family_key:
            groups.setdefault(family_key, []).append(row)
        else:
            ungrouped.append(row)

    summary = {
        "dropped_generic_due_to_specific": 0,
        "collapsed_generic_only_duplicates": 0,
        "collapsed_same_reference_duplicates": 0,
        "removed_existing_rows": 0,
        "added_new_rows": 0,
    }

    def survivor_rank(row: dict) -> tuple[int, int, int, int, int, int]:
        status = str(row.get("status", "")).strip().lower()
        status_score = 2 if status == "approved" else 1 if status else 0
        origin_score = 1 if row.get("_assist_origin") == "existing" else 0
        identity_score = 0
        if row.get("kanun_no") and row.get("kanun_ad"):
            identity_score = 3
        elif row.get("kanun_no"):
            identity_score = 2
        elif row.get("kanun_ad"):
            identity_score = 1
        detail_score = 1 if row.get("madde") else 0
        source_score = 1 if row.get("source_text") else 0
        order_score = -int(row.get("_assist_order", 0))
        return (status_score, origin_score, identity_score, detail_score, source_score, order_score)

    survivors: list[dict] = []

    def mark_removed(rows_to_remove: list[dict], field: str) -> None:
        summary[field] += len(rows_to_remove)
        summary["removed_existing_rows"] += sum(1 for row in rows_to_remove if row.get("_assist_origin") == "existing")

    for grouped_rows in groups.values():
        specifics = [row for row in grouped_rows if is_specific_reference(row)]
        generics = [row for row in grouped_rows if not is_specific_reference(row)]
        if specifics:
            survivors.extend(grouped_rows if not generics else specifics)
            if generics:
                mark_removed(generics, "dropped_generic_due_to_specific")
            continue

        if len(generics) <= 1:
            survivors.extend(generics)
            continue

        best_generic = max(generics, key=survivor_rank)
        survivors.append(best_generic)
        mark_removed([row for row in generics if row is not best_generic], "collapsed_generic_only_duplicates")

    survivors.extend(ungrouped)

    unique_rows: OrderedDict[tuple[str, str, str, str], dict] = OrderedDict()
    for row in survivors:
        key = reference_tuple_key(row)
        existing = unique_rows.get(key)
        if existing is None:
            unique_rows[key] = row
            continue

        best = max((existing, row), key=survivor_rank)
        removed = row if best is existing else existing
        if removed.get("_assist_origin") == "existing":
            summary["removed_existing_rows"] += 1
        summary["collapsed_same_reference_duplicates"] += 1
        unique_rows[key] = best

    final_rows: list[dict] = []
    for row in unique_rows.values():
        clean = {
            field: row[field]
            for field in REFERENCE_FIELDS
            if field in row
        }
        if "status" in row:
            clean["status"] = row["status"]
        final_rows.append(clean)
        if row.get("_assist_origin") == "new":
            summary["added_new_rows"] += 1

    return final_rows, summary


def compact_annotation_assist_references(references: Iterable[dict]) -> tuple[list[dict], dict[str, int]]:
    """Backward-compatible alias for annotator assistance."""
    return compact_law_references(references)


def snippet(text: str, start: int, end: int, min_len: int = 30, max_len: int = 180) -> str:
    """Build a compact source snippet around a span."""
    s = max(0, start)
    e = min(len(text), end)
    if e <= s:
        return ""
    if e - s < min_len:
        needed = min_len - (e - s)
        left = needed // 2
        right = needed - left
        s = max(0, s - left)
        e = min(len(text), e + right)
    if e - s > max_len:
        e = s + max_len
    return collapse_ws(text[s:e])

