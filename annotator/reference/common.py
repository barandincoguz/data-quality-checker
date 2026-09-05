"""Shared helpers for reference-style legal annotation outputs."""

from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict
from typing import Iterable

REFERENCE_TYPE = "kanun_madde_referansi"
REFERENCE_FIELDS = ("type", "kanun_no", "kanun_ad", "madde", "fikra", "bent", "source_text")
CORE_FIELDS = ("kanun_no", "kanun_ad", "madde", "fikra", "bent")

ORDINAL_MAP = {
    "birinci": "1",
    "ikinci": "2",
    "ucuncu": "3",
    "dorduncu": "4",
    "besinci": "5",
    "altinci": "6",
    "yedinci": "7",
    "sekizinci": "8",
    "dokuzuncu": "9",
    "onuncu": "10",
}

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

ORDINAL_WORD_PATTERN = r"birinci|ikinci|ucuncu|dorduncu|besinci|altinci|yedinci|sekizinci|dokuzuncu|onuncu"

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


def normalize_kanun_no(value: str) -> str:
    """Normalize law number while keeping `/` and `-` separators."""
    if not value:
        return ""
    cleaned = RE_NON_NO_CHARS.sub("", collapse_ws(str(value)))
    cleaned = cleaned.strip("/-")
    cleaned = re.sub(r"\b0+(\d)", r"\1", cleaned)
    return collapse_ws(cleaned)


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
    value = collapse_ws(value.strip(" ,;:.()[]{}"))
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


def normalize_identifier(value: str) -> str:
    """Normalize identifier values used in `fikra` and `bent` fields."""
    if not value:
        return ""
    raw = collapse_ws(value).strip("()[]{}")
    lowered = _normalize_turkish_key(raw).lower()
    return ORDINAL_MAP.get(lowered, raw)


RE_IDENTIFIER_SERIES_ITEM = re.compile(
    rf"\((?P<par>[A-Za-z0-9]+)\)|(?P<num>\d+)(?:\s*{ORDINAL_SUFFIX_RE})?|(?P<ord>{ORDINAL_WORD_PATTERN})",
    re.IGNORECASE | re.UNICODE,
)


def extract_identifier_series(text: str) -> list[str]:
    """Extract ordered identifier lists such as `(1), (2) ve (4)`."""
    if not text:
        return []
    values: list[str] = []
    seen: set[str] = set()
    for match in RE_IDENTIFIER_SERIES_ITEM.finditer(text):
        value = match.group("par") or match.group("num") or match.group("ord") or ""
        normalized = normalize_identifier(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return values


def normalize_madde(value: str) -> str:
    """Normalize article text to canonical `madde` style."""
    if not value:
        return ""
    raw = collapse_ws(value)
    raw = re.sub(r"[\u2018\u2019`´]", "'", raw)
    raw = re.sub(r"^madd\w*\s+", "", raw, flags=re.IGNORECASE)
    raw = re.sub(rf"\s+{ORDINAL_SUFFIX_RE}$", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*madd\w*$", "", raw, flags=re.IGNORECASE)
    raw = raw.strip(" .()")
    raw = re.sub(r"\s+", " ", raw)
    if re.match(r"(?i)^ek\s+madde\s+\d+$", raw):
        number = raw.split()[-1]
        raw = f"Ek {number}"
    return collapse_ws(raw)


def parse_madde_token(token: str) -> tuple[str, str, str]:
    """Parse article token into `(madde, fikra, bent)` components."""
    cleaned = normalize_madde(token)
    if not cleaned:
        return "", "", ""

    head = cleaned
    fikra = ""
    bent = ""

    if "/" in cleaned:
        left, right = cleaned.split("/", 1)
        head = left.strip()
        right = right.strip()
        if "-" in right:
            first, second = right.split("-", 1)
            if first.isdigit():
                fikra = first
                bent = normalize_identifier(second)
            else:
                bent = normalize_identifier(right)
        elif right.isdigit():
            fikra = right
        else:
            # In this project, `13/a maddesi` is an article surface form, not a bent.
            if re.fullmatch(r"[A-Za-zÇĞİÖŞÜçğıöşü]", right):
                return cleaned, "", ""
            bent = normalize_identifier(right)

    return head, fikra, bent


def normalize_reference(reference: dict) -> dict:
    """Return a safe, schema-compatible reference object."""
    kanun_no = normalize_kanun_no(str(reference.get("kanun_no", "")))
    kanun_ad = normalize_kanun_adi(str(reference.get("kanun_ad", "")), kanun_no=kanun_no)

    ref = {
        "type": REFERENCE_TYPE,
        "kanun_no": kanun_no,
        "kanun_ad": kanun_ad,
        "madde": normalize_madde(str(reference.get("madde", ""))),
        "fikra": normalize_identifier(str(reference.get("fikra", ""))),
        "bent": normalize_identifier(str(reference.get("bent", ""))),
        "source_text": collapse_ws(str(reference.get("source_text", ""))),
    }
    return ref


def deduplicate_references(references: Iterable[dict]) -> list[dict]:
    """Deduplicate references while preserving insertion order."""
    unique = OrderedDict()
    for raw_ref in references:
        ref = normalize_reference(raw_ref)
        key = tuple(ref[field] for field in REFERENCE_FIELDS)
        if key not in unique:
            unique[key] = ref
    return list(unique.values())


def reference_tuple_key(reference: dict) -> tuple[str, str, str, str, str, str]:
    """Return a document-level identity key that ignores `source_text`."""
    ref = normalize_reference(reference)
    return (
        ref["type"],
        ref["kanun_no"],
        ref["kanun_ad"],
        ref["madde"],
        ref["fikra"],
        ref["bent"],
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
    """Return whether a reference carries article / paragraph / subparagraph detail."""
    ref = normalize_reference(reference)
    return bool(ref["madde"] or ref["fikra"] or ref["bent"])


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
        detail_score = sum(1 for field in ("madde", "fikra", "bent") if row.get(field))
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

    unique_rows: OrderedDict[tuple[str, str, str, str, str, str], dict] = OrderedDict()
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

