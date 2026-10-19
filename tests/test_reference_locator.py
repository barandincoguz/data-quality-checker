"""Tests for locating a known reference's span in raw document text.

Each text fixture below was taken from an actual M01 document; none is
invented. `locate_reference_span` is given a reference whose identity is
already known (harvested from the model's output, absent from the
annotator's) and must return a verbatim substring of the document that
supports it, or `None` when it should not guess.
"""

from data_quality_checker.reference_locator import locate_reference_span


def test_locates_an_ordinary_article_mention() -> None:
    text = "213 sayılı Vergi Usul Kanununun 413 üncü maddesinde belirtilen usul"
    span = locate_reference_span(
        text,
        {
            "kanun_no": "213",
            "kanun_ad": "Vergi Usul Kanunu",
            "madde": "413",
            "fikra": "",
            "bent": "",
        },
    )
    assert span is not None
    assert span in text
    assert "413" in span


def test_locates_each_number_of_a_shared_ordinal_series() -> None:
    """ "2, 3 ve 4 üncü maddeleri": only the last number carries the suffix,
    yet every number in the series names an article. Taken from d000014."""
    text = "Mezkûr Kararın 2, 3 ve 4 üncü maddeleri gereğince, internet ortamında"
    for madde in ("2", "3", "4"):
        span = locate_reference_span(
            text,
            {"kanun_no": "476", "kanun_ad": "", "madde": madde, "fikra": "", "bent": ""},
        )
        assert span is not None and span in text


def test_a_law_level_reference_is_located_by_its_law_mention() -> None:
    text = "5957 sayılı Sebze ve Meyveler ile Yeterli Arz ve Talep Derinliği bulunan"
    span = locate_reference_span(
        text,
        {
            "kanun_no": "5957",
            "kanun_ad": "Sebze ve Meyveler",
            "madde": "",
            "fikra": "",
            "bent": "",
        },
    )
    assert span is not None and span in text


def test_refuses_a_serial_number_masquerading_as_an_article() -> None:
    """d000052: the model read communiqué serial numbers as VUK articles. The
    locator must refuse rather than hand back the law mention, because an
    unlocatable article is evidence the reference itself is wrong."""
    text = (
        "uyulacak usul ve esaslar sırasıyla 350, 362, 381 ve 396 Sıra No.lu "
        "Vergi Usul Kanunu Genel Tebliğleri ile duyurulmuştur"
    )
    span = locate_reference_span(
        text,
        {
            "kanun_no": "213",
            "kanun_ad": "Vergi Usul Kanunu",
            "madde": "350",
            "fikra": "",
            "bent": "",
        },
    )
    assert span is None


def test_every_returned_span_is_a_verbatim_substring() -> None:
    """The only property context_windows.py actually depends on."""
    text = "3065 sayılı Katma Değer Vergisi Kanununun 17/4-k maddesi uyarınca"
    for madde in ("17/4-k", "17"):
        span = locate_reference_span(
            text,
            {
                "kanun_no": "3065",
                "kanun_ad": "Katma Değer Vergisi Kanunu",
                "madde": madde,
                "fikra": "",
                "bent": "",
            },
        )
        if span is not None:
            assert span in text


def _ref(kanun_no: str, madde: str) -> dict[str, str]:
    return {
        "kanun_no": kanun_no,
        "kanun_ad": "",
        "madde": madde,
        "fikra": "",
        "bent": "",
        "source_text": "",
    }


def test_a_series_may_carry_prefixed_and_subdivided_articles() -> None:
    """R01 (27 Sep 2026): the series pattern stopped at "mükerrer" and "150/e"."""
    vuk = "Diğer taraftan 213 sayılı Vergi Usul Kanununun 148, 149 ve mükerrer 257 nci maddelerinin verdiği yetki"
    for madde in ("148", "149", "mükerrer 257"):
        span = locate_reference_span(vuk, _ref("213", madde))
        assert span is not None and span in vuk, madde
    iik = "2004 sayılı İcra ve İflas Kanununun hükmüne göre 106 ve 150/e maddelerindeki süreler işlemez."
    for madde in ("106", "150"):
        assert locate_reference_span(iik, _ref("2004", madde)) is not None, madde


def test_an_article_range_names_every_article_inside_it() -> None:
    text = "Borçlar Kanununun 638-642 inci madde hükümleri kapsamında iştirak halinde mülkiyet"
    for madde in ("638", "640", "642"):
        assert locate_reference_span(text, _ref("6098", madde)) is not None, madde
    for madde in ("637", "643"):
        assert locate_reference_span(text, _ref("6098", madde)) is None, madde


def test_a_paragraph_range_is_not_an_article_range() -> None:
    text = "193 sayılı Kanunun 51/1-8 maddesinde sayılan kazançlar"
    assert locate_reference_span(text, _ref("193", "5")) is None
    assert locate_reference_span(text, _ref("193", "51")) is not None


def test_an_article_followed_by_the_ones_after_it_is_located() -> None:
    for text in (
        "defterlerin tasdiki Kanunun 220 ve müteakip maddelerinde düzenlenmiştir.",
        "Kanunun 229 uncu ve takip eden maddelerinde sayılmayan belgeler",
    ):
        madde = "220" if "220" in text else "229"
        assert locate_reference_span(text, _ref("213", madde)) is not None, text


def test_a_series_whose_every_article_carries_its_own_ordinal() -> None:
    text = "6758 sayılı Kanunun 19 uncu ve 20 nci maddeleri ile 5411 sayılı Kanunun"
    for madde in ("19", "20"):
        assert locate_reference_span(text, _ref("6758", madde)) is not None, madde
    assert locate_reference_span(text, _ref("6758", "21")) is None


def test_a_number_in_the_paragraph_chain_is_not_an_article() -> None:
    """R: "23 üncü maddesinin 1 inci fıkrasının (a) bendi" cites article 23, never article 1."""
    text = "5520 sayılı Kurumlar Vergisi Kanununun 23 üncü maddesinin 1 inci fıkrasının (a) bendi uyarınca"
    assert locate_reference_span(text, _ref("5520", "1")) is None
    assert locate_reference_span(text, _ref("5520", "23")) is not None


def test_a_bare_subparagraph_number_is_not_an_article() -> None:
    text = "Bu hüküm, maddenin (6) numaralı bendine göre uygulanır."
    assert locate_reference_span(text, _ref("5520", "6")) is None


def test_an_article_written_as_an_ordinal_word() -> None:
    text = "5520 sayılı Kurumlar Vergisi Kanununun birinci maddesinde sayılan kurumlar"
    span = locate_reference_span(text, _ref("5520", "1"))
    assert span is not None and "birinci maddesinde" in span


def test_the_law_outranks_the_paragraph_chain() -> None:
    """A mention of the right article of another law must never beat the right law's
    own mention because its paragraph chain happens to fit."""
    text = (
        "193 sayılı Gelir Vergisi Kanununun 94 üncü maddesinin birinci fıkrasında hüküm vardır. "
        "5520 sayılı Kurumlar Vergisi Kanununda da düzenleme olup, bu Kanunun 94 üncü maddesinin "
        "ikinci fıkrasında da benzeri yer alır."
    )
    reference = {**_ref("193", "94"), "fikra": "2"}
    span = locate_reference_span(text, reference)
    assert span is not None and span.startswith("193 sayılı")


def test_spelling_of_the_article_never_moves_the_placement() -> None:
    text = "3218 sayılı Serbest Bölgeler Kanununun geçici 3 üncü maddesinin 2 nci fıkrasında"
    places = {
        locate_reference_span(text, {**_ref("3218", spelling), "fikra": "2"})
        for spelling in ("geçici 3", "Geçici 3", "GEÇİCİ 3", "geçiçi 3", "Geçici Madde 3")
    }
    assert len(places) == 1 and None not in places


def test_an_article_named_on_its_own_beats_a_range_that_contains_it() -> None:
    text = (
        "6098 sayılı Borçlar Kanununun 620-645 inci maddelerinde adi ortaklık düzenlenmiştir. "
        "Anılan Kanunun 622 nci maddesinde ise sermaye payı açıklanmıştır."
    )
    span = locate_reference_span(text, _ref("6098", "622"))
    assert span is not None and "622 nci" in span


def test_a_lowercase_l_read_as_one_by_the_text_extraction() -> None:
    """ "Damga Vergisi Kanununun l inci maddesinde" (R01 d000216, GT750 doc_119)."""
    text = "488 sayılı Damga Vergisi Kanununun l inci maddesinde, bu Kanuna ekli (1) sayılı tabloda"
    span = locate_reference_span(text, _ref("488", "1"))
    assert span is not None and "l inci maddesinde" in span


def test_an_article_without_its_qualifier_is_a_different_article() -> None:
    """VUK 257 and mükerrer 257 are two articles; placing one on the other would
    hide the annotator's error (R01 d000096, d000101, d000244)."""
    text = "Anılan 213 sayılı Kanunun mükerrer 257 nci maddesinin birinci fıkrası"
    assert locate_reference_span(text, _ref("213", "257")) is None
    assert locate_reference_span(text, _ref("213", "mükerrer 257")) is not None


# --- 28 Sep 2026: law recognition, article ranges, law-level references -----
# Fixtures cut from GT750 documents (named per test) and R01 d000076.

from data_quality_checker.reference_locator import place_reference  # noqa: E402


def _full(
    kanun_no: str, madde: str, fikra: str = "", bent: str = "", kanun_ad: str = ""
) -> dict[str, str]:
    return {
        "kanun_no": kanun_no,
        "kanun_ad": kanun_ad,
        "madde": madde,
        "fikra": fikra,
        "bent": bent,
        "source_text": "",
    }


def test_a_law_title_may_carry_its_acronym_in_parentheses() -> None:
    """GT750 doc_166: "Katma Değer Vergisi (KDV) Kanununun" is the law written before
    the article, not an unknown one."""
    text = (
        "Öte yandan 193 sayılı Gelir Vergisi Kanununun 17 nci maddesinde istisna düzenlenmiştir. "
        "Bu uygulamanın Katma Değer Vergisi (KDV) Kanununun 17/2-b ve 30/a maddesi kapsamında değerlendirilmesi"
    )
    placement = place_reference(text, _full("3065", "17", "2", "b"))
    assert placement is not None and placement.law == 0
    assert text[placement.start : placement.end].startswith("Katma Değer Vergisi (KDV) Kanununun")


def test_the_law_an_article_was_added_to_owns_it() -> None:
    """GT750 doc_175: "Katma Değer Vergisi Kanununa eklenen Geçici 28 inci maddesi" is
    KDV's article. The dative law receives the article; only "Kanunla"/"Kanun ile"
    names the amending law."""
    text = (
        "5904 sayılı Kanunun 13/b maddesi ile madde metninden çıkartılmış, 5904 sayılı yasanın "
        "16 ncı maddesi ile de Katma Değer Vergisi Kanununa eklenen Geçici 28 inci maddesinde: "
        '"Bu maddenin yürürlüğe girdiği tarihte'
    )
    added = place_reference(text, _full("3065", "geçici 28"))
    assert added is not None and added.law in (0, 1)
    amending = place_reference(text, _full("5904", "16"))
    assert amending is not None and amending.law == 0
    assert text[amending.start : amending.end].startswith("5904 sayılı yasanın 16")


def test_an_article_added_by_another_laws_article_belongs_to_the_receiving_law() -> None:
    """GT750 doc_339: in "5520 sayılı KVK'ya 5838 sayılı Kanunun 9 uncu maddesiyle
    eklenen 32/A maddesi", 32/A is 5520's article and 9 is 5838's."""
    text = (
        "hususunda Başkanlığımız görüşü talep edilmektedir. 5520 sayılı Kurumlar Vergisi Kanununa "
        '5838 sayılı Kanunun 9 uncu maddesiyle eklenen 32/A maddesinde; "(1) Finans ve sigortacılık'
    )
    added = place_reference(text, _full("5520", "32/A"))
    assert added is not None and added.law in (0, 1)
    amending = place_reference(text, _full("5838", "9"))
    assert amending is not None and amending.law == 0


def test_words_before_a_short_law_name_do_not_hide_it() -> None:
    """GT750 doc_561 ("Ayrıca, KDV Kanununun") and doc_225 ("… A.Ş’ye KDV Kanununun"):
    a sentence opener or a company name glued before the title is not part of it."""
    for text, madde, bent in (
        (
            "%8 oranında KDV uygulanması öngörülmüştür. Ayrıca, KDV Kanununun 13 üncü maddesinin (l) bendiyle; Sağlık",
            "13",
            "l",
        ),
        (
            "Avrasya Tüneli İşletme İnşaat ve Yatırım A.Ş’ye KDV Kanununun geçici 29 uncu maddesi uyarınca KDV",
            "geçici 29",
            "",
        ),
    ):
        placement = place_reference(text, _full("3065", madde, bent=bent))
        assert placement is not None and placement.law == 0, text


def test_a_numeric_article_range_is_a_range_not_an_article_with_a_subpart() -> None:
    """GT750 doc_129: "520-541 inci maddelerinde" names 520 to 541; the article named on
    its own right after it is where 520 is cited."""
    text = (
        "Adi ortaklıklar Borçlar Kanununun 520-541 inci maddelerinde düzenlenmiştir. "
        "Anılan Kanunun 520 inci maddesinde adi ortaklık, iki veya daha fazla kişinin"
    )
    span = locate_reference_span(text, _full("", "520", kanun_ad="Borçlar Kanunu"))
    assert span is not None and span.startswith("Anılan Kanunun 520")
    inside = locate_reference_span(text, _full("", "530", kanun_ad="Borçlar Kanunu"))
    assert inside is not None and "520-541" in inside


def test_a_law_level_reference_prefers_the_law_written_in_full() -> None:
    """GT750 doc_171: "2330 sayılı Kanuna" first, the title later -- the citation that
    names the law is the better evidence."""
    text = (
        "Milli Savunma Bakanlığı tarafından 16/5/2019 tarihinde 2330 sayılı Kanuna istinaden tarafınıza "
        "nakdi tazminat ödendiği belirtilmiştir. 2330 sayılı Nakdi Tazminat ve Aylık Bağlanması Hakkında "
        "Kanuna göre ödenen tazminatlar"
    )
    span = locate_reference_span(text, _full("2330", ""))
    assert span is not None and "Nakdi Tazminat" in span


def test_a_hyphenated_law_title_is_read_whole() -> None:
    text = "Bu durumda ödenen primler Bağ-Kur Kanunu kapsamında değerlendirilir."
    span = locate_reference_span(text, _full("1479", "", kanun_ad="Bağ-Kur Kanunu"))
    assert span == "Bağ-Kur Kanunu"


def test_a_decree_numbered_by_year_is_found_by_its_number() -> None:
    """R01 d000076: "2007/13033 sayılı Bakanlar Kurulu Kararı" (KDV rates)."""
    text = (
        "Bu kapsamda KDV oranları, 2007/13033 sayılı Bakanlar Kurulu Kararı (BKK) eki (I) sayılı "
        "listede yer alan teslim ve hizmetler için"
    )
    span = locate_reference_span(text, _full("2007/13033", ""))
    assert span is not None and span.startswith("2007/13033 sayılı")


def test_a_qualifier_carries_over_the_series_it_opens() -> None:
    """GT750 doc_148: "Geçici 19 ve 20 inci maddelerinde" cites geçici 19 and geçici 20."""
    text = (
        "Diğer taraftan 4447 sayılı İşsizlik Sigortası Kanununun Geçici 19 ve 20 inci maddelerinde; "
        "fondan bu maddeler kapsamında karşılanan"
    )
    for madde in ("geçici 19", "geçici 20"):
        span = locate_reference_span(text, _full("4447", madde))
        assert span is not None and "Geçici 19 ve 20" in span, madde


def test_a_lowercase_l_read_as_one_before_a_subdivision() -> None:
    """GT750 doc_390: "Katma Değer Vergisi (KDV) Kanununun l/3-f maddesinde"."""
    text = "Başkanlığımızdan bilgi talep edilmektedir. Katma Değer Vergisi (KDV) Kanununun l/3-f maddesinde; Gelir"
    placement = place_reference(text, _full("3065", "1", "3", "f"))
    assert placement is not None and "l/3-f" in text[placement.start : placement.end]


def test_a_bent_written_with_its_closing_parenthesis_is_the_same_bent() -> None:
    """GT750 doc_106: "c)" is how a bent is printed; it must place exactly like "c"."""
    text = (
        "4760 sayılı Özel Tüketim Vergisi Kanununun 7 nci maddesinde düzenlenmiştir. "
        "Özel Tüketim Vergisi Kanununun (7/2-c) maddesi uyarınca"
    )
    plain = place_reference(text, _full("4760", "7", "2", "c"))
    printed = place_reference(text, _full("4760", "7", "2", "c)"))
    assert plain is not None and printed is not None
    assert (printed.start, printed.end) == (plain.start, plain.end)
