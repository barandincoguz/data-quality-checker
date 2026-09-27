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
