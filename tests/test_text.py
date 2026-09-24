from data_quality_checker.text import html_to_text


def test_inline_tags_do_not_split_a_word():
    assert html_to_text("<p>234 <span>ü</span>ncü maddesi</p>") == "234 üncü maddesi"


def test_block_tags_are_boundaries():
    assert html_to_text("<p>bir</p><p>iki</p>") == "bir iki"


def test_the_html_head_never_reaches_the_text():
    html = "<html><head><title>Untitled document</title></head><body><p>İlgide kayıtlı</p></body></html>"
    assert html_to_text(html) == "İlgide kayıtlı"
