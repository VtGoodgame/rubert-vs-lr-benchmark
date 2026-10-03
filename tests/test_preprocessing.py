"""Тесты очистки текста перед токенизацией.

Зафиксировано фактическое поведение clean_text: регистр и пунктуация
не трогаются, удаляются только ссылки, кавычки, звёздочки и лишние пробелы.
"""

from model.preprocessing.clean import clean_text


def test_plain_text_unchanged():
    assert clean_text("Hello World") == "Hello World"


def test_case_is_preserved():
    assert clean_text("SPAM") == "SPAM"


def test_punctuation_is_preserved():
    """Точка, запятая и восклицательный знак остаются: они несут смысл."""
    assert clean_text("SPAM!!!") == "SPAM!!!"
    assert clean_text("цена 100 руб.") == "цена 100 руб."


def test_http_url_replaced():
    result = clean_text("заходи http://example.com сюда")
    assert "[URL]" in result
    assert "example.com" not in result


def test_https_url_replaced():
    assert "[URL]" in clean_text("https://example.com/page")


def test_www_url_replaced():
    result = clean_text("www.example.com")
    assert "[URL]" in result
    assert "example.com" not in result


def test_multiple_urls_all_replaced():
    result = clean_text("http://a.com и www.b.com")
    assert result.count("[URL]") == 2


def test_quotes_removed():
    assert '"' not in clean_text('say "hello"')
    assert clean_text('say "hello"') == "say hello"


def test_stars_removed():
    assert "*" not in clean_text("hello**world")


def test_quotes_and_stars_together():
    result = clean_text('"привет"*')
    assert result == "привет"


def test_whitespace_collapsed():
    assert clean_text("a   b") == "a b"
    assert clean_text("a\t\tb") == "a b"
    assert clean_text("a\n\nb") == "a b"


def test_stripped_edges():
    assert clean_text("   много пробелов   ") == "много пробелов"


def test_newlines_become_spaces():
    assert "\n" not in clean_text("первая\nвторая")


def test_empty_string():
    assert clean_text("") == ""


def test_only_whitespace():
    assert clean_text("   \t  ") == ""


def test_cyrillic_preserved():
    assert clean_text("привет мир") == "привет мир"


def test_url_inside_long_text():
    result = clean_text("переходи по http://spam.com и получи приз")
    assert result == "переходи по [URL] и получи приз"
