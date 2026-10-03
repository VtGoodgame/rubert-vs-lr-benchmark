# tests/test_preprocessing.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.preprocessing.clean import clean_text


def test_clean_text_basic():
    assert clean_text("Hello World") == "Hello World"


def test_clean_text_punctuation_preserved():
    # Текущая реализация не удаляет пунктуацию, кроме " и *
    assert clean_text("SPAM!!!") == "SPAM!!!"


def test_clean_text_urls():
    s = clean_text("http://example.com test")
    assert "[URL]" in s
    assert "example.com" not in s


def test_clean_text_whitespace():
    s = clean_text("a   b\t\nc")
    assert s == "a b c"


def test_clean_text_quotes_and_stars_removed():
    s = clean_text('say "hello"*')
    assert '"' not in s
    assert "*" not in s
    assert "hello" in s