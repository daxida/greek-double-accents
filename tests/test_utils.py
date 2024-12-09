import pytest

from greek_double_accents.utils import add_accent


@pytest.mark.parametrize(
    "word, expected",
    [
        ("λόγος", "λόγός"),
        ("λόγός", "λόγός"),
        ("ανθρωπος", "ανθρωπός"),
        ("", ""),
    ],
)
def test_add_accent(word: str, expected: str) -> None:
    assert add_accent(word) == expected
