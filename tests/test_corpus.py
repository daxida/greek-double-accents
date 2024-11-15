from itertools import zip_longest
from pathlib import Path

from greek_double_accents.main import analyze_text


def make_test(corpus_fpath: str) -> None:
    # A fixture needs to follow the structure:
    #
    # tests/fixtures/
    # ├── corpus_clean (cleaned version)
    # └── corpus_wrong (altered version)
    #
    # So we just need to pass the path to "corpus"
    fixtures_path = Path("tests/fixtures")
    clean_corpus_path = fixtures_path / f"{corpus_fpath}_clean.txt"
    wrong_corpus_path = fixtures_path / f"{corpus_fpath}_wrong.txt"
    clean_text = clean_corpus_path.open("r", encoding="utf-8").read()
    wrong_text = wrong_corpus_path.open("r", encoding="utf-8").read()

    wrong_text_fixed, _ = analyze_text(wrong_text)
    words_it = zip_longest(clean_text.split(), wrong_text_fixed.split())

    fails = 0
    total = 0
    for expected, received in words_it:
        if expected != received:
            fails += 1
        total += 1

    success_ratio = (total - fails) / total
    assert success_ratio > 0.95


def test_golden_corpus() -> None:
    make_test("hnc_golden_corpus")


def test_educational() -> None:
    make_test("corpus_educational")
