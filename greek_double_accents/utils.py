import re

from grac import add_acute_at
from grac import syllabify_el as syllabify

VOWEL_ACCENTED = re.compile(r"[έόίύάήώ]")
PUNCT = re.compile(r"[,.!?;:\n«»\"'·…]")
LINE_RE = re.compile(r"\.*[^.!?;:…»]+(?:[.!?;:…»\n]+,?)?")
# \b\w*[έόίύάήώΈΌΊΎΆΉΏ]\w*[έόίύάήώΈΌΊΎΆΉΏ]\w*\b
REMOVE_TRANS = str.maketrans("έόίύάήώ", "εοιυαηω")


def add_accent(word: str) -> str:
    """Add acute to the last syllable."""
    return add_acute_at(word, 1)


def remove_accent(word: str) -> str:
    """Remove acute from the last syllable."""
    s = syllabify(word)
    return "".join(s[:-1] + [s[-1].translate(REMOVE_TRANS)])


def has_accent(syllable: str) -> bool:
    """Check if the syllable has a lowercase acute accent."""
    return VOWEL_ACCENTED.search(syllable) is not None


def has_correct_double_accent(word: str) -> bool:
    if "-" in word:
        return False
    s = syllabify(word)
    return len(s) > 2 and has_accent(s[-3]) and has_accent(s[-1])


def is_simple_proparoxytone(word: str) -> bool:
    s = syllabify(word)
    return len(s) > 2 and has_accent(s[-3]) and not has_accent(s[-1])


def deep_flatten(nested):  # noqa
    """Flatten an arbitrarily deep nesting of lists."""
    for item in nested:
        if isinstance(item, list):
            yield from deep_flatten(item)
        else:
            yield item


def split_punctuation(word: str) -> tuple[str, str | None]:
    """Splits a word into its base form and any trailing punctuation."""
    if mtch := PUNCT.search(word):
        return word[: mtch.start()], word[mtch.start() :]
    return word, None


def split_text(text: str) -> list[list[list[str]]]:
    return [
        [line.split(" ") for line in split_paragraph(par)] for par in text.splitlines(keepends=True)
    ]


def join_words(words: list[list[list[str]]]) -> str:
    """The reverse of split_text."""
    new_pars = []
    for par in words:
        new_line = []
        for line in par:
            new_line.append(" ".join(line))
        new_pars.append("".join(new_line))
    return "".join(new_pars)


def split_paragraph(paragraph: str) -> list[str]:
    return LINE_RE.findall(paragraph)
