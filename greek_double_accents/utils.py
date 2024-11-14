import re

VOWEL_ACCENTED = re.compile(r"[έόίύάήώ]")
PUNCT = re.compile(r"[,.!?;:\n«»\"'·…]")
LINE_RE = re.compile(r"[^.!?;:…»]+(?:[.!?;:…»\n]+,?)?")


def has_accent(syllable: str) -> bool:
    return VOWEL_ACCENTED.search(syllable) is not None


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
