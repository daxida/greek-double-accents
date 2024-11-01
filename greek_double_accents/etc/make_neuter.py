"""Write a list with all the neuter nouns ending in iota.

Actually, it selects words that end in iota such that 'word + α'
is also in the dictionary. Turns out to be quite precise.
"""

import re
from pathlib import Path

from greek_accentuation.syllabify import syllabify

VOWEL_ACCENTED = re.compile(r"[έόίύάήώ]")

# Downloaded from http://www.elspell.gr/
ppath = Path("greek_double_accents/etc")
dic_path = ppath / "el_GR.dic"
output_path = ppath / "neuters.txt"


def load_words() -> list[str]:
    with dic_path.open("r", encoding="iso-8859-7") as dic_file:
        dic_file.readline()  # Number of entries
        words = dic_file.read().splitlines()
    return words


def filter_neuter(words: list[str]) -> list[str]:
    words_set = set(words)
    neuter_words = []
    for word in words:
        if word[0].isupper():
            continue
        if len(word) < 2:
            continue
        if word[-1] != "ι":
            continue
        if word[-2] in "αεηιου":
            continue
        plural = word + "α"
        if plural not in words_set:
            continue
        syllables = syllabify(plural)
        if len(syllables) < 3 or not VOWEL_ACCENTED.search(syllables[-3]):
            continue
        neuter_words.append(plural)

    return neuter_words


def main() -> None:
    words = load_words()
    neuter_words = filter_neuter(words)

    with output_path.open("w", encoding="utf-8") as of:
        of.writelines(line + "\n" for line in neuter_words)


if __name__ == "__main__":
    main()
