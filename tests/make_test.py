"""Remove double accents to make test suites.

Takes a .txt file, then:
1. Remove lines that do not contain a word with double accents.
   Output at {title}_clean.txt
2. Remove double accents from the clean text.
   Output at {title}_wrong.txt

The goal being to use both files to create a test.
"""

import argparse
from pathlib import Path

from greek_double_accents.utils import (
    has_correct_double_accent,
    join_words,
    remove_accent,
    split_text,
)


def process(text: str) -> tuple[str, str]:
    """Return both clean and wrong versions of text.

    Splits on sentences based on utils::split_text logic.
    """
    stext = split_text(text)
    lineno = sum(len(lines) for lines in stext)
    clean_lines = []
    for lines in stext:
        for line in lines:
            if any(has_correct_double_accent(word) for word in line):
                clean_lines.append(" ".join(line))

    print("Original lineno:", lineno)
    print("Clean lineno:   ", len(clean_lines))
    clean_text = "".join(clean_lines)
    wrong_text = add_errors(clean_text)

    return clean_text, wrong_text


def add_errors(text: str) -> str:
    n_correct = 0
    stext = split_text(text)
    nstext = []
    for paragraph in stext:
        nparagraph = []
        for line in paragraph:
            nline = []
            for word in line:
                if has_correct_double_accent(word):
                    n_correct += 1
                    word = remove_accent(word)
                nline.append(word)
            nparagraph.append(nline)
        nstext.append(nparagraph)

    print(f"Found {n_correct} double accents.")
    return join_words(nstext)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove double accents to make test suites.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the input file",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    ipath = args.input_path
    text = ipath.open("r", encoding="utf-8").read().strip()

    clean_text, wrong_text = process(text)

    opath_clean = ipath.with_stem(ipath.stem + "_clean")
    with opath_clean.open("w", encoding="utf-8") as file:
        file.write(clean_text)

    opath_wrong = ipath.with_stem(ipath.stem + "_wrong")
    with opath_wrong.open("w", encoding="utf-8") as file:
        file.write(wrong_text)

    print(f"Wrote new texts at:\n{opath_clean}\n{opath_wrong}")


if __name__ == "__main__":
    main()
