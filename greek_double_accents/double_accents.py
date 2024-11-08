"""
TODO:
- Clean
- Time it
- Make some tests

Does spacy syllabify?
"""

import argparse
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import spacy
import spacy.cli

# For ancient greek but seems to work fine for modern
# pip install greek-accentuation==1.2.0
from greek_accentuation.accentuation import syllable_add_accent
from greek_accentuation.syllabify import ACUTE, syllabify
from spacy.tokens import Doc

from .constants import (
    FALSE_TRISYL,
    PRON,
    PRON_GEN,
)

PUNCT = re.compile(r"[,.!?;:\n«»\"'·…]")
VOWEL_ACCENTED = re.compile(r"[έόίύάήώ]")

# Import spacy model: greek small.
model_name = "el_core_news_sm"
try:
    nlp = spacy.load(model_name)
    # print(nlp.path)
except OSError:
    print(f"Model '{model_name}' not found. Downloading...")
    spacy.cli.download(model_name)
    nlp = spacy.load(model_name)


def split_punctuation(word: str) -> tuple[str, str | None]:
    """Splits a word into its base form and any trailing punctuation."""
    if mtch := PUNCT.search(word):
        return word[: mtch.start()], word[mtch.start() :]
    return word, None


class State(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PENDING = "pending"
    AMBIGUOUS = "ambiguous"


@dataclass
class StateMsg:
    state: State
    msg: str


DEFAULT_STATEMSG = StateMsg(State.PENDING, "TODO")


@dataclass
class Entry:
    word: str
    word_idx: int
    line: list[str]
    line_number: int = 0
    entry_id: int = 0
    # Otherwise mutable default error etc.
    statemsg: StateMsg = field(default_factory=lambda: DEFAULT_STATEMSG)
    semantic_info: list[dict[str, str]] | None = None
    # words?

    @property
    def line_ctx(self) -> str:
        fr = max(0, self.word_idx - 1)
        to = min(len(self.line), self.word_idx + 5)
        continues = self.word_idx + 5 < len(self.line)
        continues_msg = "[...]" if continues else ""
        ctx = " ".join(self.line[fr:to])
        return f"{ctx} {continues_msg}"

    def add_semantic_info(self, doc: Doc) -> None:
        assert self.word_idx < len(self.line) - 2, "Faulty sentence with no final punctuation."

        words = [split_punctuation(w)[0] for w in self.line[self.word_idx : self.word_idx + 3]]
        self.words = words
        assert len(words) == 3

        # TODO: Keep the tokens for the whole sentence to debug

        # Reconcile both splitting methods (can FAIL but rare)
        # Uses the fact the we know that word 1 and 2 have no punctuation
        doc_buf = []
        for token in doc:
            if token.pos_ == "PUNCT":
                continue
            cur_idx = len(doc_buf)
            if words[cur_idx] in token.text:
                doc_buf.append(token)
                if cur_idx == 2:
                    break
            else:
                doc_buf = []
        assert len(doc_buf) == 3

        # The key can't be the word in case of duplicates:
        # words = ['φρούριο', 'σε', 'φρούριο']
        semantic_info = [{"None": "None"} for _ in range(3)]
        for idx, token in enumerate(doc_buf):
            word = token.text
            semantic_info[idx] = {}
            semantic_info[idx]["word"] = word
            semantic_info[idx]["pos"] = token.pos_
            if word == words[1]:
                # Why does spacy thinks that σου is an ADV/ADJ/NOUN?
                if word != "σου":
                    assert token.pos_ in (
                        "DET",
                        "PRON",
                        "ADP",  # με
                        # FIXME: ADP should always be correct 'με φρίκη' etc.
                    ), f"Unexpected pos {token.pos_} from {token.text}"
            semantic_info[idx]["case"] = token.morph.get("Case", ["X"])[0]

        assert len(semantic_info) == 3, f"{semantic_info}\n{words}\n{self.word} || {self.line}"

        self.semantic_info = semantic_info

    def show_semantic_info(self, detail: bool = False) -> None:
        wi = self
        if not wi.semantic_info:
            print("No semantic info")
            return

        cyan = "\033[36m"
        cend = "\033[0m"

        si1, si2, si3 = wi.semantic_info
        semantic_info = (
            f"{cyan}{si1['pos']} {si2['pos']} {si3['pos']}"
            " || "
            f"{si1['case']} {si2['case']} {si3['case']}{cend}"
        )
        print(self if detail else "", semantic_info)

    def detailed_str(self) -> str:
        hstart = "\033[1m"
        hend = "\033[0m"
        line_ctx = self.line_ctx
        hctx = line_ctx.replace(self.word, f"{hstart}{self.word}{hend}")
        state_colors = {
            State.CORRECT: "\033[32m",  # Green
            State.INCORRECT: "\033[31m",  # Red
            State.PENDING: "\033[33m",  # Yellow
            State.AMBIGUOUS: "\033[34m",  # Blue
        }

        color = state_colors.get(self.statemsg.state, "\033[0m")
        return f"{color}{str(self.statemsg.state)[6:]:<9} [{self.statemsg.msg:<12}]{hend} {hctx}"

    def __str__(self) -> str:
        # return f"{self.state:<15} {self.get_line_ctx}"
        return self.detailed_str()


def add_accent(word: str) -> str:
    syls = syllabify(word)
    nsyls = syls[:-1] + [syllable_add_accent(syls[-1], ACUTE)]
    return "".join(nsyls)


def analyze_text(text: str, replace: bool, print_states: list[State]) -> str:
    paragraphs = text.splitlines()
    n_entries_total = 0
    record = {
        State.CORRECT: 0,
        State.INCORRECT: 0,
        State.PENDING: 0,
        State.AMBIGUOUS: 0,
    }

    line_re = re.compile(r"[^.!?;:…»]+(?:[.!?;:…»\n]+,?)?")
    new_text = []

    for parno, paragraph in enumerate(paragraphs, start=1):
        new_paragraph = []
        par_lines = line_re.findall(paragraph)

        for lineno, line in enumerate(par_lines, start=1):
            new_line = []
            if line := line.strip():
                # print(f"[{parno}:{lineno}] Line:", line, "\n", paragraph)
                line_info = analyze_line(line, parno, n_entries_total, print_states)
                for word, info in line_info:
                    if info is None:
                        new_line.append(word)
                    else:
                        state = info.statemsg.state
                        msg = info.statemsg.msg
                        n_entries_total += 1
                        record[state] += 1

                        if replace and state == State.INCORRECT and msg != "2PUNCT":
                            new_line.append(add_accent(word))
                        else:
                            new_line.append(word)

            new_paragraph.append(" ".join(new_line))
        new_text.append(" ".join(new_paragraph))

        # TODO: remove
        if n_entries_total >= 7000:
            break

    print(f"\nFound {n_entries_total} candidates.")
    for state, cnt in record.items():
        print(f"{str(state)[6:]:<9} {cnt}")

    return "\n".join(new_text)


def analyze_line(
    line: str,
    lineno: int,
    n_entries_total: int,
    print_states: list[State],
) -> list[tuple[str, None | Entry]]:
    words = line.split()
    cnt = 0
    states = []
    line_info = []
    cached_doc = None

    for idx, word in enumerate(words):
        if simple_word_checks(word, idx, len(words)):
            line_info.append((word, None))
            continue

        # From here on, it is tricky
        entry = Entry(word, idx, words, lineno, n_entries_total + cnt)
        cnt += 1

        statemsg = simple_entry_checks(entry)
        if statemsg == DEFAULT_STATEMSG:
            cached_doc = cached_doc or nlp(line)
            entry.add_semantic_info(cached_doc)
            statemsg = semantic_analysis(entry)

        entry.statemsg = statemsg

        # Tested to correctly work: ignore them
        to_ignore = ("2~3SYL", "2~PRON")
        if statemsg.msg in to_ignore:
            line_info.append((word, None))
            continue

        # Print information
        if entry.statemsg.state in print_states:
            if entry.statemsg.msg != "2PUNCT":
                entry.show_semantic_info(detail=True)
        # print(entry)
        # # Debug print semantic info if PENDING
        # if entry.statemsg.state == State.PENDING:
        #     entry.show_semantic_info()

        line_info.append((word, entry))
        states.append(entry.statemsg.state)

    return line_info


def simple_word_checks(word: str, idx: int, lwords: int) -> bool:
    """Discard a word based on punctuation and number of syllables.
    Returns True if we can discard the word, False otherwise.
    """
    # Word is at the end
    if idx == lwords - 1:
        return True

    # Punctuation automatically makes this word correct
    word, wpunct = split_punctuation(word)
    if wpunct:
        return True

    # Need at least three syllables, with the antepenult accented...
    syllables = syllabify(word)
    if len(syllables) < 3 or not VOWEL_ACCENTED.search(syllables[-3]):
        return True
    # ...and the last one unaccented (otherwise it is not an error)
    if VOWEL_ACCENTED.search(syllables[-1]):
        return True

    return False


def simple_entry_checks(entry: Entry) -> StateMsg:
    """Does NOT use semantic analysis."""
    # Verify that the word is not banned (False trisyllables)
    if entry.word in FALSE_TRISYL:
        return StateMsg(State.CORRECT, "1~3SYL")

    # Next word (we assume it exists), must be a pronoun
    detpron, punct = split_punctuation(entry.line[entry.word_idx + 1])
    if detpron not in PRON:
        return StateMsg(State.CORRECT, "2~PRON")

    # This is a mistake and it is fixable
    if punct:
        return StateMsg(State.INCORRECT, "2PUNCT")

    return DEFAULT_STATEMSG


def semantic_analysis(wi: Entry) -> StateMsg:
    """Return True if correct, False if incorrect or undecidable."""
    if not wi.semantic_info:
        print("Warning: this should only happen in tests")
        doc = nlp(" ".join(wi.line))
        wi.add_semantic_info(doc)

    if not wi.semantic_info:
        raise ValueError("No semantic info added.")

    w1, w2, w3 = wi.words[:3]
    si1, si2, si3 = wi.semantic_info
    pos1 = si1["pos"]
    pos2 = si2["pos"]
    pos3 = si3["pos"]

    if "X" in (pos1 + pos3):
        # Ambiguous: incomplete information
        return StateMsg(State.AMBIGUOUS, "NO INFO")

    same_case12 = si1["case"] == si2["case"]
    same_case13 = si1["case"] == si3["case"]
    same_case23 = si2["case"] == si3["case"]

    match pos1:
        case "VERB":
            return StateMsg(State.PENDING, "1VERB")
        case "NOUN":
            # The pronoun must be genitive
            if w2 not in PRON_GEN:
                return StateMsg(State.CORRECT, f"1NOUN 2{w2}~GEN")

            match pos3:
                case "NOUN":
                    if same_case23:
                        return StateMsg(State.CORRECT, "13NOUN 23SC")
                case "VERB":
                    # CEx: Το άνθρωπο της έδωσε / Το άνθρωπο τής έδωσε.
                    return StateMsg(State.AMBIGUOUS, "1NOUN 3VERB")
                case "ADP":
                    # στα, στο, στην κτλ.
                    # Ex: τον Βασίλειο σας στην Τριαδίτσα.
                    return StateMsg(State.INCORRECT, "1NOUN 3ADP")
                case "CCONJ":
                    # και, κι
                    # Ex: τα γόνατα της και σωριάστηκε στο...
                    return StateMsg(State.INCORRECT, "1NOUN 3CCONJ")

            return StateMsg(State.PENDING, "1NOUN")
        case "ADJ":
            # The pronoun must be genitive
            if w2 not in PRON_GEN:
                return StateMsg(State.CORRECT, f"1ADJ 2{w2}~GEN")

            match pos3:
                # Ambiguous for nominalized adjectives even in the same case
                # CEx. του όμορφου του Νίκου / του όμορφού του Νίκου
                case "NOUN":
                    if same_case23:
                        return StateMsg(State.AMBIGUOUS, "1ADJ 2NOUN 23SC")
                case "VERB":
                    return StateMsg(State.AMBIGUOUS, "1ADJ 3VERB")

            return StateMsg(State.PENDING, "1ADJ")
        case "PROPN":
            # High chance of being correct
            if pos3 == "VERB":
                # CEx: Ο Άνγελός μου είπε...
                return StateMsg(State.AMBIGUOUS, "1PROPN 3VERB")
            else:
                return StateMsg(State.CORRECT, "1PROPN")
        case "ADV":
            return StateMsg(State.CORRECT, "1ADV")

    return DEFAULT_STATEMSG


def parse_args() -> list[State]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--select",
        type=str,
        default="I",
        help="Select states using C (CORRECT), I (INCORRECT), P (PENDING), A (AMBIGUOUS)",
    )

    args = parser.parse_args()

    state_map = {
        "C": State.CORRECT,
        "I": State.INCORRECT,
        "P": State.PENDING,
        "A": State.AMBIGUOUS,
    }

    selected_states = [state_map[char] for char in args.select if char in state_map]
    return selected_states


def main(replace: bool = True) -> None:
    print_states = parse_args()

    filepath = Path(__file__).parent / "etc/book.txt"
    with filepath.open("r", encoding="utf-8") as file:
        text = file.read().strip()
        new_text = analyze_text(text, replace, print_states)

    if replace:
        opath = filepath.with_stem("book_fix")
        with opath.open("w", encoding="utf-8") as file:
            file.write(new_text)
        print(f"The text has been updated in '{opath}'.")


if __name__ == "__main__":
    main()
