"""
TODO:
- Make some tests > Detect False positives
- Does spacy syllabify? Yes, but low priority (syllabify is fast)

The states need to share prefixes based on detail:
1VERB > 1VERB 3ADJ etc.
This makes for easier debugging
"""

import argparse
import re
from argparse import Namespace
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import time
from typing import Any, Literal

import spacy
import spacy.cli

# For ancient greek but seems to work fine for modern
# pip install greek-accentuation==1.2.0
from greek_accentuation.accentuation import syllable_add_accent
from greek_accentuation.syllabify import ACUTE, syllabify
from spacy.tokens import Doc

from greek_double_accents.constants import (
    FALSE_TRISYL,
    PRON,
    PRON_GEN,
)

DEFAULT_PATH = Path(__file__).parent / "etc/book.txt"

PUNCT = re.compile(r"[,.!?;:\n«»\"'·…]")
VOWEL_ACCENTED = re.compile(r"[έόίύάήώ]")

# Import spacy model: greek small.
model_name = "el_core_news_sm"  # sm / md / lg
try:
    nlp = spacy.load(
        model_name,
        disable=["parser", "ner", "lemmatizer", "textcat"],
    )
    # print(nlp.pipe_names)
    # print(nlp.path)
except OSError:
    print(f"Model '{model_name}' not found. Downloading...")
    spacy.cli.download(model_name)  # type: ignore
    nlp = spacy.load(
        model_name,
        disable=["parser", "ner", "lemmatizer", "textcat"],
    )


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
    semantic_info: list[dict[str, Any]] | None = None
    # words?

    @property
    def line_ctx(self) -> str:
        fr = max(0, self.word_idx - 1)
        to = min(len(self.line), self.word_idx + 5)
        continues = self.word_idx + 5 < len(self.line)
        continues_msg = "[...]" if continues else ""
        ctx = " ".join(self.line[fr:to])
        return f"{ctx} {continues_msg}"

    def add_semantic_info(self, doc: Doc) -> Literal[0, 1]:
        """Returns 0 in case of success."""
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

        # Can happen if a word in words gets wrongly tagged as PUNCT:
        # Ex. ('δέχεσαι', 'PUNCT')
        if not len(doc_buf) == 3:
            print(f"Warning: doc_buf is of size < 3 for {self.word}")
            return 1

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
                        "PROPN",  # μου ???
                        # FIXME: ADP should always be correct 'με φρίκη' etc.
                    ), f"Unexpected pos {token.pos_} from {token.text}"
            semantic_info[idx]["case"] = token.morph.get("Case", ["X"])[0]

            # Debug
            # https://universaldependencies.org/u/feat/index.html
            semantic_info[idx]["token"] = token
            semantic_info[idx]["morph"] = token.morph
            semantic_info[idx]["verbForm"] = token.morph.get("VerbForm", ["X"])

        assert len(semantic_info) == 3, f"{semantic_info}\n{words}\n{self.word} || {self.line}"

        self.semantic_info = semantic_info

        return 0

    def show_semantic_info(self, detail: bool = False) -> None:
        wi = self
        if not wi.semantic_info:
            print(f"No semantic info for {self.word}")
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
        state_let = str(self.statemsg.state)[6]
        return f"{color}[{state_let} {self.statemsg.msg:<12}]{hend} {hctx}"

    def __str__(self) -> str:
        # return f"{self.state:<15} {self.get_line_ctx}"
        return self.detailed_str()


def add_accent(word: str) -> str:
    syls = syllabify(word)
    nsyls = syls[:-1] + [syllable_add_accent(syls[-1], ACUTE)]
    return "".join(nsyls)


def analyze_text(text: str, replace: bool, args: Namespace) -> str:
    paragraphs = text.splitlines()
    n_entries_total = 0
    record_states = Counter()
    record_msgs = Counter()

    line_re = re.compile(r"[^.!?;:…»]+(?:[.!?;:…»\n]+,?)?")
    new_text = []

    for parno, paragraph in enumerate(paragraphs, start=1):
        new_paragraph = []
        par_lines = line_re.findall(paragraph)

        for _, line in enumerate(par_lines, start=1):
            new_line = []
            if line := line.strip():
                # print(f"[{parno}:{lineno}] Line:", line, "\n", paragraph)
                line_info = analyze_line(line, parno, n_entries_total, args)
                for word, info in line_info:
                    if info is None:
                        new_line.append(word)
                    else:
                        state = info.statemsg.state
                        msg: str = info.statemsg.msg
                        n_entries_total += 1
                        record_states[state] += 1
                        record_msgs[msg] += 1

                        if replace and state == State.INCORRECT and msg != "2PUNCT":
                            new_line.append(add_accent(word))
                        else:
                            new_line.append(word)

            new_paragraph.append(" ".join(new_line))
        new_text.append(" ".join(new_paragraph))

        # TODO: remove
        if n_entries_total >= 7000:
            break

    print_summary(n_entries_total, record_states, record_msgs)

    return "\n".join(new_text)


def print_summary(
    n_entries_total: int,
    record_states: Counter,
    record_msgs: Counter,
) -> None:
    print(f"\nFound {n_entries_total} candidates.")
    total = 0
    for state, cnt in record_states.items():
        total += cnt
        print(f"{str(state)[6:]:<9} {cnt}")
    print()
    assert total == n_entries_total

    mf_key, mf_count = record_msgs.most_common(1)[0]
    print(f"The most frequent msg is: '{mf_key}' ({mf_count} times).")

    not_pending = total - record_states[State.PENDING]
    print(f"Coverage  {not_pending / total:.02f}%")


def analyze_line(
    line: str,
    lineno: int,
    n_entries_total: int,
    args: Namespace,
) -> list[tuple[str, None | Entry]]:
    words = line.split()
    cnt = 0
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
            error_code = entry.add_semantic_info(cached_doc)
            if error_code != 0:
                entry.statemsg = StateMsg(State.AMBIGUOUS, "SEMFAIL")
                line_info.append((word, entry))
                continue
            statemsg = semantic_analysis(entry)

        entry.statemsg = statemsg

        # Tested to correctly work: ignore them
        to_ignore = ("2~3SYL", "2~PRON")
        if statemsg.msg in to_ignore:
            line_info.append((word, None))
            continue

        # Print information
        if entry.statemsg.state in args.select:
            # Custom discard
            if entry.statemsg.msg != "2PUNCT":
                if not args.message or re.match(args.message, entry.statemsg.msg):
                    entry.show_semantic_info(detail=True)

        line_info.append((word, entry))

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


def semantic_analysis(entry: Entry) -> StateMsg:  # noqa: C901
    """Return True if correct, False if incorrect or undecidable."""
    if not entry.semantic_info:
        print("Warning: this should only happen in tests")
        doc = nlp(" ".join(entry.line))
        entry.add_semantic_info(doc)

    if not entry.semantic_info:
        raise ValueError(f"No semantic info for {entry.word}.")

    w1, w2, _ = entry.words[:3]
    si1, si2, si3 = entry.semantic_info
    pos1 = si1["pos"]
    pos2 = si2["pos"]
    pos3 = si3["pos"]

    if "X" in (pos1 + pos3):
        # Ambiguous: incomplete information
        return StateMsg(State.AMBIGUOUS, "NO INFO")

    # same_case12 = si1["case"] == si2["case"]
    # same_case13 = si1["case"] == si3["case"]
    same_case23 = si2["case"] == si3["case"]

    _default_statemsg = StateMsg(State.PENDING, f"1{pos1} 2{pos2} 3{pos3}")

    match pos1:
        case "VERB":
            # print(w1, si1["pos"], " > ", si1["morph"])
            # For a verb to have double accents it NEEDS (yet it does not
            # suffice), to either:
            # (1) To be a gerund (μετοχή)

            # This is equivalent to:
            # Use morph::VerbForm::Conv for βλέποντας, σφίγγοντας... i.e.
            # if si1["morph"].get("VerbForm", ["X"])[0] == "Conv":
            if w1.endswith(("οντας", "ωντας")):
                # Base rule
                if pos2 == "PRON":
                    return StateMsg(State.INCORRECT, "1VERBP 2PRON")

                # Γρηγόρης, γυρεύοντας με τον ήσυχο τόνο [...]
                # χέρι, σκοντάφτοντας σε κάθε βήμα...
                if pos2 == "ADP":
                    return StateMsg(State.CORRECT, "1VERBP 2ADP")

                # Ex. ζυγώνοντας τον άρπαξε
                if pos3 == "VERB":
                    return StateMsg(State.INCORRECT, "1VERBP 3VERB")

                # There are counter examples but there are very rare
                # and idiomatic (set phrases).
                # CEx. βλέποντάς τον σπίτι έφυγε...
                if pos3 == "NOUN" and w2 not in PRON_GEN:
                    return StateMsg(State.CORRECT, f"1VERBP 2{w2}~GEN 3NOUN")
                if pos3 == "PROPN" and w2 not in PRON_GEN:
                    return StateMsg(State.CORRECT, f"1VERBP 2{w2}~GEN 3PROPN")

                return StateMsg(State.PENDING, "1VERBP")

            # (2) To be an imperative verb (which implies, only 2nd person)
            #     Unfortunately spaCy is defective when it comes to detect person
            #     (and it's not easy due to ambiguity); i.e.
            #     'ρώτησε' is Imperative/2P but also Past/3P

            # Spacy morph::Mood::Imp (Imperative) is defective and detects nothing.

            # (2.1.) Person=3 and Number=Plur should be safe. Ex.
            # - γέροι έκλαιγαν με λυγμούς.
            # - την περιτύλιγαν με τα πλούσια σγουρά
            person = si1["morph"].get("Person", ["X"])[0]
            number = si1["morph"].get("Number", ["X"])[0]
            if person == "3" and number == "Plur":
                return StateMsg(State.CORRECT, "1VERB3PL")
            # (2.2.) Same reasoning for 1PL
            if person == "1":
                if number == "Plur":
                    return StateMsg(State.CORRECT, "1VERB1PL")
                elif number == "Sing":
                    return StateMsg(State.CORRECT, "1VERB1S")

            # - άφησέ τον ήσυχο
            # / αφήνοντάς τον ήσυχο
            if pos3 == "ADJ":
                return StateMsg(State.AMBIGUOUS, "1VERB 3ADJ")

            # - Κωνσταντίνος έσφιξε το χέρι του φίλου
            # - Μιχαήλ έσκυψε το κεφάλι στα χέρια
            if pos2 == "DET" and pos3 == "NOUN":
                return StateMsg(State.CORRECT, "1VERB 2DET 3NOUN")

        case "NOUN":
            # The pronoun must be genitive
            if w2 not in PRON_GEN:
                return StateMsg(State.CORRECT, f"1NOUN 2{w2}~GEN")

            match pos3:
                case "NOUN":
                    if same_case23:
                        return StateMsg(State.CORRECT, "13NOUN 23SC")
                case "VERB":
                    # - Το άνθρωπο της έδωσε
                    # / Το άνθρωπο τής έδωσε.
                    return StateMsg(State.AMBIGUOUS, "1NOUN 3VERB")
                case "ADP":
                    # ADP: στα, στο, στην κτλ.
                    # - τον Βασίλειο σας στην Τριαδίτσα.
                    return StateMsg(State.INCORRECT, "1NOUN 3ADP")
                case "CCONJ":
                    # CCONJ: και, κι
                    # - τα γόνατα της και σωριάστηκε στο...
                    return StateMsg(State.INCORRECT, "1NOUN 3CCONJ")
                case "DET":
                    # - Το μήνυμα σου το έγραψες ελληνικά.
                    # - Στο πρόσωπο του η φρίκη ήταν [...]
                    return StateMsg(State.INCORRECT, "1NOUN 3DET")

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

        case "PROPN":
            # High chance of being correct
            if pos3 == "VERB":
                # CEx: Ο Άνγελός μου είπε...
                return StateMsg(State.AMBIGUOUS, "1PROPN 3VERB")
            else:
                return StateMsg(State.CORRECT, "1PROPN")
        case "ADV":
            return StateMsg(State.CORRECT, "1ADV")

    return _default_statemsg


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--select",
        type=str,
        default="I",
        help="Select states using C (CORRECT), I (INCORRECT), P (PENDING), A (AMBIGUOUS)",
    )
    parser.add_argument("-m", "--message", type=str, default="", help="State message")
    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        default=DEFAULT_PATH,
        help="Path to the input file",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=None,
        help="Path to the output file",
    )

    args = parser.parse_args()

    if not args.output_path:
        args.output_path = f"{args.input_path.name}_fix.txt"

    state_map = {
        "C": State.CORRECT,
        "I": State.INCORRECT,
        "P": State.PENDING,
        "A": State.AMBIGUOUS,
    }
    args.select = [state_map[char] for char in args.select if char in state_map]

    return args


def main(replace: bool = True) -> None:
    args = parse_args()

    filepath = args.input_path
    with filepath.open("r", encoding="utf-8") as file:
        text = file.read().strip()

    start = time()
    new_text = analyze_text(text, replace, args)
    print(f"Ellapsed {time() - start:.3f}sec")

    if replace:
        opath = filepath.with_stem(f"{filepath.stem}_fix")
        with opath.open("w", encoding="utf-8") as file:
            file.write(new_text)
        print(f"The text has been updated in '{opath}'.")


if __name__ == "__main__":
    main()
