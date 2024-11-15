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
from spacy.tokens import Doc

from greek_double_accents.constants import (
    FALSE_TRISYL,
    PRON,
    PRON_GEN,
)
from greek_double_accents.utils import (
    add_accent,
    deep_flatten,
    is_simple_proparoxytone,
    split_punctuation,
    split_text,
)

WARNINGS = False


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


class State(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PENDING = "pending"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)  # For hashing
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
    # Otherwise mutable default error etc.
    statemsg: StateMsg = field(default_factory=lambda: DEFAULT_STATEMSG)
    semantic_info: list[dict[str, Any]] | None = None
    words: list[str] | None = None

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
        if self.word_idx + 3 > len(self.line):
            # Note that this could happen in titles, where there can
            # be no final punctuation.
            if WARNINGS:
                print(
                    "Warning "
                    "Faulty sentence with no final punctuation.\n"
                    f"Word: {self.word}\n"
                    f"Sentence {self.line}"
                )
            return 1

        words = [split_punctuation(w)[0] for w in self.line[self.word_idx : self.word_idx + 3]]
        self.words = words

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
        if len(doc_buf) != 3:
            if WARNINGS:
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
            semantic_info[idx]["case"] = token.morph.get("Case", ["X"])[0]

            # Debug
            # https://universaldependencies.org/u/feat/index.html
            # https://universaldependencies.org/u/pos/index.html
            semantic_info[idx]["token"] = token
            semantic_info[idx]["morph"] = token.morph

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
        line_ctx = self.line_ctx.replace("\n", "⏎")
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


TaggedWord = tuple[str, Entry | None]
TaggedLine = list[TaggedWord]
TaggedText = list[list[TaggedLine]]


def tag_text(text: str) -> TaggedText:
    stext = split_text(text)

    tagged_paragraphs = []
    for parno, paragraph in enumerate(stext, start=1):
        tagged_paragraph = []
        for line in paragraph:
            line_info = analyze_line(line, parno)
            tagged_paragraph.append(line_info)
        tagged_paragraphs.append(tagged_paragraph)

    return tagged_paragraphs


def tagged_text_to_raw(tagged_paragraphs: TaggedText) -> str:
    """Convert a TaggedText back to a string."""
    new_paragraphs = []

    for tagged_paragraph in tagged_paragraphs:
        new_paragraph = []
        for line_info in tagged_paragraph:
            new_line = []
            for word, info in line_info:
                nword = word
                if info and info.statemsg.state == State.INCORRECT:
                    nword = add_accent(word)
                new_line.append(nword)
            new_paragraph.append(" ".join(new_line))
        new_paragraphs.append("".join(new_paragraph))

    return "".join(new_paragraphs)


def analyze_text(
    text: str,
    *,
    fix: bool = True,
    print_state: str = "",
    print_statemsg: str = "",
    reference_path: Path | None = None,
    diagnostics: bool = False,
) -> str:
    tagged_paragraphs = tag_text(text)
    if not fix:
        print_tagged_text(
            tagged_paragraphs,
            print_state=print_state,
            print_statemsg=print_statemsg,
        )
    new_text = tagged_text_to_raw(tagged_paragraphs)

    if reference_path:
        compare_with_reference(tagged_paragraphs, reference_path)

    if diagnostics:
        _diagnostics(tagged_paragraphs)

    return new_text


def _diagnostics(tagged_paragraphs: TaggedText) -> None:
    record_statemsgs = Counter()
    record_states = Counter()
    record_msgs = Counter()
    for _, entry in deep_flatten(tagged_paragraphs):
        if entry:
            state = entry.statemsg.state
            msg: str = entry.statemsg.msg
            record_statemsgs[entry.statemsg] += 1
            record_states[state] += 1
            record_msgs[msg] += 1

    n_total_states = sum(record_states.values())
    print("\nDiagnostics")
    print(f"* Found {n_total_states} entries.\n")
    for state, cnt in record_states.items():
        print(f"* {str(state)[6:]:<9} {cnt}")
    print()

    if mc := record_msgs.most_common(1):
        mf_key, mf_count = mc[0]
        print(f"* The most frequent msg is: '{mf_key}' ({mf_count} times).")

    pending_counter = Counter(
        {k: v for k, v in record_statemsgs.items() if k.state == State.PENDING}
    )
    if pmc := pending_counter.most_common(1):
        pen_mf_key, pen_mf_count = pmc[0]
        print(f"* The most frequent (pending) msg is: '{pen_mf_key.msg}' ({pen_mf_count} times).")

    not_pending = n_total_states - record_states[State.PENDING]
    if n_total_states:
        print(f"* Coverage  {100 * not_pending / n_total_states:.02f}%")
    print()


def compare_with_reference(
    tagged_paragraphs: TaggedText,
    reference_path: Path,
    *,
    print_false_pn: bool = False,
) -> None:
    """Debugging function.

    Compares predicted results against actual ones.
    """
    ref_text = reference_path.open("r", encoding="utf-8").read().strip()
    ref_words = split_text(ref_text)
    assert len(ref_words) == len(tagged_paragraphs), f"{len(ref_words)} != {len(tagged_paragraphs)}"

    false_positives = []
    false_negatives = []
    true_positives = 0
    true_negatives = 0

    words_it = zip(
        deep_flatten(tagged_paragraphs),
        deep_flatten(ref_words),
    )

    for (word, entry), refword in words_it:
        if not entry:
            continue

        if word and not refword:
            raise ValueError(f"Error with the split logic? {word=} but {refword=}")

        state = entry.statemsg.state
        if word == refword and state == State.INCORRECT:
            # These are scary
            if print_false_pn:
                print(f"\033[41m[FPos]\033[0m: {entry}")
            false_positives.append(word)
        if word != refword and state == State.CORRECT:
            # These are whatever
            if print_false_pn:
                print(f"\033[43m[FNeg]\033[0m: {entry}")
            false_negatives.append(word)
        if word == refword and state == State.CORRECT:
            true_positives += 1
        if word != refword and state == State.INCORRECT:
            true_negatives += 1

    # False positives/negatives
    print("                   Predicted Positive    Predicted Negative")
    print("Actual Positive    " f"TP: {true_positives}             " f"FN: {len(false_negatives)}")
    print(
        "Actual Negative    " f"FP: {len(false_positives)}               " f"TN: {true_negatives}"
    )
    # relevant_false_positives = [f for f in false_positives if f[-1] in "αο"]
    # relevant_false_positives = sorted(set(relevant_false_positives))
    # print(f"Relevant False positives ({len(relevant_false_positives)}):")
    # print(relevant_false_positives)
    # print(f"Relevant False negatives ({len(false_negatives)}):")
    # print(sorted(set(false_negatives)))


def analyze_line(words: list[str], lineno: int, cached_doc: Doc | None = None) -> TaggedLine:
    line_info = []

    # Tested to correctly work: ignore them (=do not store the entry)
    to_ignore = ("2~3SYL", "2~CPRON")

    for idx, word in enumerate(words):
        info: Entry | None = None

        if not simple_word_checks(word, idx, len(words)):
            entry = Entry(word, idx, words, lineno)

            statemsg = simple_entry_checks(entry)

            if statemsg is not None:
                if statemsg.msg not in to_ignore:
                    entry.statemsg = statemsg
                    info = entry
            else:
                cached_doc = cached_doc or nlp(" ".join(words))
                error_code = entry.add_semantic_info(cached_doc)
                if error_code != 0:
                    statemsg = StateMsg(State.AMBIGUOUS, "SEMFAIL")
                else:
                    statemsg = semantic_analysis(entry)
                entry.statemsg = statemsg
                info = entry

        line_info.append((word, info))

    return line_info


def print_tagged_text(
    paragraphs: TaggedText,
    *,
    print_state: str = "",
    print_statemsg: str = "",
) -> None:
    if not print_state:
        return
    for paragraph in paragraphs:
        for line_info in paragraph:
            print_line_info(
                line_info,
                print_state=print_state,
                print_statemsg=print_statemsg,
            )


def print_line_info(
    line_info: TaggedLine,
    *,
    print_state: str = "",
    print_statemsg: str = "",
) -> None:
    for _, entry in line_info:
        if entry and entry.statemsg.state in print_state:
            # Custom discard
            if entry.statemsg.msg != "2PUNCT":
                if not print_statemsg or re.match(print_statemsg, entry.statemsg.msg):
                    entry.show_semantic_info(detail=True)


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

    if not is_simple_proparoxytone(word):
        return True

    return False


def simple_entry_checks(entry: Entry) -> StateMsg | None:
    """Does NOT use semantic analysis."""
    # Verify that the word is not banned (False trisyllables)
    if entry.word.lower() in FALSE_TRISYL:
        return StateMsg(State.CORRECT, "1~3SYL")

    # Next word (we assume it exists), must be a pronoun
    detpron, punct = split_punctuation(entry.line[entry.word_idx + 1])
    if detpron not in PRON:
        # CPRON = Custom PRON, to differentiate it from spaCy PRON
        return StateMsg(State.CORRECT, "2~CPRON")

    # This is a mistake and it is fixable
    if punct:
        return StateMsg(State.INCORRECT, "2PUNCT")

    return None


def semantic_analysis(entry: Entry) -> StateMsg:  # noqa: C901
    """Return True if correct, False if incorrect or undecidable."""
    if not entry.semantic_info:
        if WARNINGS:
            print("Warning: this should only happen in tests")
        doc = nlp(" ".join(entry.line))
        entry.add_semantic_info(doc)

    if not entry.semantic_info:
        raise ValueError(f"No semantic info for {entry.word}.")

    if not entry.words:
        raise ValueError(f"No words for {entry.word}.")

    w1, w2, w3 = entry.words[:3]
    si1, si2, si3 = entry.semantic_info
    pos1 = si1["pos"]
    pos2 = si2["pos"]
    pos3 = si3["pos"]

    if "X" in (pos1 + pos3):
        # Ambiguous: incomplete information
        return StateMsg(State.AMBIGUOUS, "NO INFO")

    _default_statemsg = StateMsg(State.PENDING, f"1{pos1} 2{pos2} 3{pos3}")

    if pos3 in ("CCONJ", "SCONJ"):
        # Works like a stop, and so the logic of punctuation
        # seems to also apply here.
        #
        # VERBS
        # > CCONJ: και, κι
        # - αυτό, άκουσε τον κι αυτόν, εσύ, [...]
        # - εκλεκτό, ζύμωσε το και κάνε πίττες».
        # NOUNS
        # > CCONJ
        # - τα γόνατα της και σωριάστηκε στο...
        # > SCONJ
        # - η κράτηση τους όταν περάσει ο [...]
        # - στα επιτεύγματα της καθώς φέτος κλείνουν
        # ADJS
        # > CCONJ
        # - τα μυστήρια της και ιδιαίτερα με
        # > SCONJ
        # - το τηλέφωνο σας όταν βρίσκεστε σε
        return StateMsg(State.INCORRECT, "CONJ")

    if pos2 not in ("DET", "PRON", "ADP"):
        # It may be wrongly tagged as NOUN | VERB | ADV | NUM
        return StateMsg(State.AMBIGUOUS, "2POS WRONG")

    match pos1:
        case "VERB":
            # All gerunds + imperative end in α, ε, υ (from ου) or ς
            # This quick check detects a lot of correct tenses.
            if w1[-1] not in "αευς":
                return StateMsg(State.CORRECT, "1VERB ENDING")

            if pos2 == "ADP":
                # - να αμείβεται σε αρκετές περιπτώσεις χαμηλότερα [...]
                # - που βρίσκεται σε φάση υποχώρησης και [...]
                # - Γρηγόρης, γυρεύοντας με τον ήσυχο τόνο [...]
                return StateMsg(State.CORRECT, "1VERB 2ADP")

            # - άφησέ τον ήσυχο
            # / αφήνοντάς τον ήσυχο
            if pos3 == "ADJ":
                return StateMsg(State.AMBIGUOUS, "1VERB 3ADJ")

            # - άφησέ τον γρήγορα
            # / αφήνοντάς τον πολύ ήσυχο
            if pos3 == "ADV":
                return StateMsg(State.AMBIGUOUS, "1VERB 3ADV")

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

                match pos3:
                    case "ADP":
                        # - Παΐσιος κοιτάζοντας τον με πικραμένη απορία.  VERB DET ADP
                        # - άλογα πηγαίνοντας τον στο σταθμό της [...] VERB DET ADP
                        return StateMsg(State.INCORRECT, "1VERBP 3ADP")
                    case "DET":
                        # - Βλέποντας τον η Φένια έβαλε [...] VERB DET DET
                        # - και δείχνοντας του τη «θεραπευμένη»  VERB DET DET
                        return StateMsg(State.INCORRECT, "1VERBP 3DET")
                    case "VERB":
                        # - ζυγώνοντας τον άρπαξε
                        # / κάπως κομπιάζοντάς τη ρώτησε όσο μπορούσε πιο ευγενικά,
                        return StateMsg(State.AMBIGUOUS, "1VERBP 3VERB")
                    case "NOUN" | "PROPN":
                        # There are counter examples but rare and idiomatic (set phrases).
                        # / βλέποντάς τον σπίτι έφυγε...
                        if w2 not in PRON_GEN:
                            return StateMsg(State.CORRECT, f"1VERBP 2{w2}~GEN 3NOUN-PROPN")

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
            # (2.2.) Same reasoning for 1 (Plur and Sing)
            if person == "1":
                return StateMsg(State.CORRECT, "1VERB1")

            match pos3:
                case "ADP":
                    # - και συγκρίνετε την με την προηγούμενη.
                    return StateMsg(State.INCORRECT, "1VERB 3ADV")
                case "DET":
                    # - Φίλησε μου τη Νικόλ και [...]
                    return StateMsg(State.CORRECT, "1VERB 3DET")
                case "NOUN":
                    # - Κωνσταντίνος έσφιξε το χέρι του φίλου
                    # - Μιχαήλ έσκυψε το κεφάλι στα χέρια
                    # - ζωής, παρακάμπτοντας τους μύθους.
                    # / του, μεταφέροντας του αξίες, πρότυπα και [...]
                    return StateMsg(State.AMBIGUOUS, "1VERB 3NOUN")
                case "NUM":
                    # - ποδοσφαίρου έγινε το 1872 μεταξύ Αγγλίας [...]
                    # / Επανάλαβε το 2-3 φορές την [...]
                    # / άσκηση εκτέλεσε την 2 φορές από [...]
                    return StateMsg(State.AMBIGUOUS, "1VERB 3NUM")
                case "PRON":
                    # / Ρώτησε τους τι τους αρέσει [...]
                    return StateMsg(State.INCORRECT, "1VERB 3PRON")
                case "PROPN":
                    # - πεδίο βαρύτητας της Γης είναι ομογενές.
                    # - της προστάτευε τον Ιωνά από τον
                    # - «Αυτός βλασφήμησε το Θεό και το [...]
                    return StateMsg(State.CORRECT, "1VERB 3PROPN")
                case "VERB":
                    # Is this even possible?
                    return StateMsg(State.CORRECT, "1VERB 3VERB")

        case "NOUN":
            # The pronoun must be genitive
            if w2 not in PRON_GEN:
                return StateMsg(State.CORRECT, f"1NOUN 2{w2}~GEN")

            match pos3:
                case "ADJ":
                    # - Η ανάλυση της καταναλωτικής συμπεριφοράς, του [...]
                    # - σαν Πρόεδρος της Επίτιμης Επιτροπής θα [...]
                    return StateMsg(State.CORRECT, "1NOUN 3ADJ")
                case "ADP":
                    # ADP: στα, στο, στην κτλ.
                    # - τον Βασίλειο σας στην Τριαδίτσα.
                    return StateMsg(State.INCORRECT, "1NOUN 3ADP")
                case "ADV":
                    # / άφησε το αυτοκίνητό του κοντά
                    return StateMsg(State.INCORRECT, "1NOUN 3ADV")
                case "DET":
                    # - Το μήνυμα σου το έγραψες ελληνικά.
                    # - Στο πρόσωπο του η φρίκη ήταν [...]
                    # / Η κύρια μου τα εξήγησε όλα, [...]
                    return StateMsg(State.AMBIGUOUS, "1NOUN 3DET")
                case "NOUN" | "PROPN":
                    # if same_case23:
                    #     return StateMsg(State.CORRECT, "1NOUN 23SC 3NOUN-PROPN")
                    return StateMsg(State.CORRECT, "1NOUN 3NOUN-PROPN")
                case "NUM":
                    # - Τον Οκτώβριο του 2007 παρότι ουδέποτε
                    # - στην προέλευση της πρώτης ύλης τους, [...]
                    return StateMsg(State.CORRECT, "1NOUN 3NUM")
                case "PART":
                    if w3 == "μη":
                        # - στην περίπτωση της μη αντιστρεπτής μεταβολής.
                        return StateMsg(State.CORRECT, "1NOUN 3MI")
                    else:
                        # / Το κείμενο σας δεν μπορεί να [...]
                        # / η κίνηση τους δεν είναι ευθύγραμμη [...]
                        return StateMsg(State.INCORRECT, "1NOUN 3PART")
                case "PRON":
                    # - την εξέλιξη της οποίας παρακολουθήσαμε
                    # - χορευτικό ρεπερτόριο του κάθε νησιού παρουσιάζει
                    # / τη συγκίνηση της που βρίσκεται στην
                    return StateMsg(State.AMBIGUOUS, "1NOUN 3PRON")
                case "VERB":
                    # - Το άνθρωπο της έδωσε
                    # / Το άνθρωπο τής έδωσε.
                    return StateMsg(State.AMBIGUOUS, "1NOUN 3VERB")

        case "ADJ":
            # The pronoun must be genitive
            if w2 not in PRON_GEN:
                return StateMsg(State.CORRECT, f"1ADJ 2{w2}~GEN")

            match pos3:
                case "ADJ":
                    # - πιο αντάξια του ανθρώπινου γένους.
                    return StateMsg(State.CORRECT, "1ADJ 3ADJ")
                case "ADP":
                    # / τις δεξιότητες τους για τη συλλογή [...]
                    return StateMsg(State.INCORRECT, "1ADJ 3ADP")
                case "ADV":
                    # / ως αντιπρόσωποι του πάνω στη γη.
                    # / οι απόγονοι σου όπως οι κόκκοι [...]
                    return StateMsg(State.INCORRECT, "1ADJ 3ADV")
                case "DET":
                    # Note that ίδιο, the main case, is now treated as
                    # false trisyllable.
                    # / τον ίδιο τους τον εαυτό.
                    # / στο ημερολόγιο σας το χρονικό μιας [...]
                    return StateMsg(State.INCORRECT, "1ADJ 3DET")
                case "NOUN":
                    # Ambiguous for nominalized adjectives even in the same case
                    # * του όμορφου του Νίκου / του όμορφού του Νίκου
                    # - στον αντίστοιχο του δεκαδικό.
                    # / στο προηγούμενό του μέγεθος,
                    # / το μεγαλύτερό της πλεονέκτημα
                    return StateMsg(State.AMBIGUOUS, "1ADJ 3NOUN")
                case "NUM":
                    # - είναι μεγαλύτερο του 5
                    return StateMsg(State.CORRECT, "1ADJ 3NUM")
                case "PART":
                    if w3 == "μη":
                        return StateMsg(State.CORRECT, "1ADJ 3MI")
                    else:
                        # / που παρόμοιά της δεν υπάρχει αλλού [...]
                        return StateMsg(State.INCORRECT, "1ADJ 3PART")
                case "PRON":
                    return StateMsg(State.INCORRECT, "1ADJ 3PRON")
                case "PROPN":
                    # - το αντίστοιχο του Σόλωνα.
                    # - του ίδιου του Καποδίστρια, για λόγους [...]
                    return StateMsg(State.CORRECT, "1ADJ 3PROPN")
                case "VERB":
                    return StateMsg(State.AMBIGUOUS, "1ADJ 3VERB")

        case "PROPN":
            # High chance of being correct
            if pos3 == "VERB":
                # / Ο Άνγελός μου είπε...
                return StateMsg(State.AMBIGUOUS, "1PROPN 3VERB")
            else:
                return StateMsg(State.CORRECT, "1PROPN")

        case "ADV":
            # - το τοπίο ανάμεσα στα δυο χωριά
            # / χώρες, ανάμεσά τους και η Ελλάδα.
            if w1.lower() == "ανάμεσα":
                return StateMsg(State.AMBIGUOUS, "1ADVANA")

            return StateMsg(State.CORRECT, "1ADV")

        case _:
            # Not NOUN | PROPN | VERB | ADJ | ADV
            return StateMsg(State.CORRECT, "REST")

    return _default_statemsg


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the input file",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Replace the input file",
    )
    parser.add_argument(
        "-s",
        "--select",
        type=str,
        default="I",
        help="Select states using C (CORRECT), I (INCORRECT), P (PENDING), A (AMBIGUOUS)",
    )
    parser.add_argument("-m", "--message", type=str, default="", help="State message")
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        help="Path to the output file",
    )
    parser.add_argument(
        "-r",
        "--reference-path",
        type=Path,
        help="(DEBUG) Path to the reference file",
    )
    parser.add_argument(
        "-d",
        "--diagnostics",
        action="store_true",
        help="(DEBUG) Enable diagnostics output",
    )

    args = parser.parse_args()

    if not args.output_path:
        args.output_path = args.input_path

    state_map = {
        "C": State.CORRECT,
        "I": State.INCORRECT,
        "P": State.PENDING,
        "A": State.AMBIGUOUS,
    }
    args.select = [state_map[char] for char in args.select if char in state_map]

    return args


def main() -> None:
    args = parse_args()

    filepath = args.input_path
    with filepath.open("r", encoding="utf-8") as file:
        text = file.read().strip()

    start = time()
    new_text = analyze_text(
        text,
        fix=args.fix,
        print_state=args.select,
        print_statemsg=args.message,
        reference_path=args.reference_path,
        diagnostics=args.diagnostics,
    )
    print(f"Ellapsed {time() - start:.3f}sec")

    if args.fix:
        opath = args.output_path
        with opath.open("w", encoding="utf-8") as file:
            file.write(new_text)
        print(f"The text has been updated in '{opath}'.")


if __name__ == "__main__":
    main()
