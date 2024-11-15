from greek_double_accents.main import (
    Entry,
    State,
    StateMsg,
    semantic_analysis,
    simple_entry_checks,
)


def make_test_simple(
    word: str,
    word_idx: int,
    line_str: str,
    state: State,
    msg: str,
) -> None:
    """No semantic info is needed for this."""
    entry = Entry(word, word_idx, line_str.split())
    received = simple_entry_checks(entry)
    expected = StateMsg(state, msg)
    assert received == expected


def make_test(
    word: str,
    word_idx: int,
    line_str: str,
    state: State,
    msg: str,
    spacy_checks: dict[str, dict[str, str]] | None = None,
) -> None:
    entry = Entry(word, word_idx, line_str.split())
    received: StateMsg = semantic_analysis(entry)
    expected = StateMsg(state, msg)

    # (Optional) Test that the spaCy analysis is sound.
    if spacy_checks is not None:
        sis = entry.semantic_info
        assert sis is not None

        for sc_word, sc_constraint in spacy_checks.items():
            si = None
            for _si in sis:
                # Match with the first find
                if _si["word"] == sc_word:
                    si = _si
                    break
            assert si is not None

            for grammar_type, value in sc_constraint.items():
                assert si[grammar_type] == value

    assert received == expected


def test_incorrect() -> None:
    make_test_simple(
        word="πρωτεύουσα",
        word_idx=1,
        line_str="η πρωτεύουσα του.",
        state=State.INCORRECT,
        msg="2PUNCT",
    )


def test_false_trisyllable() -> None:
    make_test_simple(
        word="μάτια",
        word_idx=1,
        line_str="τα μάτια του.",
        state=State.CORRECT,
        msg="1~3SYL",
    )


def test_verb_ambiguous() -> None:
    # Ο άνθρωπος μου είπε / Ο άνθρωπός μου είπε
    make_test(
        word="άνθρωπος",
        word_idx=1,
        line_str="O άνθρωπος μου είπε",
        state=State.AMBIGUOUS,
        msg="1NOUN 3VERB",
        spacy_checks={
            "άνθρωπος": {"pos": "NOUN"},
            "είπε": {"pos": "VERB"},
        },
    )
