from greek_double_accents.main import (
    Entry,
    State,
    StateMsg,
    lazy_load_spacy_model,
    semantic_analysis,
    simple_entry_checks,
    simple_word_checks,
)

lazy_load_spacy_model()


def make_test_simple(
    word: str,
    word_idx: int,
    line_str: str,
    state_or_bool: State | bool,
    msg: str,
) -> None:
    """No semantic info is needed for this."""
    entry = Entry(word, word_idx, line_str.split())
    received = simple_entry_checks(entry)
    if isinstance(state_or_bool, bool):
        assert received == state_or_bool
    else:
        expected = StateMsg(state_or_bool, msg)
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
        state_or_bool=State.INCORRECT,
        msg="2PUNCT",
    )


def test_incorrect_ellipsis() -> None:
    """Should not return INCORRECT on ellipsis."""
    make_test_simple(
        word="πρωτεύουσα",
        word_idx=1,
        line_str="η πρωτεύουσα του...",
        state_or_bool=False,
        msg="2PUNCT",
    )
    make_test_simple(
        word="πρωτεύουσα",
        word_idx=1,
        line_str="η πρωτεύουσα του…",
        state_or_bool=False,
        msg="2PUNCT",
    )


def test_incorrect_conj() -> None:
    for word in {"και", "κι", "όταν"}:
        make_test_simple(
            word="πρωτεύουσα",
            word_idx=1,
            line_str=f"η πρωτεύουσα του {word}",
            state_or_bool=State.INCORRECT,
            msg="3CONJ",
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


def test_word_checks() -> None:
    assert simple_word_checks("word_at_end", 0, 1) is True
    assert simple_word_checks("προτεύουσα.", 0, 100) is True
    assert simple_word_checks("προτεύουσα", 0, 100) is False
