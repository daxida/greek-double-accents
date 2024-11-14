from greek_double_accents.double_accents import join_words, split_text


def test_recompose() -> None:
    text = """
        Το 2014 τροποποιήθηκε (μείωση έκτασής του) με το ΦΕΚ 336/ Δ/24. 07. 2014.
        [...] που παραδοσιακά χρησιμοποιούν οι ψαράδες της λίμνης , την καλαμοπλεκτική κ.ά.

        Η επόμενη ενότητα ενημερώνει τον επισκέπτη για την επίδραση του ανθρώπινου παράγοντα.
        Ο αριθμός των μηκών κύματος που     προσπίπτουν στη διαχωριστική επιφάνεια.
    """

    stext = split_text(text)
    new_text = join_words(stext)

    # for a, b in zip(text, new_text):
    #     assert a == b, f"'{a}' '{b}'"
    assert text == new_text
