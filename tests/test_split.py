from greek_double_accents.utils import join_words, split_text


def make_test(text: str) -> None:
    stext = split_text(text)
    new_text = join_words(stext)

    # for a, b in zip(text, new_text):
    #     assert a == b, f"'{a}' '{b}'"
    assert text == new_text


def test_recompose() -> None:
    text = """
    Το 2014 τροποποιήθηκε (μείωση έκτασής του) με το ΦΕΚ 336/ Δ/24. 07. 2014.
    [...] που παραδοσιακά χρησιμοποιούν οι ψαράδες της λίμνης , την καλαμοπλεκτική κ.ά.

    Η επόμενη ενότητα ενημερώνει τον επισκέπτη για την επίδραση του ανθρώπινου παράγοντα.
    Ο αριθμός των μηκών κύματος που     προσπίπτουν στη διαχωριστική επιφάνεια.
    """
    make_test(text)


def test_recompose_initial_punct() -> None:
    text = """
    ...της σε όλον τον κόσμο.
    ... Να υποτάσσεστε ο ένας στον άλλο με φόβο Χριστού.
    - Κίνησε το με μεγάλη ταχύτητα.
    -Κίνησε.
    """
    text = "\n".join(line.strip() for line in text.splitlines())
    make_test(text)
