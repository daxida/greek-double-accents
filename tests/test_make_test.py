from make_test import process


def test_should_preserve_structure() -> None:
    text = """ΤΙ ΔΕΝ ΠΡΕΠΕΙ ΝΑ ΞΕΧΑΣΩ  - Η επανάσταση στην πληροφορική...    

    - Η πληροφορική συνέβαλε κ.ά. Όμως:  - Η πληροφορική,... - Στην Εκκλησία ευλογείται..
    Αποστολή σε όλον τον κόσμο.

    ... Να υποτάσσεστε ο ένας στον άλλο με φόβο Χριστού.  Εφεσ. 5, 21-33"""  # noqa: W291
    clean_text, wrong_text = process(text)
    assert clean_text == wrong_text


def test_should_preserve_structure_two() -> None:
    text = """
    ΤΙ ΔΕΝ ΠΡΕΠΕΙ ΝΑ ΞΕΧΑΣΩ  - Η επανάσταση στην πληροφορική...    

    - Η πληροφορική συνέβαλε κ.ά. Όμως:  - Η πληροφορική,... - Στην Εκκλησία ευλογείται..
    Αποστολή σε όλον τον κόσμο.

    ... Να υποτάσσεστε ο ένας στον άλλο με φόβο Χριστού.  Εφεσ. 5, 21-33
    """  # noqa: W291
    text = "\n".join(line.strip() for line in text.splitlines())
    clean_text, wrong_text = process(text)
    assert clean_text == wrong_text
