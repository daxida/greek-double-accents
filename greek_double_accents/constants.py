from pathlib import Path

# These words are parsed as trisyllables by syllabify
# but they actually have only two syllables.
_FALSE_TRISYL = {
    "δίκιο",
    "δίκια",
    "λόγια",  # Always bisyl as NOUN (can be trisyl as adj.)
    "κουράγιο",
    "καινούριο",
    "καινούργιο",
}

neuters_path = Path(__file__).parent / "etc/neuters.txt"
with neuters_path.open("r", encoding="utf-8") as f:
    _FALSE_TRISYL |= set(f.read().splitlines())

FALSE_TRISYL = frozenset(_FALSE_TRISYL)

# Grammar
# http://ebooks.edu.gr/ebooks/v/html/8547/2009/Grammatiki_E-ST-Dimotikou_html-apli/index_C8a.html
PRON_ACC_SING = {
    "με",
    "σε",
    "τον",
    "την",
    "τη",
    "το",
}

PRON_ACC_PLUR = {
    "μας",
    "σας",
    "τους",
    "τις",
    "τα",
}

PRON_ACC = {*PRON_ACC_SING, *PRON_ACC_PLUR}

PRON_GEN_SING = {
    "μου",
    "σου",
    "του",
    "της",
    "του",
}

PRON_GEN_PLUR = {
    "μας",
    "σας",
    "τους",
}

PRON_GEN = {*PRON_GEN_SING, *PRON_GEN_PLUR}

PRON = {*PRON_ACC, *PRON_GEN}
