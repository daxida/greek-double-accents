from pathlib import Path

# These words are parsed as trisyllables by syllabify
# but they actually have only two syllables.
# The grammatical term is: συνίζηση
#
# Note that many diminutives other than those extracted
# from el_GR are not included, i.e. αδερφούλια, λεφτουδάκια


def _add_endings(lemmas: set[str], endings: str) -> set[str]:
    return {f"{lemma}{ending}" for lemma in lemmas for ending in endings.split()}


# Nouns ending in ια (singular in ια, genitive in ιας).
# Ex. αρρώστια
#
# Note that if the ια form has συνίζηση so does the equivalent
# εια form, if it exists (ζήλια/εια, περιφάνια/εια)
IA_NOUN_LEMMA = {
    "αλήθει",
    "αρρώστι",
    "φτώχει",
    "φτώχι",
    "συμπόνι",
    "περηφάνει",
    "περηφάνι",
    "ορφάνι",
    "ζήλει",
    "ζήλι",
}
IA_NOUN = _add_endings(IA_NOUN_LEMMA, "α ας ες")

# Adjectives ending in ιος / ια / ιο.
# Ex. αλογίσιος
IA_ADJ_LEMMA = {
    "αλογίσι",
}
IA_ADJ = _add_endings(IA_ADJ_LEMMA, "ος ου ο ε α ας ων ους ες")

# Nouns ending in ιο (singular in ιο, plural in ια).
# Ex. μπάνιο
IO_IA_NOUN_LEMMA = {
    "δίκι",
    "μπάνι",
    "ίδι",  # Ambiguous: can also be trisyl (but much more common as bisyl)
}
IO_IA_NOUN = _add_endings(IO_IA_NOUN_LEMMA, "ο ου α ων")

# Nouns ending in ιο (singular in ιο, plural in ιος).
IO_IOS_NOUN_LEMMA = {
    "ίσκι",
}
IO_IOS_NOUN = _add_endings(IO_IOS_NOUN_LEMMA, "ος ου ο ε οι ων ους")

# Nouns ending in ι (singular in ι / plural in ια)
I_IO_NOUN = {
    "παντζούρια",
}
neuters_path = Path(__file__).parent / "etc/neuters.txt"
with neuters_path.open("r", encoding="utf-8") as f:
    I_IO_NOUN |= set(f.read().splitlines())

_FALSE_TRISYL = {
    "λόγια",  # Always bisyl as NOUN (can be trisyl as adj.)
    "έγνοια",  # Always bisyl with this orthography (but έννοια can be both)
    "κουράγιο",
    "καινούριο",
    "καινούργιο",
    "χρόνια",
    "χούγια",
    # Other ια (singular)
    "ίσια",
    *IA_NOUN,
    *IA_ADJ,
    *IO_IA_NOUN,
    *IO_IOS_NOUN,
    *I_IO_NOUN,
}

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
