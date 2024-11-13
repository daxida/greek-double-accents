from pathlib import Path

# These words are parsed as trisyllables by syllabify
# but they actually have only two syllables.
# The grammatical term is: συνίζηση
#
# Note that many diminutives other than those extracted
# from el_GR are not included, i.e. αδερφούλια, λεφτουδάκια

# Nouns ending in ια (singular in ια, genitive in ιας).
# Note that if the ια form has συνίζηση so does the equivalent
# εια form, if it exists (ζήλια/εια, περιφάνια/εια)
IA_NOUN = {
    "αλήθεια",
    "αρρώστια",
    "φτώχεια",
    "φτώχια",
    "συμπόνια",
    "περηφάνεια",
    "περηφάνια",
    "ορφάνια",
    "ζήλεια",
    "ζήλια",
}
IA_NOUN |= {f"{noun}ς" for noun in IA_NOUN}
IA_NOUN |= {f"{noun[:-1]}ες" for noun in IA_NOUN}

# Adjectives ending in ιος / ια / ιο.
# Use the lemma (i.e. αλογίσι for αλογίσιος)
IA_ADJ_LEMMA = {
    "αλογίσι",
}
IA_ADJ = set()
_endings = "ος ου ο ε α ας ων ους ες".split()
for lemma in IA_ADJ_LEMMA:
    for ending in _endings:
        IA_ADJ.add(f"{lemma}{ending}")

# Nouns ending in ιο (singular in ιο, plural in ια).
IO_IA_NOUN_LEMMA = {
    "δίκι",
    "μπάνι",
    "ίδι",  # Ambiguous: can also be trisyl (but much more common as bisyl)
}
IO_IA_NOUN = set()
_endings = "ο ου α ων".split()
for lemma in IO_IA_NOUN_LEMMA:
    for ending in _endings:
        IO_IA_NOUN.add(f"{lemma}{ending}")

# Nouns ending in ιο (singular in ιο, plural in ιος).
IO_IOS_NOUN_LEMMA = {
    "ίσκι",
}
IO_IOS_NOUN = set()
_endings = "ος ου ο ε οι ων ους".split()
for lemma in IO_IOS_NOUN_LEMMA:
    for ending in _endings:
        IO_IOS_NOUN.add(f"{lemma}{ending}")

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
