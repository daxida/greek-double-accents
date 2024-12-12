Scan text and fix missing double accents in Greek.

```diff
- μάρτυρας του κακουργήματος του.
+ μάρτυρας του κακουργήματός του.
```

## How to install

```
pip install git+https://github.com/daxida/greek-double-accents
```

## How to run

```
# Print errors
gda path/to/file.txt

# Fix everything under the texts folder
gda texts/** --fix

# (Experimental) 
# Use spaCy and semantic analysis to detect more errors.
gda path/to/file.txt -a

# Print A(mbiguous) states with message "NO INFO"
gda path/to/file.txt -a --select A --message "NO INFO"
```

## Rationale

By default it applies only a series of simple heuristics. While semantic analysis is supported, the results are unpredictable and highly depend on the quality of the Greek model.

The heuristics correctness is also dependent of the quality of the underlying syllabification done by [grac](https://github.com/daxida/grac). If a word like `χέρια` is wrongly considered to have three syllables, then it naturally becomes a potential candidate for an error.

Due to the dangerous thickness of the if forest that constitutes the logic body of this code, any feedback, either here or at the grac repo, is greatly appreciated.

## Links

The corpora used for testing (which unfortunately, also contain many errors...) were found in [clarin:el](https://inventory.clarin.gr/) at:
- [Hellenic National Corpus](https://inventory.clarin.gr/corpus/870)
- [Educational Textbooks](https://inventory.clarin.gr/corpus/908)