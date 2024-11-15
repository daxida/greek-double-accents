Scan Greek text for missing double accents.

## How to install

```
pip install git+https://github.com/daxida/greek-double-accents
```

## How to run

```
# Print errors
gda "path/to/file.txt"

# Print A(mbiguous) states with message "NO INFO"
gda "path/to/file.txt" -s A -m "NO INFO"

# Modify file.txt
gda "path/to/file.txt" --fix

# Write fixed text to "out.txt"
gda "path/to/file.txt" --fix -o "path/to/out.txt"
```

## Links

The corpora used for testing were found in [clarin:el](https://inventory.clarin.gr/) at:
- [Hellenic National Corpus](https://inventory.clarin.gr/corpus/870)
- [Educational Textbooks](https://inventory.clarin.gr/corpus/908)