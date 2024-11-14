WIP.

Analyze Greek sentences for missing double accents.

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

# Print errors and modify file.txt
gda "path/to/file.txt" --fix

# Print errors and write fixed text to "out.txt"
gda "path/to/file.txt" --fix -o "path/to/out.txt"
```
