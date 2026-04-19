#!/bin/bash

OUTPUT_FILE="combined_output.txt"
TARGET_DIR="."

# Empty the output file if it exists, or create it
> "$OUTPUT_FILE"

echo "Combining files..."

# Find files, ignore specific directories, and process them safely handling spaces in names
find "$TARGET_DIR" \
    -type d \( -name ".git" -o -name "node_modules" -o -name "venv" -o -name "env" -o -name "__pycache__" \) -prune -o \
    -type f -print0 | while IFS= read -r -d $'\0' file; do
    
    # Remove leading ./ from find output for a cleaner filename
    clean_path="${file#./}"

    # Skip the output file itself so we don't create an infinite loop
    if [[ "$clean_path" == "$OUTPUT_FILE" ]]; then
        continue
    fi

    # Use the 'file' command to check if it's a binary file
    encoding=$(file -b --mime-encoding "$file")

    # If it is NOT binary, it's safe to read
    if [[ "$encoding" != "binary" ]]; then
        echo "$clean_path" >> "$OUTPUT_FILE"
        cat "$file" >> "$OUTPUT_FILE"
        echo -e "\n" >> "$OUTPUT_FILE"
    fi
done

echo "Done! Check $OUTPUT_FILE in your current directory."
