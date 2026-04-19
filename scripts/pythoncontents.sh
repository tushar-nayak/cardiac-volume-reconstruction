#!/bin/bash

OUTPUT_FILE="combined_python_files.txt"
TARGET_DIR="."

# Empty the output file if it exists, or create it
> "$OUTPUT_FILE"

echo "Combining Python files..."

# Find ONLY .py files, ignoring specific directories
find "$TARGET_DIR" \
    -type d \( -name ".git" -o -name "venv" -o -name "env" -o -name "__pycache__" -o -name ".idea" -o -name ".vscode" \) -prune -o \
    -type f -name "*.py" -print0 | while IFS= read -r -d $'\0' file; do
    
    # Remove leading ./ from find output for a cleaner filename
    clean_path="${file#./}"

    # Skip the output file itself (just in case you named it something ending in .py!)
    if [[ "$clean_path" == "$OUTPUT_FILE" ]]; then
        continue
    fi

    # Append the filename and contents to the output file
    echo "$clean_path" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo -e "\n" >> "$OUTPUT_FILE"
    
done

echo "Done! Check $OUTPUT_FILE in your current directory."
