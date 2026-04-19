#!/bin/bash

# Define the name of the output zip file
OUTPUT_ZIP="cardiac_reconstruction_project.zip"
TEMP_DIR="cardiac_temp_isolate"

# Create a clean temporary directory
mkdir -p $TEMP_DIR

echo "Isolating 3D Cardiac Reconstruction files..."

# List of specific files identified as relevant to the cardiac project
FILES=(
    "minimal_starter_6.py"
    "ablation_studies_6.py"
    "viewer_3d_reconstruction.py"
    "mywandb.py"
    "point_sampling.py"
    "inference.py"
)

# Copy the files to the temp directory if they exist
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$TEMP_DIR/"
        echo "Added: $file"
    else
        echo "Warning: $file not found in the current directory."
    fi
done

# Check if we actually found any files
if [ "$(ls -A $TEMP_DIR)" ]; then
    # Create the zip folder
    zip -r "$OUTPUT_ZIP" "$TEMP_DIR"
    
    echo "------------------------------------------------"
    echo "Success! Project isolated in: $OUTPUT_ZIP"
    echo "------------------------------------------------"
else
    echo "Error: No project files were found. Make sure you are in the correct directory."
fi

# Cleanup temporary directory
rm -rf "$TEMP_DIR"
