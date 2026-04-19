import os

def combine_files_to_txt(output_filename="combined_output.txt", target_dir="."):
    # Directories to ignore so we don't dump virtual environments or git history
    ignore_dirs = {'.git', 'venv', 'env', '__pycache__', 'node_modules', '.idea', '.vscode'}
    
    # Common binary/image extensions to skip early to save time
    ignore_exts = {
        '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.pdf', 
        '.zip', '.tar', '.gz', '.7z', '.exe', '.dll', '.so', 
        '.pyc', '.mp4', '.mp3', '.wav', '.sqlite3', '.db', '.bin'
    }

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(target_dir):
            # Modify the 'dirs' list in-place to skip ignored directories entirely
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                # Skip the output file itself to prevent recursion/duplication
                if file == output_filename:
                    continue
                    
                ext = os.path.splitext(file)[1].lower()
                if ext in ignore_exts:
                    continue
                    
                filepath = os.path.join(root, file)
                
                try:
                    # Try reading as utf-8 text. 
                    # If it fails, it's likely a binary/volume file we missed.
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        
                    # Write to the output file in the exact requested format
                    outfile.write(f"{filepath}\n")
                    outfile.write(f"{content}\n\n")
                    
                except UnicodeDecodeError:
                    # Skip files that aren't valid utf-8 text
                    pass
                except Exception as e:
                    print(f"Skipped {filepath} due to error: {e}")

if __name__ == "__main__":
    print("Combining files...")
    combine_files_to_txt()
    print("Done! Check combined_output.txt in your current directory.")
