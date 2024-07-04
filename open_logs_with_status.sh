#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 <directory> <keyword1> [keyword2] ... [log search string]"
    echo "Example: $0 results/llama-1B/64_GPUS timeout pending \"Some NCCL operations have failed\""
    exit 1
}

# Check if at least two arguments are provided (directory and at least one keyword)
if [ $# -lt 2 ]; then
    usage
fi

# First argument is the directory
directory="$1"
shift  # Remove the first argument (directory) from the list

# Check if the directory exists
if [ ! -d "$directory" ]; then
    echo "Error: Directory '$directory' does not exist."
    exit 1
fi

# Initialize variables
keywords=()
log_search_string=""
files_found=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    if [[ "$1" == *" "* ]] || [[ "${#keywords[@]}" -ge 1 ]]; then
        # If an argument contains a space or we already have at least one keyword,
        # treat this and all following args as the log search string
        log_search_string="$*"
        break
    else
        keywords+=("$1")
        shift
    fi
done

echo "Directory: $directory"
echo "Keywords: ${keywords[*]}"
echo "Log search string: $log_search_string"

# Find all .txt files in the specified directory
txt_files=$(find "$directory" -name "*.txt")

# Loop through each .txt file
for file in $txt_files; do
    # Check if the file is named status.txt
    if [[ $(basename "$file") == "status.txt" ]]; then
        # Check if status.txt contains any of the keywords
        if grep -qE "$(IFS="|"; echo "${keywords[*]}")" "$file"; then
            # Get the directory of the current file
            dir=$(dirname "$file")
            # Check if log.out exists in the same directory
            if [[ -f "$dir/log.out" ]]; then
                # If log_search_string is provided, grep for it in log.out
                if [[ -n "$log_search_string" ]]; then
                    if grep -Fq "$log_search_string" "$dir/log.out"; then
                        echo "Opening $dir/log.out (contains search string)"
                        code "$dir/log.out"
                        ((files_found++))
                    else
                        echo "Search string not found in $dir/log.out"
                    fi
                else
                    echo "Opening $dir/log.out"
                    code "$dir/log.out"
                    ((files_found++))
                fi
            else
                echo "log.out not found in $dir"
            fi
        fi
    fi
done

# Report the number of files found
echo "Total files found and opened: $files_found"