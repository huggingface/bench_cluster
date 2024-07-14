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
files_found=0

# Loop through each status.txt file
for file in $txt_files; do
    if [[ $(basename "$file") == "status.txt" ]]; then
        if grep -qE "$(IFS="|"; echo "${keywords[*]}")" "$file"; then
            dir=$(dirname "$file")
            log_files=("$dir"/log_*.out)
            if [ ${#log_files[@]} -gt 0 ]; then
                for log_file in "${log_files[@]}"; do
                    if [ -f "$log_file" ]; then
                        if [[ -n "$log_search_string" ]]; then
                            if grep -Fq "$log_search_string" "$log_file"; then
                                echo "Opening $log_file (contains search string)"
                                ((files_found++))
                            fi
                        else
                            echo "Opening $log_file"
                            ((files_found++))
                        fi
                    fi
                done
            else
                echo "No log_*.out files found in $dir"
            fi
        fi
    fi
done

# Report the number of files found
echo "Total files found and opened: $files_found"