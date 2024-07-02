#!/bin/bash

# Initialize counters
declare -A counts
statuses=("init" "pending" "running" "fail" "oom" "timeout" "completed")
for status in "${statuses[@]}"; do
    counts[$status]=0
done

# Find and process all status.txt files
while IFS= read -r -d '' file; do
    status=$(cat "$file" | tr -d '[:space:]')
    if [[ " ${statuses[@]} " =~ " ${status} " ]]; then
        ((counts[$status]++))
    fi
done < <(find "$1" -name "status.txt" -print0)

# Calculate total
total=0
for count in "${counts[@]}"; do
    ((total += count))
done

# Print the results
echo "Status     | Count"
echo "-----------|---------"
for status in "${statuses[@]}"; do
    printf "%-10s | %d\n" "$status" "${counts[$status]}"
done
echo "-----------|---------"
echo "Total      | $total"