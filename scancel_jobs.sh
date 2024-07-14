#!/bin/bash

# Function to print usage
usage() {
    echo "Usage: $0 <job_ids_to_keep>"
    echo "Example: $0 1234 5678 9012"
    exit 1
}

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    usage
fi

# Array to store jobs to keep
keep_jobs=("$@")

# Get all job IDs for the current user, including job arrays and their dependencies
all_jobs=$(squeue -u $USER -h -o "%A,%T,%j,%P")

# Function to check if a job should be kept
should_keep_job() {
    local job_id=$1
    for keep_job in "${keep_jobs[@]}"; do
        if [ "$job_id" = "$keep_job" ]; then
            return 0
        fi
    done
    return 1
}

# Function to process job and its dependencies
process_job() {
    local job_info=$1
    local job_id=$(echo $job_info | cut -d',' -f1)
    local job_state=$(echo $job_info | cut -d',' -f2)
    local job_name=$(echo $job_info | cut -d',' -f3)
    local job_array=$(echo $job_info | cut -d',' -f4)

    if should_keep_job "$job_id"; then
        echo "Keeping job $job_id ($job_name)"
    else
        # Check if it's a job array
        if [[ $job_array == *"_"* ]]; then
            echo "Cancelling job array $job_id ($job_name)"
            scancel "$job_id"
        else
            echo "Cancelling job $job_id ($job_name)"
            scancel "$job_id"
        fi
    fi
}

# Process all jobs
IFS=$'\n'
for job_info in $all_jobs; do
    process_job "$job_info"
done

echo "Job cancellation complete."

# Check for orphaned dependencies and cancel them
orphaned_deps=$(squeue -u $USER -h -o "%A,%T,%j,%P" | grep "PENDING" | grep "Dependency")
if [ ! -z "$orphaned_deps" ]; then
    echo "Cancelling orphaned dependencies:"
    while IFS= read -r dep_job; do
        dep_job_id=$(echo $dep_job | cut -d',' -f1)
        dep_job_name=$(echo $dep_job | cut -d',' -f3)
        echo "Cancelling orphaned dependency $dep_job_id ($dep_job_name)"
        scancel "$dep_job_id"
    done <<< "$orphaned_deps"
fi

echo "Orphaned dependency cancellation complete."