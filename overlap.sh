#!/bin/bash

# Script name: overlap.sh
# Usage: ./overlap.sh [custom_command]
# Example: ./overlap.sh bash

# Default command is 'watch -n 1 nvidia-smi'
CMD=${1:-"watch -n 1 nvidia-smi"}

# Fetch the full list of jobs and format it, including the entire squeue line
mapfile -t jobs < <(squeue | grep "$USER" | sed 's/^\s*//')

# Check if there are any jobs
if [[ ${#jobs[@]} -eq 0 ]]; then
    echo "No jobs found. Exiting."
    exit 1
fi

# Create a menu for job selection
echo "Select a job:"
select job_selection in "${jobs[@]}"; do
    if [[ -n "$job_selection" ]]; then
        break
    else
        echo "Invalid selection. Try again."
    fi
done

# Extract the job ID and node name from the selected line
job_id=$(echo $job_selection | awk '{print $1}')
node_id=$(echo $job_selection | awk '{print $NF}')  # Extracts the last field

# Construct the srun command and execute it
srun_cmd="srun --overlap --pty --jobid=$job_id -w $node_id $CMD"
echo "Running command: $srun_cmd"
eval $srun_cmd
