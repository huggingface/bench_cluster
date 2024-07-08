#!/bin/bash
#SBATCH --job-name=healthcheck-jobs-10min # job name
#SBATCH --partition=hopper-dev
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --time=0:10:00
#SBATCH --output=%x.out

# Will cancel jobs that were not properly cancel by slurm (to avoid wasting ressources)

# ensure to restart self first
next_run=$(date -d "+10 minutes" +"%Y-%m-%dT%H:%M:%S")
sbatch --begin="$next_run" healthcheck_jobs.slurm

# Check and cancel jobs with SIGTERM in logs, only for your jobs
running_jobs=$(squeue -h -t RUNNING -u $USER -o "%i")

for job_id in $running_jobs; do
    # Get the log file path
    log_path=$(scontrol show job $job_id | grep StdOut | awk -F= '{print $2}')
    
    # Check if log file exists and contains SIGTERM
    if [ -f "$log_path" ] && grep -q "SIGTERM" "$log_path"; then
        # Check if job is still running
        if squeue -h -j $job_id &>/dev/null; then
            echo "Job $job_id has SIGTERM in log but is still running. Cancelling..."
            
            # Get the directory of the log file
            log_dir=$(dirname "$log_path")
            
            # Path to the status.txt file
            status_file="$log_dir/status.txt"
            
            # Cancel the job
            scancel $job_id
            
            # Mark the status.txt file as fail
            printf "fail" > "$status_file"
            
            echo "Job $job_id cancelled and status marked as fail in $status_file"
        fi
    fi
done