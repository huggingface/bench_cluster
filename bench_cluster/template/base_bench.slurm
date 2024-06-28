#!/bin/bash

#SBATCH --job-name=bench_cluster
#SBATCH --time=00:59:00
#SBATCH --partition=hopper-prod
#SBATCH --nodes={{ nodes }}
#SBATCH --gres=gpu:{{ n_proc_per_node }}
#SBATCH --qos={{ qos }}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --output={{ root_path }}/log.out
#SBATCH --error={{ root_path }}/log.out

# Function to update status based on squeue output
update_status() {
    job_id=$1
    status_file=$2
    # For unknown reasons, it doenst update status for pending. It only works for running 
    while true; do
        job_status=$(squeue --job $job_id --noheader --format=%T)
        echo "Job status: $job_status"
        if [ -z "$job_status" ]; then
            # Job has finished or is not found
            break
        elif [ "$job_status" = "RUNNING" ]; then
            printf "running" > $status_file
            break
        fi
        sleep 10
    done
}

# Misc initializations.
echo "========================"
echo "START TIME: $(date)"
source /fsx/ferdinandmom/miniforge3/etc/profile.d/conda.sh
conda activate /fsx/ferdinandmom/miniforge3/envs/env-bench-cluster
echo python3 version = $(python3 --version)
echo "========================"

# Slurm stuff
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((1024 + RANDOM % 64511))

export TMPDIR=/scratch
export HF_DATASETS_CACHE="/admin/home/ferdinand_mom/.cache"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_DEVICE_MAX_CONNECTIONS="1"

huggingface-cli login --token $HUGGINGFACE_TOKEN


NANOTRON_REPO="/fsx/ferdinandmom/ferdinand-hf/bench_cluster/nanotron"
CMD="$NANOTRON_REPO/run_train.py --config-file {{ config }}"

LAUNCHER="torchrun \
   --nproc_per_node {{ n_proc_per_node }} \
   --nnodes {{ nodes }} \
   --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
   --rdzv_backend c10d \
   --max_restarts 0 \
   --tee 3 \
   --node_rank ${SLURM_PROCID}"

# Checkout the bench_cluster branch
cd $NANOTRON_REPO
git checkout bench_cluster
cd ..
# Get the current job ID
job_id=${SLURM_JOB_ID}

# Update status to "pending" or "running" in the background
update_status $job_id {{ root_path }}/status.txt &

# Run the main command
srun -u $LAUNCHER $CMD
exit_status=$?

# Update status based on the exit status of `srun`
if [ $exit_status -eq 0 ]; then
    printf "completed" > {{ root_path }}/status.txt
else
    if grep -q "OutOfMemoryError" {{ root_path }}/log-.out; then
        printf "oom" > {{ root_path }}/status.txt
    elif grep -q " CUDA error: an illegal memory access" {{ root_path }}/log-.out; then
        printf "oom" > {{ root_path }}/status.txt
    elif grep -q "Timeout at NCCL" {{ root_path }}/log-.out; then
        printf "timeout" > {{ root_path }}/status.txt
    else
        printf "fail" > {{ root_path }}/status.txt
    fi 
fi

# Run the report script if the job completed successfully
if [ $exit_status -eq 0 ]; then
    python /fsx/ferdinandmom/ferdinand-hf/bench_cluster/main.py report --inp_dir {{ root_path }} --is_logs
    python /fsx/ferdinandmom/ferdinand-hf/bench_cluster/main.py report --inp_dir {{ root_path }} --is_profiler
fi


# Push to hub the folder using huggingface_cli
huggingface-cli upload nanotron/bench_cluster {{ root_path }} {{ target_path_hf_hub }} --commit-message "Upload {{ target_path_hf_hub }}"

# Verify the upload
if [ $? -eq 0 ]; then
    echo "Uploading to Huggingface Hub successful"
else
    echo "Failed to upload to Huggingface Hub"
fi