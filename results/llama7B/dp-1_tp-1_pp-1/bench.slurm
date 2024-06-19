#!/bin/bash

#SBATCH --job-name=bench_cluster-12345
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --qos=high
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hopper-prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=11
#SBATCH --output=/fsx/ferdinandmom/ferdinand-hf/bench_cluster/results/llama7B/dp-1_tp-1_pp-1/log-12345.out
#SBATCH --error=/fsx/ferdinandmom/ferdinand-hf/bench_cluster/results/llama7B/dp-1_tp-1_pp-1/log-12345.out

# Misc initializations.
echo "START TIME: $(date)"
source /fsx/ferdinandmom/miniforge3/etc/profile.d/conda.sh
conda activate /fsx/ferdinandmom/miniforge3/envs/env-bench-cluster
echo python3 version = $(python3 --version)

# Slurm stuff
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=6000

export TMPDIR=/scratch
export HF_DATASETS_CACHE="/admin/home/ferdinand_mom/.cache"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_DEVICE_MAX_CONNECTIONS="1"

NANOTRON_REPO="/fsx/ferdinandmom/ferdinand-hf/bench_cluster/nanotron"
CMD="$NANOTRON_REPO/run_train.py --config-file /fsx/ferdinandmom/ferdinand-hf/bench_cluster/results/llama7B/dp-1_tp-1_pp-1/config.yaml"

LAUNCHER="torchrun \
   --nproc_per_node 1 \
   --nnodes 1 \
   --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
   --rdzv_backend c10d \
   --max_restarts 0 \
   --tee 3 \
   --node_rank ${SLURM_PROCID}"

srun -u $LAUNCHER $CMD \
  && echo completed > /fsx/ferdinandmom/ferdinand-hf/bench_cluster/results/llama7B/dp-1_tp-1_pp-1/status.txt || echo failed > /fsx/ferdinandmom/ferdinand-hf/bench_cluster/results/llama7B/dp-1_tp-1_pp-1/status.txt