#!/bin/bash

#SBATCH --job-name=network_bench_cluster
#SBATCH --time=00:05:00
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --qos=high
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --output=results/network_bench_8_GPUS/log-.out
#SBATCH --error=results/network_bench_8_GPUS/log-.out

source /fsx/ferdinandmom/miniforge3/etc/profile.d/conda.sh
conda activate /fsx/ferdinandmom/miniforge3/envs/env-bench-cluster

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_DEVICE_MAX_CONNECTIONS="1"

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

LAUNCHER="torchrun \
   --nproc_per_node 8 \
   --nnodes 1 \
   --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
   --rdzv_backend c10d \
   --max_restarts 0 \
   --tee 3"

CMD="/fsx/ferdinandmom/ferdinand-hf/bench_cluster/bench_cluster/network_bench.py \
    --trials 50 \
    --warmups 5 \
    --maxsize 24 \
     \
    --bw_unit Gbps \
     \
     \
    --dtype float \
    --mem_factor 0.1 \
    "

srun -u $LAUNCHER $CMD