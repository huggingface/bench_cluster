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
#SBATCH --output={{ root_path }}/log_{{ nodes * n_proc_per_node }}_gpus.out
#SBATCH --error={{ root_path }}/log_{{ nodes * n_proc_per_node }}_gpus.out

source /fsx/ferdinandmom/miniforge3/etc/profile.d/conda.sh
conda activate /fsx/ferdinandmom/miniforge3/envs/env-bench-cluster

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_DEVICE_MAX_CONNECTIONS="1"

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

LAUNCHER="torchrun \
   --nproc_per_node {{ n_proc_per_node }} \
   --nnodes {{ nodes }} \
   --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
   --rdzv_backend c10d \
   --max_restarts 0 \
   --tee 3 \
   --node_rank ${SLURM_PROCID}"

cd /fsx/ferdinandmom/ferdinand-hf/bench_cluster/bench_cluster

# Run all_gather
$LAUNCHER /fsx/ferdinandmom/ferdinand-hf/bench_cluster/bench_cluster/communication/all_gather.py --trials {{ trials }}  --warmups {{ warmups }} --maxsize {{ maxsize }} --bw_unit {{ bw_unit }} --dtype {{ dtype }} --mem_factor {{ mem_factor }}
# Run all_reduce
$LAUNCHER /fsx/ferdinandmom/ferdinand-hf/bench_cluster/bench_cluster/communication/all_reduce.py --trials {{ trials }}  --warmups {{ warmups }} --maxsize {{ maxsize }} --bw_unit {{ bw_unit }} --dtype {{ dtype }} --mem_factor {{ mem_factor }}
# Run all_to_all
$LAUNCHER /fsx/ferdinandmom/ferdinand-hf/bench_cluster/bench_cluster/communication/all_to_all.py --trials {{ trials }}  --warmups {{ warmups }} --maxsize {{ maxsize }} --bw_unit {{ bw_unit }} --dtype {{ dtype }} --mem_factor {{ mem_factor }}
# Run broadcast
$LAUNCHER /fsx/ferdinandmom/ferdinand-hf/bench_cluster/bench_cluster/communication/broadcast.py --trials {{ trials }}  --warmups {{ warmups }} --maxsize {{ maxsize }} --bw_unit {{ bw_unit }} --dtype {{ dtype }} --mem_factor {{ mem_factor }}
# Run p2p
$LAUNCHER /fsx/ferdinandmom/ferdinand-hf/bench_cluster/bench_cluster/communication/p2p.py --trials {{ trials }}  --warmups {{ warmups }} --maxsize {{ maxsize }} --bw_unit {{ bw_unit }} --dtype {{ dtype }} --mem_factor {{ mem_factor }}