# ========= SEQLEN 4096 ======

# Dp only experiments
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 4 --exp_name 4_GPUS_DP_ONLY_no_profiler --no_profiler --pp_max 1 --tp_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 8 --exp_name 8_GPUS_DP_ONLY_no_profiler --no_profiler --pp_max 1 --tp_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 16 --exp_name 16_GPUS_DP_ONLY_no_profiler --no_profiler --pp_max 1 --tp_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 32 --exp_name 32_GPUS_DP_ONLY_no_profiler --no_profiler --pp_max 1 --tp_max 1 --cluster "swiss-ai"

python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 4 --exp_name 4_GPUS_DP_ONLY_no_profiler --no_profiler --pp_max 1 --tp_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 8 --exp_name 8_GPUS_DP_ONLY_no_profiler --no_profiler --pp_max 1 --tp_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 16 --exp_name 16_GPUS_DP_ONLY_no_profiler --no_profiler --pp_max 1 --tp_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 32 --exp_name 32_GPUS_DP_ONLY_no_profiler --no_profiler --pp_max 1 --tp_max 1 --cluster "swiss-ai"

# DP only with bapr=1
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 4 --exp_name 4_GPUS_DP_ONLY_no_profiler --no_profiler --pp_max 1 --tp_max 1 --bapr_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 8 --exp_name 8_GPUS_DP_ONLY_no_profiler --no_profiler --pp_max 1 --tp_max 1 --bapr_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 16 --exp_name 16_GPUS_DP_ONLY_no_profiler --no_profiler --pp_max 1 --tp_max 1 --bapr_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 32 --exp_name 32_GPUS_DP_ONLY_no_profiler --no_profiler --pp_max 1 --tp_max 1 --bapr_max 1 --cluster "swiss-ai"

python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 4 --exp_name 4_GPUS_DP_ONLY_no_profiler --no_profiler --pp_max 1 --tp_max 1 --bapr_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 8 --exp_name 8_GPUS_DP_ONLY_no_profiler --no_profiler --pp_max 1 --tp_max 1 --bapr_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 16 --exp_name 16_GPUS_DP_ONLY_no_profiler --no_profiler --pp_max 1 --tp_max 1 --bapr_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 32 --exp_name 32_GPUS_DP_ONLY_no_profiler --no_profiler --pp_max 1 --tp_max 1 --bapr_max 1 --cluster "swiss-ai"

# DP + TP
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 4 --exp_name 4_GPUS_DP_TP_no_profiler --no_profiler --pp_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 8 --exp_name 8_GPUS_DP_TP_no_profiler --no_profiler --pp_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 16 --exp_name 16_GPUS_DP_TP_no_profiler --no_profiler --pp_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 32 --exp_name 32_GPUS_DP_TP_no_profiler --no_profiler --pp_max 1 --cluster "swiss-ai"

python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 4 --exp_name 4_GPUS_DP_TP_no_profiler --no_profiler --pp_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 8 --exp_name 8_GPUS_DP_TP_no_profiler --no_profiler --pp_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 16 --exp_name 16_GPUS_DP_TP_no_profiler --no_profiler --pp_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 32 --exp_name 32_GPUS_DP_TP_no_profiler --no_profiler --pp_max 1 --cluster "swiss-ai"

# DP + TP with bapr=1
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 4 --exp_name 4_GPUS_DP_TP_no_profiler --no_profiler --pp_max 1 --bapr_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 8 --exp_name 8_GPUS_DP_TP_no_profiler --no_profiler --pp_max 1 --bapr_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 16 --exp_name 16_GPUS_DP_TP_no_profiler --no_profiler --pp_max 1 --bapr_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 32 --exp_name 32_GPUS_DP_TP_no_profiler --no_profiler --pp_max 1 --bapr_max 1 --cluster "swiss-ai"

python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 4 --exp_name 4_GPUS_DP_TP_no_profiler --no_profiler --pp_max 1 --bapr_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 8 --exp_name 8_GPUS_DP_TP_no_profiler --no_profiler --pp_max 1 --bapr_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 16 --exp_name 16_GPUS_DP_TP_no_profiler --no_profiler --pp_max 1 --bapr_max 1 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 32 --exp_name 32_GPUS_DP_TP_no_profiler --no_profiler --pp_max 1 --bapr_max 1 --cluster "swiss-ai"

# ========= SEQLEN 2048 ======
# Dp only experiments
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 4 --exp_name 4_GPUS_DP_ONLY_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --tp_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 8 --exp_name 8_GPUS_DP_ONLY_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --tp_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 16 --exp_name 16_GPUS_DP_ONLY_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --tp_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 32 --exp_name 32_GPUS_DP_ONLY_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --tp_max 1 --seq_len 2048 --cluster "swiss-ai"

python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 4 --exp_name 4_GPUS_DP_ONLY_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --tp_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 8 --exp_name 8_GPUS_DP_ONLY_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --tp_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 16 --exp_name 16_GPUS_DP_ONLY_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --tp_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 32 --exp_name 32_GPUS_DP_ONLY_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --tp_max 1 --seq_len 2048 --cluster "swiss-ai"

# DP only with bapr=1
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 4 --exp_name 4_GPUS_DP_ONLY_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --tp_max 1 --bapr_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 8 --exp_name 8_GPUS_DP_ONLY_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --tp_max 1 --bapr_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 16 --exp_name 16_GPUS_DP_ONLY_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --tp_max 1 --bapr_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 32 --exp_name 32_GPUS_DP_ONLY_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --tp_max 1 --bapr_max 1 --seq_len 2048 --cluster "swiss-ai"

python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 4 --exp_name 4_GPUS_DP_ONLY_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --tp_max 1 --bapr_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 8 --exp_name 8_GPUS_DP_ONLY_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --tp_max 1 --bapr_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 16 --exp_name 16_GPUS_DP_ONLY_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --tp_max 1 --bapr_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 32 --exp_name 32_GPUS_DP_ONLY_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --tp_max 1 --bapr_max 1 --seq_len 2048 --cluster "swiss-ai"

# DP + TP
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 4 --exp_name 4_GPUS_DP_TP_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 8 --exp_name 8_GPUS_DP_TP_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 16 --exp_name 16_GPUS_DP_TP_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 32 --exp_name 32_GPUS_DP_TP_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --seq_len 2048 --cluster "swiss-ai"

python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 4 --exp_name 4_GPUS_DP_TP_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 8 --exp_name 8_GPUS_DP_TP_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 16 --exp_name 16_GPUS_DP_TP_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 32 --exp_name 32_GPUS_DP_TP_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --seq_len 2048 --cluster "swiss-ai"

# DP + TP with bapr=1
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 4 --exp_name 4_GPUS_DP_TP_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --bapr_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 8 --exp_name 8_GPUS_DP_TP_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --bapr_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 16 --exp_name 16_GPUS_DP_TP_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --bapr_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-1B --gpus 32 --exp_name 32_GPUS_DP_TP_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --bapr_max 1 --seq_len 2048 --cluster "swiss-ai"

python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 4 --exp_name 4_GPUS_DP_TP_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --bapr_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 8 --exp_name 8_GPUS_DP_TP_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --bapr_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 16 --exp_name 16_GPUS_DP_TP_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --bapr_max 1 --seq_len 2048 --cluster "swiss-ai"
python main.py create_configs --out_dir /capstor/scratch/cscs/fmom/new-local-results-epfl --model llama-7B --gpus 32 --exp_name 32_GPUS_DP_TP_SEQLEN_2048_no_profiler --no_profiler --pp_max 1 --bapr_max 1 --seq_len 2048 --cluster "swiss-ai"