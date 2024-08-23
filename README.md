# bench_cluster

- TODO: git submodule for specific nanotron branch
```
pip install -e .
pip install -r requirements.txt
cd nanotron # Checkout bench_cluster branch
pip install -e .
pip install flash_attn==2.5.0
cd ..
```

### Workflow

```
results/
    - network_bench/
        - network_bench_8_gpus.slurm
        - log_8_gpus.out
        - ...
        - network_bench_512_gpus.slurm
    - llama-1B/
        - 8_GPUS/
            - 8_GPUS_summary_results.csv
            - dp-1_tp-8_pp-1_mbz-1/
                - profiler/*.json
                - bench.slurm
                - config.yaml
                - log_metrics.csv
                - log.out
                - profiler.csv
                - status.txt
            ...
            - dp-8_tp-1_pp-1_mbz-256/
        ...
        - 512_GPUS/
    ...
    - llama-7B/
```

### Usage

```shell
# Create single config
python main.py create_single_config --out_dir tmp --model llama-1B --gpus 8 --exp_name draft --no_profiler --cluster "hf" --mbs=1 --bapr=128  --dp=8 --tp=1 --pp=1        

# Create above workflow with all possible combinations of hyper-parameters 
python main.py create_configs --out_dir "results" --model llama-1B --gpus 8

# Create configs without profiler on Swiss cluster
python main.py create_configs --out_dir "results" --model llama-1B --gpus 4 --exp_name 4_GPUS_no_profiler --no_profiler  --cluster swiss-ai

# Create above workflow with all possible combinations and name it 8_GPUS_FOLDER + disable profiler
python main.py create_configs --out_dir "results" --model llama-1B --gpus 8 --exp_name 8_GPUS_FOLDER --no_profiler

# Create above workflow with only combinations of DP 
python main.py create_configs --out_dir "results" --model llama-1B --gpus 8 --tp_max=1  --pp_max=1

# Create configs witt global batch size ranging from 0M to 4M tokens. Include config that increase every 1M tokens as well
python main.py create_configs --out_dir "results"--model llama-1B --gpus 8 --gbs_range "[0M, 4M, 1M]"

# Launch all the jobs in `results/` folder 
python main.py submit_jobs --inp_dir results/  --qos high --hf_token <YOUR_HF_TOKEN> 

# Can as well batch jobs into 4 dependencies array 
python main.py submit_jobs --inp_dir results/ --qos high --hf_token <YOUR_HF_TOKEN> --nb_slurm_array 4

# Check status of runs (INIT/PENDING/RUNNING/FAIL/OOM/COMPLETED)
./check_status.sh results/

# Will cancel jobs that were not properly cancel by slurm (to avoid wasting ressources)
sbatch healthcheck_jobs.slurm

# Automatically rerun the jobs with status FAIL
python main.py submit_jobs --inp_dir results/  --qos high --hf_token <YOUR_HF_TOKEN> --only_fails

# Bench intra/inter-connect of gpus
python main.py network_bench --out_dir results/ --qos=high --gpus=8

# Extract into CSV logs, network and profiler info (NOTE: this is automatically done when using `submit_jobs`)
python main.py report --inp_dir results/ [--is_logs | --is_network | --is_profiler]

# Create a global summary CSV file based on all exisiting csv runs file
python main.py report --inp_dir results/  --global_summary
```