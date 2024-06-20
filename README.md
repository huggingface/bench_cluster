# bench_cluster

```
pip install -e .
pip install -r requirements.txt
cd nanotron
pip install -e .
pip install flash_attn==2.5.0
cd ..
```

### Workflow

> - Pattern for folder naming `run-config-<model>-tp-*-pp-*-dp-*-seqlen-*-mbz-*-actrecomp-*`
```
results/
    - report.csv
    - nanotron.slurm
    - run-config-tp-*-pp-*.../
        - ckpts/*
        - config.yaml
        - log.out
        - parsed_log.csv
        - status.txt
        - plots/*.png
    ...
    - run-config-tp-*-pp-*.../
        - ....
```

### Usage

- Create all configs files by combining all the hyperparameters

```
python main.py create_configs -out_dir "results" --model llama7B --gpus 4
```
- Submit jobs
> - If `status.txt` doesnt exist, launch the job 
```
python main.py --submit_jobs --inp_dir="results/"
``` 
- Relaunch jobs based on status
> 1) Check those who has failed status
> 2) Resume jobs from last checkpoints
> 3) Append to existing log file

```
python main.py submit_jobs --inp_dir "results/" --qos high --hf_token=<HF_TOKEN>
# python main.py --submit_jobs --only_fails --inp_dir="results/" --qos high --hf_token=<HF_TOKEN>
```
- Check status of jobs
> Check status of all jobs and show stats [INIT/PENDING/COMPLETED/FAILS/RUNNING] 

```
python main.py --check_status --inp_dir="results/"
```   

- Create reporting
> - Create `parsed_log.csv` for files that has status=COMPLETED only (default)
> - Create a csv at top level folder with all infos of every runs like this: ![image](https://hackmd.io/_uploads/B13lzQkIC.png) 

```
python main.py --report --out_dir="results/"
```

- Create plots based on parsed logs
> - if results.csv exists, create plots

```
python main.py --plots
```
