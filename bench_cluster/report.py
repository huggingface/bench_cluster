import glob
import os
import re
import csv
import json
import pandas as pd
import torch
from statistics import mean

def units_to_float(value):
    if 'K' in value:
        return float(value.replace('K', '')) * 1000
    elif 'M' in value:
        return float(value.replace('M', '')) * 1000000
    elif 'G' in value:
        return float(value.replace('G', '')) * 1000000000
    else:
        return float(value)

def parse_logs(inp_dir, cluster: str):
    folders = [os.path.abspath(folder) for folder in glob.glob(os.path.join(inp_dir, "**"), recursive=True) if os.path.isdir(folder)]
    completed_logs_path = []

    for folder in folders:
        status_file = os.path.join(folder, "status.txt")
        if os.path.exists(status_file):
            with open(status_file, "r") as f:
                status = f.read().strip()
            if status == "completed":
                log_files = glob.glob(os.path.join(folder, "log.out"))
                if log_files:
                    completed_logs_path.append(log_files[0])

    metrics_dict = {}
    for file_path in completed_logs_path:
        metrics = {}
        current_iteration = None

        with open(file_path, 'r') as file:
            for line in file:

                if cluster == "hf":
                    match_iteration = re.search(
                        r'\[default\d+\]:\S+ \S+ \[INFO\|DP=\d+\|PP=\d+\|TP=\d+\|(nid\d+|\S+)\]: iteration: (\d+) / \d+ \| ' \
                        r'consumed_tokens: ([\d\.KM]+) \| elapsed_time_per_iteration_ms: ([\d\.KM]+) \| ' \
                        r'tokens_per_sec: ([\d\.KM]+) \| tokens_per_sec_per_gpu: ([\d\.KM]+) \| ' \
                        r'global_batch_size: ([\d\.KM]+) \| lm_loss: ([\d\.]+) \| lr: ([\de\.-]+) \| ' \
                        r'model_tflops_per_gpu: ([\d\.]+) \| hardware_tflops_per_gpu: ([\d\.]+) \| ' \
                        r'grad_norm: ([\d\.]+).*', line)
                    if match_iteration:
                        current_iteration = int(match_iteration.group(1))
                        metrics[current_iteration] = {
                            'iteration': current_iteration,
                            'consumed_tokens': units_to_float(match_iteration.group(2)),
                            'elapsed_time_per_iteration_ms': units_to_float(match_iteration.group(3)),
                            'tokens_per_sec': units_to_float(match_iteration.group(4)),
                            'tokens_per_sec_per_gpu': units_to_float(match_iteration.group(5)),
                            'global_batch_size': units_to_float(match_iteration.group(6)),
                            'lm_loss': float(match_iteration.group(7)),
                            'lr': float(match_iteration.group(8)),
                            'model_tflops_per_gpu': float(match_iteration.group(9)),
                            'hardware_tflops_per_gpu': float(match_iteration.group(10)),
                            'grad_norm': float(match_iteration.group(11))
                        }

                    match_memory = re.search(
                    r'\[default\d\]:\S+ \S+ \[INFO\|DP=\d\|PP=\d\|TP=\d\|\S+\]:  Memory usage: ([\d\.]+)MiB\. '
                    r'Peak allocated ([\d\.]+)MiB\. Peak reserved: ([\d\.]+)MiB', line)

                    if match_memory and current_iteration is not None:
                        if current_iteration in metrics:
                            metrics[current_iteration].update({
                                'memory_usage_MiB': float(match_memory.group(1)),
                                'peak_allocated_MiB': float(match_memory.group(2)),
                                'peak_reserved_MiB': float(match_memory.group(3))
                            })

                elif cluster == "swiss-ai":
                    match_iteration = re.search(
                        r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) \[INFO\|DP=(\d+)\|PP=(\d+)\|TP=(\d+)\|(nid\d+)\]: '
                        r'iteration: (\d+) / \d+ \| '
                        r'consumed_tokens: ([\d\.KM]+) \| '
                        r'elapsed_time_per_iteration_ms: ([\d\.KM]+) \| '
                        r'tokens_per_sec: ([\d\.KM]+) \| '
                        r'tokens_per_sec_per_gpu: ([\d\.KM]+) \| '
                        r'global_batch_size: ([\d\.KM]+) \| '
                        r'lm_loss: ([\d\.]+) \| '
                        r'lr: ([\de\.-]+) \| '
                        r'model_tflops_per_gpu: ([\d\.]+) \| '
                        r'hardware_tflops_per_gpu: ([\d\.]+) \| '
                        r'grad_norm: ([\d\.]+) \| '
                        r'cuda_memory_allocated: ([\d\.KMG]+) \| '
                        r'cuda_max_memory_reserved: ([\d\.KMG]+) \| '
                        r'hd_total_memory_tb: ([\d\.KMG]+) \| '
                        r'hd_used_memory_tb: ([\d\.KMG]+) \| '
                        r'hd_free_memory_tb: ([\d\.KMG]+)',
                        line
                    )
                    if match_iteration:
                        current_iteration = int(match_iteration.group(6))  # Changed from 1 to 6
                        metrics[current_iteration] = {
                            'iteration': current_iteration,
                            'consumed_tokens': units_to_float(match_iteration.group(7)),
                            'elapsed_time_per_iteration_ms': units_to_float(match_iteration.group(8)),
                            'tokens_per_sec': units_to_float(match_iteration.group(9)),
                            'tokens_per_sec_per_gpu': units_to_float(match_iteration.group(10)),
                            'global_batch_size': units_to_float(match_iteration.group(11)),
                            'lm_loss': float(match_iteration.group(12)),
                            'lr': float(match_iteration.group(13)),
                            'model_tflops_per_gpu': float(match_iteration.group(14)),
                            'hardware_tflops_per_gpu': float(match_iteration.group(15)),
                            'grad_norm': float(match_iteration.group(16)),
                            'cuda_memory_allocated': units_to_float(match_iteration.group(17)),
                            'cuda_max_memory_reserved': units_to_float(match_iteration.group(18)),
                            'hd_total_memory_tb': units_to_float(match_iteration.group(19)),
                            'hd_used_memory_tb': units_to_float(match_iteration.group(20)),
                            'hd_free_memory_tb': units_to_float(match_iteration.group(21))
                        }
                    
                    match_memory = re.search(
                        r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) \[INFO\|DP=(\d+)\|PP=(\d+)\|TP=(\d+)\|(nid\d+)\]: '
                        r'Memory usage: ([\d\.]+)MiB\. '
                        r'Peak allocated ([\d\.]+)MiB\. Peak reserved: ([\d\.]+)MiB',
                        line
                    )

                    if match_memory and current_iteration is not None:
                        if current_iteration in metrics:
                            metrics[current_iteration].update({
                                'memory_usage_MiB': float(match_memory.group(6)),
                                'peak_allocated_MiB': float(match_memory.group(7)),
                                'peak_reserved_MiB': float(match_memory.group(8))
                            })
                            # Optionally, you can also update metadata if needed
                            metadata = {
                                'timestamp': match_memory.group(1),
                                'dp': int(match_memory.group(2)),
                                'pp': int(match_memory.group(3)),
                                'tp': int(match_memory.group(4)),
                                'node_id': match_memory.group(5)
                            }

        metrics_dict[file_path] = list(metrics.values())
        
    # Save metrics to csv files
    for file_path, data in metrics_dict.items():
        base_folder = os.path.dirname(file_path)
        if data:
            csv_path = os.path.join(base_folder,  "log_metrics.csv")

            with open(csv_path, 'w', newline='') as output_file:
                fieldnames = data[0].keys()
                dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                dict_writer.writeheader()
                dict_writer.writerows(data)

    print(f"Saved {len(metrics_dict)} csv files over {len(completed_logs_path)} completed logs")

def parse_profiler(inp_dir):
    # Search for files ending in .json in the inp_dir and its subdirectories
    file_paths = glob.glob(os.path.join(inp_dir, "**", "*.json"), recursive=True)
    if not file_paths:
        raise ValueError(f"No .json file found in {inp_dir}")
    
    all_forward_durations = []
    all_backward_durations = []
    
    def _format_duration(duration):
        ms = duration // 1000
        us = duration % 1000
        return f"{ms}ms {us}μs"
        
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        
        with open(file_path, 'r') as f:
            trace_data = json.load(f)
        
        forward_durations = []
        backward_durations = []
        
        for event in trace_data['traceEvents']:
            if 'name' in event and 'dur' in event:
                if "forward" in event['name'].lower():
                    forward_durations.append(event['dur'])
                elif "backward" in event['name'].lower():
                    backward_durations.append(event['dur'])
        
        if forward_durations:
            all_forward_durations.extend(forward_durations)
        if backward_durations:
            all_backward_durations.extend(backward_durations)
        
        # Write the mean forward and backward durations to a csv file
        pattern = re.compile(r'dp-\d+_tp-\d+_pp-\d+_mbz-\d+')
        matching_index = next((i for i, part in enumerate(file_path.split("/")) if pattern.match(part)), None)
        
        if matching_index is None:
            raise ValueError(f"Could not find the specified pattern in {file_paths[0]}")

        assert matching_index < len(file_path.split("/")) - 1, "Matching index is out of bounds"
        output_file = "/".join(file_path.split("/")[:matching_index + 1]) + "/profiler.csv"
        
        if all_forward_durations or all_backward_durations:
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["forward", "backward"])
                writer.writerow([
                    _format_duration(int(mean(all_forward_durations))) if all_forward_durations else "N/A",
                    _format_duration(int(mean(all_backward_durations))) if all_backward_durations else "N/A"
                ])
            print(f"Results written to {output_file}")
        else:
            print("No forward or backward durations found in any file.")

def parse_network(inp_dir):
    file_paths = glob.glob(os.path.join(inp_dir, "*.out"))
    if not file_paths:
        raise ValueError(f"No log file found in {inp_dir}")
    
    primitives = ['all_gather', 'all_reduce', 'all_to_all', 'broadcast', 'p2p']
    headers = ['Primitive', 'Size (Bytes)', 'Description', 'Duration', 'Throughput (Gbps)', 'BusBW (Gbps)']

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            input_text = file.read()
        
        data = []
        for primitive in primitives:
            pattern = rf"---- Performance of {primitive}.*?Size \(Bytes\).*?(\d+\.?\d*\s+[GMK]?B)\s+(\S+)\s+(\d+\.?\d*\s+ms)\s+(\d+\.?\d*)\s+(\d+\.?\d*)"
            match = re.search(pattern, input_text, re.DOTALL)
            if match:
                size, description, duration, throughput, busbw = match.groups()
                data.append([primitive, size, description, duration, throughput, busbw])

        output_file = os.path.splitext(file_path)[0] + '.csv'
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(data)

        print(f"Data from {file_path} has been written to {output_file}")

# https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_02.py#L919
def get_promised_flop_per_sec(dtype: torch.dtype) -> float:
    """Return the peak FLOP/s for the GPU operating on `dtype`."""
    
    # Run nvidia-smi command and capture output
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                capture_output=True, text=True, check=True)
        gpu_name = result.stdout.strip()
    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to run nvidia-smi. Make sure it's installed and accessible.")
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi command not found. Make sure NVIDIA drivers are installed.")

    # Extract GPU model (they are exponent 12)
    if "A100" in gpu_name:
        if dtype == torch.float32:
            return 19.5  # 19.5 TFLOP/s
        if dtype in (torch.bfloat16, torch.float16):
            return 312   # 312 TFLOP/s
    elif "H100" in gpu_name or "GH200" in gpu_name:
        if dtype == torch.float32:
            return 67.5  # 67.5 TFLOP/s
        if dtype in (torch.bfloat16, torch.float16):
            return (1979 / 2)  # 989.5 TFLOP/s (half of 1979 for dense operations)
    else:
        raise ValueError(f"Unsupported GPU model: {gpu_name}")

    raise ValueError(f"Unknown dtype: {dtype}")

def create_global_summary(inp_dir):
    
    folders_path = glob.glob(os.path.join(inp_dir, '*/'))
    file_paths = glob.glob(os.path.join(inp_dir, "**", "*.csv"), recursive=True)
    if not file_paths:
        raise ValueError(f"No .csv file found in {inp_dir}")

    log_metrics_csv = [file for file in file_paths if "log_metrics" in file]
    profiler_csv = [file for file in file_paths if "profiler" in file]
    
    summary_results_pd = pd.DataFrame(columns=["model", "run_name", "status", "nnodes", "dp", "tp", "pp", "batch_accumulation_per_replica", "micro_batch_size", "tok/s/gpu", "mfu", "forward", "backward"])    
    summary_results_pd["status"] = summary_results_pd["status"].astype(str)
    summary_results_pd["forward"] = summary_results_pd["forward"].astype(str)
    summary_results_pd["backward"] = summary_results_pd["backward"].astype(str)
        
    # Create run_name column in the summary_results_pd with folder_paths
    for folder in folders_path:
        _, model, _, run_name, _ = folder.split("/")
        
        dp, tp, pp, micro_batch_size, batch_accumulation_per_replica = re.findall(r'\d+', run_name)
        dp, tp, pp = int(dp), int(tp), int(pp)
        world_size = dp * tp * pp
        
        summary_results_pd.loc[len(summary_results_pd)] = {
            "model": model,
            "run_name": f"dp-{dp}_tp-{tp}_pp-{pp}_mbz-{micro_batch_size}_bapr-{batch_accumulation_per_replica}",
            "status": str(""),
            "nnodes": max(1, world_size // 8),
            "dp": dp,
            "tp": tp,
            "pp": pp,
            "batch_accumulation_per_replica": batch_accumulation_per_replica,
            "micro_batch_size": micro_batch_size,
            "tok/s/gpu": -1,
            "mfu": -1,
            "memory": -1,
            "forward": str(""),
            "backward": str(""),
        }

    log_metrics_dfs = {}
    for file in log_metrics_csv:
        run_name = file.split("/")[-2]
        log_metrics_dfs[run_name] = pd.read_csv(file)

    profiler_dfs = {}
    for file in profiler_csv:
        run_name = file.split("/")[-2]
        profiler_dfs[run_name] = pd.read_csv(file)
    
    skip_profiling_steps = 7
    
    for run_name in summary_results_pd["run_name"]:
        print(f"Processing {run_name}")
        # Get the associated row in the summary_results csv
        index = summary_results_pd[summary_results_pd["run_name"] == run_name].index[0]
       
        # Status
        status_file = os.path.join(inp_dir, run_name, "status.txt")
        if os.path.exists(status_file):
            with open(status_file, "r") as f:
                status = f.read().strip()
            summary_results_pd.loc[index, "status"] = status

        if summary_results_pd.loc[index, "status"] in ["timeout", "oom", "fail", "pending", "running"]:
            continue
        
        if run_name not in log_metrics_dfs:
            print(f"Skipping {run_name} as it does not have log_metrics.csv")
            continue
        
        # Tokens per sec per gpu (exclude the first 6 iterations as they are part of profiling)
        summary_results_pd.loc[index, "tok/s/gpu"] = log_metrics_dfs[run_name]["tokens_per_sec_per_gpu"][skip_profiling_steps:].astype(float).mean() 
        
        # MFU (bf16) (exclude the first 3 iterations as they are profiler warmup)
        summary_results_pd.loc[index, "mfu"] = (log_metrics_dfs[run_name]["model_tflops_per_gpu"][skip_profiling_steps:].astype(int).mean() / get_promised_flop_per_sec(dtype=torch.bfloat16)) * 100
         
        if run_name not in profiler_dfs:
            print(f"Skipping profiler part for {run_name} as it does not have profiler.csv")
            continue
        
        summary_results_pd.loc[index, "forward"] = profiler_dfs[run_name]["forward"].values[0]
        # Backward
        summary_results_pd.loc[index, "backward"] = profiler_dfs[run_name]["backward"].values[0]

    num_gpus = folders_path[0].split("/")[-3]
    path = os.path.join(inp_dir, num_gpus + "_global_summary.csv")
    summary_results_pd.to_csv(path, index=False)
    print(f"Create {path} with new metrics")
    
def report(inp_dir, cluster, is_profiler=False, is_network=False, is_logs=False, global_summary=False):
    
    if is_logs:
       parse_logs(inp_dir, cluster)
    elif is_profiler:
        parse_profiler(inp_dir)
    elif is_network:
        parse_network(inp_dir)
    elif global_summary:
        create_global_summary(inp_dir) 
    else:
        raise ValueError("Please specify the type of report to generate")