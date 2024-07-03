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

def parse_logs(inp_dir):
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
                match_iteration = re.search(
                    r'\[default\d+\]:\S+ \S+ \[INFO\|DP=\d+\|PP=\d+\|TP=\d+\|\S+\]: iteration: (\d+) / \d+ \| ' \
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
def get_promised_flop_per_sec(device: str, dtype: torch.dtype) -> float:
    """Return the peak FLOP/s for `device` operating on `dtype`."""
    properties = torch.cuda.get_device_properties(device)

    if "A100" in properties.name:
        # they are exponent 12
        # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf")
        if dtype == torch.float32:
            return 19.5
        if dtype in (torch.bfloat16, torch.float16):
            return 312
        raise ValueError(f"Unknown dtype: {dtype}")

    if "H100" in properties.name:
        # https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet")
        # they are exponent 12
        if dtype == torch.float32:
            return 67.5
        if dtype in (torch.bfloat16, torch.float16):
            return 1979 / 2  # 1979 is for sparse, dense is half of that
        raise ValueError(f"Unknown dtype: {dtype}")

    raise ValueError(f"Unknown device: {device}")

def create_global_summary(inp_dir):
    file_paths = glob.glob(os.path.join(inp_dir, "**", "*.csv"), recursive=True)
    if not file_paths:
        raise ValueError(f"No .csv file found in {inp_dir}")

    summary_results_csv = [file for file in file_paths if "summary_results" in file]
    assert len(summary_results_csv) == 1, "There should be exactly one summary_results csv file"
    log_metrics_csv = [file for file in file_paths if "log_metrics" in file]
    profiler_csv = [file for file in file_paths if "profiler" in file]
    
    summary_results_pd = pd.read_csv(summary_results_csv[0])
    summary_results_pd["status"] = summary_results_pd["status"].astype(str)
    summary_results_pd["forward"] = summary_results_pd["forward"].astype(str)
    summary_results_pd["backward"] = summary_results_pd["backward"].astype(str)
    
    log_metrics_dfs = {}
    for file in log_metrics_csv:
        run_name = file.split("/")[-2]
        log_metrics_dfs[run_name] = pd.read_csv(file)

    profiler_dfs = {}
    for file in profiler_csv:
        run_name = file.split("/")[-2]
        profiler_dfs[run_name] = pd.read_csv(file)
    
    for run_name in summary_results_pd["run_name"]:
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
        
        # Tokens per sec per gpu
        summary_results_pd.loc[index, "tok/s/gpu"] = log_metrics_dfs[run_name]["tokens_per_sec_per_gpu"].astype(float).mean() 
        
        # MFU (bf16)
        summary_results_pd.loc[index, "mfu"] = (log_metrics_dfs[run_name]["model_tflops_per_gpu"].astype(int).mean() / get_promised_flop_per_sec(device="cuda", dtype=torch.bfloat16)) * 100
         
        # Forward
        summary_results_pd.loc[index, "forward"] = profiler_dfs[run_name]["forward"].values[0]
        # Backward
        summary_results_pd.loc[index, "backward"] = profiler_dfs[run_name]["backward"].values[0]
    
    summary_results_pd.to_csv(summary_results_csv[0], index=False)
    print(f"Create {summary_results_csv[0]} with new metrics")
    
def report(inp_dir, is_profiler=False, is_network=False, is_logs=False, global_summary=False):
    
    if is_logs:
       parse_logs(inp_dir)
    elif is_profiler:
        parse_profiler(inp_dir)
    elif is_network:
        parse_network(inp_dir)
    elif global_summary:
        create_global_summary(inp_dir) 
    else:
        raise ValueError("Please specify the type of report to generate")