import glob
import os
import re
import csv
import json
from statistics import mean

def units_to_float(value):
    if 'K' in value:
        return float(value.replace('K', '')) * 1000
    elif 'G' in value:
        return float(value.replace('G', '')) * 1000000000
    else:
        return float(value)

def parse_logs(inp_dir):
    #TODO(fmom): fuse memory csv file to iteration csv file    
    folders = [os.path.abspath(folder) for folder in glob.glob(os.path.join(inp_dir, "**"), recursive=True) if os.path.isdir(folder)]

    completed_logs_path = []

    for folder in folders:
        status_file = os.path.join(folder, "status.txt")
        if os.path.exists(status_file):
            with open(status_file, "r") as f:
                status = f.read().strip()
            if status == "completed":
                log_files = glob.glob(os.path.join(folder, "log-*.out"))
                if log_files:
                    completed_logs_path.append(log_files[0])

    # Extract metrics from log files
    metrics_dict = {}
    for file_path in completed_logs_path:
        iteration_metrics = []
        memory_metrics = []
        current_iteration = None

        with open(file_path, 'r') as file:
            for line in file:
                # Match the pattern for iterations
                match_iteration = re.search(
                    r'\[default\d\]:\S+ \S+ \[INFO\|DP=\d\|PP=\d\|TP=\d\|\S+\]: iteration: (\d+) / \d+ \| '
                    r'consumed_tokens: ([\d\.K]+) \| elapsed_time_per_iteration_ms: ([\d\.K]+) \| '
                    r'tokens_per_sec: ([\d\.K]+) \| tokens_per_sec_per_gpu: ([\d\.K]+) \| '
                    r'global_batch_size: (\d+) \| lm_loss: ([\d\.]+) \| lr: ([\de\.-]+) \| '
                    r'model_tflops_per_gpu: ([\d\.]+) \| hardware_tflops_per_gpu: ([\d\.]+) \| '
                    r'grad_norm: ([\d\.]+)', line)
                
                if match_iteration:
                    current_iteration = int(match_iteration.group(1))
                    metrics = {
                        'iteration': current_iteration,
                        'consumed_tokens': units_to_float(match_iteration.group(2)),
                        'elapsed_time_per_iteration_ms': units_to_float(match_iteration.group(3)),
                        'tokens_per_sec': units_to_float(match_iteration.group(4)),
                        'tokens_per_sec_per_gpu': units_to_float(match_iteration.group(5)),
                        'global_batch_size': int(match_iteration.group(6)),
                        'lm_loss': float(match_iteration.group(7)),
                        'lr': float(match_iteration.group(8)),
                        'model_tflops_per_gpu': float(match_iteration.group(9)),
                        'hardware_tflops_per_gpu': float(match_iteration.group(10)),
                        'grad_norm': float(match_iteration.group(11))
                    }
                    iteration_metrics.append(metrics)

                # Match the pattern for memory usage
                match_memory = re.search(
                    r'\[default\d\]:\S+ \S+ \[INFO\|DP=\d\|PP=\d\|TP=\d\|\S+\]:  Memory usage: ([\d\.]+)MiB\. '
                    r'Peak allocated ([\d\.]+)MiB\. Peak reserved: ([\d\.]+)MiB', line)

                if match_memory and current_iteration is not None:
                    memory_metrics.append({
                        'iteration': current_iteration,
                        'memory_usage_MiB': float(match_memory.group(1)),
                        'peak_allocated_MiB': float(match_memory.group(2)),
                        'peak_reserved_MiB': float(match_memory.group(3))
                    })

        # Combine iteration and memory metrics, if any
        combined_metrics = {
            'iterations': iteration_metrics,
            'memory': memory_metrics
        }
        metrics_dict[file_path] = combined_metrics
        
    # Save metrics to csv files
    for file_path, data in metrics_dict.items():
        base_folder = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        base_name_without_ext = os.path.splitext(base_name)[0]

        # Save iteration metrics
        if data['iterations']:
            iteration_metrics = data['iterations']
            iteration_keys = iteration_metrics[0].keys()
            csv_file_name = f"{base_name_without_ext}_iterations.csv"
            csv_path = os.path.join(base_folder, csv_file_name)

            with open(csv_path, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=iteration_keys)
                dict_writer.writeheader()
                dict_writer.writerows(iteration_metrics)

        # Save memory metrics
        if data['memory']:
            memory_metrics = data['memory']
            memory_keys = memory_metrics[0].keys()
            csv_file_name = f"{base_name_without_ext}_memory.csv"
            csv_path = os.path.join(base_folder, csv_file_name)

            with open(csv_path, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=memory_keys)
                dict_writer.writeheader()
                dict_writer.writerows(memory_metrics)

    print(f"Saved {len(metrics_dict)} csv files over {len(completed_logs_path)} completed logs")

def parse_profiler(inp_dir):
    # /fsx/ferdinandmom/ferdinand-hf/bench_cluster/results/llama-1B/8_GPUS/dp-1_tp-2_pp-2_mbz-256/20240624-095924/ip-26-0-163-220_1136603.1719223186534915479.pt.trace.json    
    
    # Search for file finishing in .json in the inp_dir
    file_path = glob.glob(os.path.join(inp_dir, "*.json"))
    if not file_path:
        raise ValueError(f"No .json file found in {inp_dir}")
    
    with open(file_path, 'r') as f:
        trace_data = json.load(f)
    
    forward_durations = []
    backward_durations = []
    
    for event in trace_data['traceEvents']:
        if 'name' in event and 'dur' in event:
            if event['name'] == "nanotron/parallel/pipeline_parallel/engine.py(26): forward":
                forward_durations.append(event['dur'])
            elif event['name'] == "nanotron/parallel/pipeline_parallel/engine.py(67): backward":
                backward_durations.append(event['dur'])
    
    def _format_duration(duration):
        ms = duration // 1000
        us = duration % 1000
        return f"{ms}ms {us}μs"
    
    # Go back one folder
    prev_inp_dir = os.path.dirname(inp_dir)
    
    with open(os.path.join(prev_inp_dir, "profiler_results.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["forward", "backward"])
        writer.writerow([_format_duration(int(mean(forward_durations))), _format_duration(int(mean(backward_durations)))])
    

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

def write_csv(data, filename='performance_data.csv'):
    headers = ['Primitive', 'Size (Bytes)', 'Description', 'Duration', 'Throughput (Gbps)', 'BusBW (Gbps)']
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data)

    print(f"Data has been written to {filename}")

def report(inp_dir, is_profiler=False, is_network=False, is_logs=False):
    
    if is_logs:
       parse_logs(inp_dir)
    elif is_profiler:
        parse_profiler(inp_dir)
    elif is_network:
        parse_network(inp_dir)
    else:
        raise ValueError("Please specify the type of report to generate")