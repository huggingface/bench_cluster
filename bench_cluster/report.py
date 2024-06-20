import glob
import os
import re
import csv
from concurrent.futures import ThreadPoolExecutor


def parse_value(value):
    if 'K' in value:
        return float(value.replace('K', '')) * 1000
    elif 'G' in value:
        return float(value.replace('G', '')) * 1000000000
    else:
        return float(value)

def extract_metrics_from_files(file_paths):
    metrics_dict = {}
    for file_path in file_paths:
        iteration_metrics = []
        memory_metrics = []

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
                    metrics = {
                        'iteration': int(match_iteration.group(1)),
                        'consumed_tokens': parse_value(match_iteration.group(2)),
                        'elapsed_time_per_iteration_ms': parse_value(match_iteration.group(3)),
                        'tokens_per_sec': parse_value(match_iteration.group(4)),
                        'tokens_per_sec_per_gpu': parse_value(match_iteration.group(5)),
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

                if match_memory:
                    memory_metrics.append({
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
    return metrics_dict

def save_metrics_to_csv(metrics_dict):
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
                dict_writer
                dict_writer.writeheader()
                dict_writer.writerows(memory_metrics)

def report(inp_dir):
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

    metrics_dict = extract_metrics_from_files(completed_logs_path)
    save_metrics_to_csv(metrics_dict)
    
    print(f"Saved {len(metrics_dict)} csv files over {len(completed_logs_path)} completed logs")
