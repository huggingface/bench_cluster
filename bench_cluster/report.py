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

        with open(file_path, 'r') as file:
            for line in file:
                # Match the pattern for iterations
                match = re.search(r'\[default\d\]:\S+ \S+ \[INFO\|DP=\d\|PP=\d\|TP=\d\|\S+\]: iteration: (\d+) / \d+ \| '
                                  r'consumed_tokens: ([\d\.K]+) \| elapsed_time_per_iteration_ms: ([\d\.K]+) \| '
                                  r'tokens_per_sec: ([\d\.K]+) \| tokens_per_sec_per_gpu: ([\d\.K]+) \| '
                                  r'global_batch_size: (\d+) \| lm_loss: ([\d\.]+) \| lr: ([\de\.-]+) \| '
                                  r'model_tflops_per_gpu: ([\d\.]+) \| hardware_tflops_per_gpu: ([\d\.]+) \| '
                                  r'grad_norm: ([\d\.]+)', line)
                if match:
                    metrics = {
                        'iteration': int(match.group(1)),
                        'consumed_tokens': parse_value(match.group(2)),
                        'elapsed_time_per_iteration_ms': parse_value(match.group(3)),
                        'tokens_per_sec': parse_value(match.group(4)),
                        'tokens_per_sec_per_gpu': parse_value(match.group(5)),
                        'global_batch_size': int(match.group(6)),
                        'lm_loss': float(match.group(7)),
                        'lr': float(match.group(8)),
                        'model_tflops_per_gpu': float(match.group(9)),
                        'hardware_tflops_per_gpu': float(match.group(10)),
                        'grad_norm': float(match.group(11))
                    }
                    iteration_metrics.append(metrics)

        metrics_dict[file_path] = iteration_metrics
    return metrics_dict

def save_metrics_to_csv(metrics_dict):
    for file_path, metrics in metrics_dict.items():
        if metrics:
            # Get the list of keys from the first dictionary (assuming all dicts have the same keys)
            keys = metrics[0].keys()
            base_folder = os.path.dirname(file_path)
            base_name = os.path.basename(file_path)
            csv_file_name = f"{os.path.splitext(base_name)[0]}.csv"
            csv_path = os.path.join(base_folder, csv_file_name)

            with open(csv_path, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(metrics)

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