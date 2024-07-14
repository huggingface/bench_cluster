import argparse
from argparse import ArgumentParser

from bench_cluster.create_configs import create_configs
from bench_cluster.submit_jobs import submit_jobs
from bench_cluster.network_bench import network_bench
from bench_cluster.report import report
from bench_cluster.communication.constants import DEFAULT_TRIALS, DEFAULT_WARMUPS, DEFAULT_UNIT, DEFAULT_TYPE

def parse_range(range_str):
    def parse_value(value):
        value = value.strip()
        if value.endswith('M'):
            return int(value[:-1]) * 1_000_000
        elif value.endswith('K'):
            return int(value[:-1]) * 1_000
        else:
            raise ValueError("Unit for range not supported")

    try:
        # Remove brackets and split the string
        values = range_str.strip('[]').split(',')
        
        if len(values) != 3:
            raise ValueError("Range must have exactly 3 values")

        start = parse_value(values[0])
        end = parse_value(values[1])
        step = parse_value(values[2])
        
        return start, end, step
    except (ValueError, IndexError) as e:
        raise argparse.ArgumentTypeError(f"Invalid range format. Use '[start, end, step]'. Error: {str(e)}")

if __name__ == '__main__':
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")
    
    # Create configs
    create_configs_parser = subparsers.add_parser("create_configs")
    create_configs_parser.add_argument("--out_dir", type=str, required=True)
    create_configs_parser.add_argument("--model", type=str, required=True)
    create_configs_parser.add_argument("--gpus", type=int, required=True, choices=[4, 8, 16, 32, 64, 128, 256, 512])
    create_configs_parser.add_argument("--exp_name", type=str, default=None)
    create_configs_parser.add_argument("--no_profiler", action="store_true")
    create_configs_parser.add_argument("--cluster", type=str, default="hf", choices=["hf", "swiss-ai"])
    create_configs_parser.add_argument("--dp_max", type=int, default=None)
    create_configs_parser.add_argument("--tp_max", type=int, default=None)
    create_configs_parser.add_argument("--pp_max", type=int, default=None)
    create_configs_parser.add_argument("--bapr_max", type=int, default=None, help="Set maximum batch_accumulation_per_replica.")
    create_configs_parser.add_argument("--gbs_range", type=parse_range, default="[4M, 8M, 1M]", help='Specify range as "[start, end, step]". In example, [4M, 8M, 1M] -> go from 4M to 8M and increase by 1M every step.')
    create_configs_parser.add_argument("--seq_len", type=int, default=4096, choices=[2048, 4096])
    create_configs_parser.add_argument("--recompute_layer", action="store_true", default=False, help="Recompute each Transformer layer.")    
    
    # Submit jobs
    submit_jobs_parser = subparsers.add_parser("submit_jobs")
    submit_jobs_parser.add_argument("--inp_dir", type=str, required=True)
    submit_jobs_parser.add_argument("--qos", type=str, required=True, choices=["low", "normal", "high", "prod"]) 
    submit_jobs_parser.add_argument("--only", type=str, default=None, choices=["fail", "pending", "timeout", "running"])
    submit_jobs_parser.add_argument("--hf_token", type=str, required=True)
    submit_jobs_parser.add_argument("--nb_slurm_array", type=int, default=0)
    submit_jobs_parser.add_argument("--cluster", type=str, default="hf", choices=["hf", "swiss-ai"])
    
    #  Network bench
    network_bench_parser = subparsers.add_parser("network_bench")
    network_bench_parser.add_argument("--out_dir", type=str, required=True)
    network_bench_parser.add_argument("--gpus", type=int, required=True, choices=[8, 16, 32, 64, 128, 256, 512])
    network_bench_parser.add_argument("--qos", type=str, required=True, choices=["low", "normal", "high", "prod"])
    network_bench_parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help='Number of timed iterations')
    network_bench_parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS, help='Number of warmup (non-timed) iterations')
    network_bench_parser.add_argument("--maxsize", type=int, default=24, help='Max message size as a power of 2')
    network_bench_parser.add_argument("--async-op", action="store_true", help='Enables non-blocking communication')
    network_bench_parser.add_argument("--bw_unit", type=str, default=DEFAULT_UNIT, choices=['Gbps', 'GBps'])
    network_bench_parser.add_argument("--scan", action="store_true", help='Enables scanning all message sizes')
    network_bench_parser.add_argument("--raw", action="store_true", help='Print the message size and latency without units')
    network_bench_parser.add_argument("--dtype", type=str, default=DEFAULT_TYPE, help='PyTorch tensor dtype')
    network_bench_parser.add_argument("--mem_factor", type=float, default=.1, help='Proportion of max available GPU memory to use for single-size evals')
    network_bench_parser.add_argument("--debug", action="store_true", help='Enables all_to_all debug prints')
    
    # Report
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--inp_dir", type=str, required=True)  
    report_parser.add_argument("--is_profiler", action="store_true", default=False)
    report_parser.add_argument("--is_network", action="store_true", default=False)  
    report_parser.add_argument("--is_logs", action="store_true", default=False)
    report_parser.add_argument("--global_summary", action="store_true", default=False)
    report_parser.add_argument("--cluster", type=str, default="hf", choices=["hf", "swiss-ai"])

    # Plots
    plots_parser = subparsers.add_parser("plots")
    
    args = parser.parse_args()
    
    if args.action == "create_configs":
        create_configs(args.out_dir, args.model, args.gpus, args.dp_max, args.tp_max, args.pp_max, args.bapr_max, args.gbs_range, args.no_profiler, args.cluster, args.exp_name, args.seq_len, args.recompute_layer)
    elif args.action == "submit_jobs":
        submit_jobs(args.inp_dir, args.qos, args.hf_token, args.nb_slurm_array, cluster=args.cluster, only=args.only)
    elif args.action == "network_bench":
        #TODO: take into account boolean into scripts
        network_bench(args.out_dir, args.gpus, args.qos, args.trials, args.warmups, args.maxsize, args.async_op, args.bw_unit, args.scan, args.raw, args.dtype, args.mem_factor, args.debug)
    elif args.action == "report":
        report(args.inp_dir, args.cluster, args.is_profiler, args.is_network, args.is_logs, args.global_summary)
    elif args.action == "plots":
        pass
    else:
        raise ValueError("Invalid action")
