from argparse import ArgumentParser

from bench_cluster.create_configs import create_configs
from bench_cluster.submit_jobs import submit_jobs
from bench_cluster.check_status import check_status

if __name__ == '__main__':
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")
    
    # Create configs
    create_configs_parser = subparsers.add_parser("create_configs")
    create_configs_parser.add_argument("--out_dir", type=str, required=True)
    create_configs_parser.add_argument("--model", type=str, required=True)
    create_configs_parser.add_argument("--gpus", type=int, required=True)

    # Submit jobs
    submit_jobs_parser = subparsers.add_parser("submit_jobs")
    submit_jobs_parser.add_argument("--inp_dir", type=str, required=True)
    submit_jobs_parser.add_argument("--qos", type=str, required=True, choices=["low", "normal", "high", "prod"]) 
    submit_jobs_parser.add_argument("--only_fails", action="store_true", default=False)
    # Check status
    check_status_parser = subparsers.add_parser("check_status")
    check_status_parser.add_argument("--inp_dir", type=str, required=True)
    
    # Parse logs
    parse_logs_parser = subparsers.add_parser("parse_logs")
    
    # Report
    report_parser = subparsers.add_parser("report")
    
    # Plots
    plots_parser = subparsers.add_parser("plots")
    
    args = parser.parse_args()
    
    if args.action == "create_configs":
        create_configs(args.out_dir, args.model, args.gpus)
    elif args.action == "submit_jobs":
        submit_jobs(args.inp_dir, args.qos, only_fails=args.only_fails)
    elif args.action == "check_status":
        check_status(args.inp_dir)
    elif args.action == "parse_logs":
        pass
    elif args.action == "report":
        pass
    elif args.action == "plots":
        pass
    else:
        raise ValueError("Invalid action")