import os
import torch
import argparse
import torch.distributed as dist
from bench_cluster.communication.utils import sync_all, print_rank_0, get_bw, get_metric_strings, convert_size, print_header, max_numel, init_torch_distributed

def timed_p2p(input, start_event, end_event, warmups, trials, async_op, bw_unit, raw):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    sync_all()
    # Warmups, establish connections, etc.
    for _ in range(warmups):
        for i in range(world_size):
            if i != rank:
                if async_op:
                    if rank < i:
                        dist.isend(input, i)
                    else:
                        dist.irecv(input, src=i)
                else:
                    if rank < i:
                        dist.send(input, i)
                    else:
                        dist.recv(input, src=i)
    sync_all()

    # time the actual comm op trials times and average it
    start_event.record()
    for _ in range(trials):
        for i in range(world_size):
            if i != rank:
                if async_op:
                    if rank < i:
                        dist.isend(input, i)
                    else:
                        dist.irecv(input, src=i)
                else:
                    if rank < i:
                        dist.send(input, i)
                    else:
                        dist.recv(input, src=i)

    end_event.record()
    sync_all()
    duration = start_event.elapsed_time(end_event) / 1000

    # maintain and clean performance data
    avg_duration = duration / trials
    size = input.element_size() * input.nelement()
    n = world_size
    tput, busbw = get_bw(bw_unit, 'p2p', size * (n - 1), avg_duration)  # Multiply size by (n-1) as each process communicates with all others
    tput_str, busbw_str, duration_str = get_metric_strings(raw, tput, busbw, avg_duration)
    desc = f'{input.nelement()}x{input.element_size()}'

    if not raw:
        size = convert_size(size)

    print_rank_0(f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")

def run_p2p(local_rank, trials, warmups, maxsize, async_op, bw_unit, scan, raw, dtype, mem_factor, debug=False):
    # Prepare benchmark header
    print_header(bw_unit, raw, 'p2p')
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    if scan:
        # Create list of message sizes
        M_LIST = [2**p for p in range(1, maxsize)]

        sync_all()
        # loop over various tensor sizes
        for M in M_LIST:
            try:
                mat = torch.ones(M, dtype=getattr(torch, dtype)).cuda(local_rank)
                sync_all()
                input = mat.mul_(float(global_rank))
                del mat
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print_rank_0('WARNING: Ran out of GPU memory. Exiting comm op.')
                    sync_all()
                    break
                else:
                    raise e
            sync_all()
            timed_p2p(input, start_event, end_event, warmups, trials, async_op, bw_unit, raw)
    else:
        # Send the biggest message size our GPUs can fit. If you're facing OOM errors, reduce the mem_factor
        # Don't need output tensor, so double mem_factor
        elements_per_gpu = max_numel('p2p', getattr(torch, dtype), mem_factor * 2, local_rank)
        try:
            mat = torch.ones(elements_per_gpu, dtype=getattr(torch, dtype)).cuda(local_rank)
            input = mat.mul_(float(global_rank))
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print_rank_0('WARNING: Ran out of GPU memory. Try to reduce the --mem-factor argument!')
                sync_all()
                return
        sync_all()
        timed_p2p(input, start_event, end_event, warmups, trials, async_op, bw_unit, raw)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--maxsize', type=int, default=24)
    parser.add_argument('--async_op', action='store_true')
    parser.add_argument('--bw_unit', type=str, default='Gbps')
    parser.add_argument('--scan', action='store_true')
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--mem_factor', type=float, default=0.1)
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    init_torch_distributed("nccl", local_rank)
    run_p2p(local_rank, args.trials, args.warmups, args.maxsize, args.async_op, args.bw_unit, args.scan, args.raw, args.dtype, args.mem_factor, args.debug)