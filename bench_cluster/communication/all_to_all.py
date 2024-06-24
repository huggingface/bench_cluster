import torch
import torch.distributed as dist
from communication.utils import *
from communication.constants import *

def timed_all_to_all(input, output, start_event, end_event, warmups, trials, async_op, bw_unit, raw):
    sync_all()
    # Warmups, establish connections, etc.
    for i in range(warmups):
        dist.all_to_all_single(output, input, async_op=async_op)
    sync_all()

    # time the actual comm op trials times and average it
    start_event.record()
    for i in range(trials):
        dist.all_to_all_single(output, input, async_op=async_op)
    end_event.record()
    sync_all()
    duration = start_event.elapsed_time(end_event) / 1000

    # maintain and clean performance data
    avg_duration = duration / trials
    size = input.element_size() * input.nelement()
    n = dist.get_world_size()
    tput, busbw = get_bw(bw_unit, 'all_to_all', size, avg_duration)
    tput_str, busbw_str, duration_str = get_metric_strings(raw, tput, busbw, avg_duration)
    desc = f'{input.nelement()}x{input.element_size()}'

    if not raw:
        size = convert_size(size)

    print_rank_0(f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")


def run_all_to_all(local_rank, trials, warmups, maxsize, async_op, bw_unit, scan, raw, dtype, mem_factor, debug=False):
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    # Prepare benchmark header
    print_header(bw_unit, raw, 'all_to_all')

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    if scan:
        M_LIST = []
        for x in (2**p for p in range(1, maxsize)):
            M_LIST.append(x)

        sync_all()
        # loop over various tensor sizes
        for M in M_LIST:
            global_rank = dist.get_rank()
            try:
                mat = torch.ones(world_size, M, dtype=getattr(torch, dtype)).cuda(local_rank)
                assert mat.numel() % world_size == 0, f"tensor cannot be divided in {world_size} chunks"
                sync_all()
                input = ((mat.mul_(float(global_rank))).view(-1))
                output = (mat.clone().view(-1))
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print_rank_0('WARNING: Ran out of GPU memory. Exiting comm op.')
                    sync_all()
                    break
                else:
                    raise e
            sync_all()
            timed_all_to_all(input, output, start_event, end_event, warmups, trials, async_op, bw_unit, raw)
    else:
        # Send the biggest message size our GPUs can fit. If you're facing OOM errors, reduce the mem_factor
        elements_per_gpu = max_numel('all_to_all', getattr(torch, dtype), mem_factor, local_rank)
        try:
            mat = torch.ones(elements_per_gpu, dtype=getattr(torch, dtype)).cuda(local_rank)
            assert mat.numel(
            ) % world_size == 0, f"tensor with {mat.numel()} elements cannot be divided in {world_size} chunks"
            input = ((mat.mul_(float(global_rank))).view(-1))
            # Delete original mat to avoid OOM
            del mat
            torch.cuda.empty_cache()
            output = torch.zeros(elements_per_gpu, dtype=getattr(torch, dtype)).cuda(local_rank)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print_rank_0('WARNING: Ran out of GPU memory. Try to reduce the --mem-factor argument!')
                sync_all()
                return
            else:
                raise e
        sync_all()

        if debug:
            for i in range(world_size):
                if i == global_rank:
                    print(f"Before AllToAll Input List at rank {global_rank}: {input}")
                dist.barrier()

        timed_all_to_all(input, output, start_event, end_event, warmups, trials, async_op, bw_unit, raw)

        if debug:
            for i in range(world_size):
                if i == global_rank:
                    print(f"AllToAll Results at rank {global_rank}: {output}")
                dist.barrier()
