import torch
import torch.distributed as dist
from communication.utils import *
from communication.constants import *


def timed_broadcast(input, start_event, end_event, warmups, trials, async_op, bw_unit, raw):
    sync_all()
    # Warmups, establish connections, etc.
    for i in range(warmups):
        dist.broadcast(input, 0, async_op=async_op)
    sync_all()

    # time the actual comm op trials times and average it
    start_event.record()
    for i in range(trials):
        dist.broadcast(input, 0, async_op=async_op)
    end_event.record()
    sync_all()
    duration = start_event.elapsed_time(end_event) / 1000

    # maintain and clean performance data
    avg_duration = duration / trials
    size = input.element_size() * input.nelement()
    n = dist.get_world_size()
    tput, busbw = get_bw(bw_unit, 'broadcast', size, avg_duration)
    tput_str, busbw_str, duration_str = get_metric_strings(raw, tput, busbw, avg_duration)
    desc = f'{input.nelement()}x{input.element_size()}'

    if not raw:
        size = convert_size(size)

    print_rank_0(f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")


def run_broadcast(local_rank, trials, warmups, maxsize, async_op, bw_unit, scan, raw, dtype, mem_factor, debug=False):
    # Prepare benchmark header
    print_header(bw_unit, raw, 'broadcast')

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

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
                sync_all()
                input = ((mat.mul_(float(global_rank))).view(-1))
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
            timed_broadcast(input, start_event, end_event, warmups, trials, async_op, bw_unit, raw)
    else:
        # Send the biggest message size our GPUs can fit. If you're facing OOM errors, reduce the mem_factor
        # Don't need output tensor, so we double mem_factor
        elements_per_gpu = max_numel('broadcast', getattr(torch, dtype), mem_factor * 2, local_rank)
        try:
            mat = torch.ones(elements_per_gpu, dtype=getattr(torch, dtype)).cuda(local_rank)
            input = ((mat.mul_(float(global_rank))).view(-1))
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print_rank_0('WARNING: Ran out of GPU memory. Try to reduce the --mem-factor argument!')
                sync_all()
                return
        sync_all()
        timed_broadcast(input, start_event, end_event, warmups, trials, async_op, bw_unit, raw)