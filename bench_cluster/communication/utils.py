import torch
import os, sys
import math
import torch.distributed as dist
from .constants import *

def print_rank_0(message):
    if dist.get_rank() == 0:
        print(message)

def env2int(env_list, default=-1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0: return val
    return default

def init_torch_distributed(backend: str, local_rank: int):
    dist.init_process_group(backend, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
    torch.cuda.set_device(local_rank)
    print_rank_0(f"Initializing distributed backend: {backend}")
    print_rank_0(f"RANK: {os.environ['RANK']}")
    print_rank_0(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}")
   

def print_header(bw_unit, raw, comm_op):
    world_size = dist.get_world_size()
    tput = f'Throughput ({bw_unit})'
    busbw = f'BusBW ({bw_unit})'
    header = f"\n---- Performance of {comm_op} on {world_size} devices ---------------------------------------------------------\n"
    duration_str = 'Duration'
    if raw:
        duration_str += ' (us)'
    header += f"{'Size (Bytes)':20s} {'Description':25s} {duration_str:20s} {tput:20s} {busbw:20s}\n"
    header += "----------------------------------------------------------------------------------------------------"
    print_rank_0(header)


def get_bw(bw_unit, comm_op, size, duration):
    n = dist.get_world_size()
    tput = 0
    busbw = 0
    if comm_op == "all_to_all":
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_gather":
        size *= n
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_reduce":
        tput = (size * 2 / duration)
        busbw = (size / duration) * (2 * (n - 1) / n)
    elif comm_op == "p2p" or comm_op == "broadcast":
        tput = (size / duration)
        busbw = tput
    else:
        print_rank_0("wrong comm_op specified")
        exit(0)

    if bw_unit == 'Gbps':
        tput *= 8
        busbw *= 8

    return tput, busbw


def get_metric_strings(raw, tput, busbw, duration):
    duration_ms = duration * 1e3
    duration_us = duration * 1e6
    tput = f'{tput / 1e9:.3f}'
    busbw = f'{busbw /1e9:.3f}'

    if duration_us < 1e3 or raw:
        duration = f'{duration_us:.3f}'
        if not raw:
            duration += ' us'
    else:
        duration = f'{duration_ms:.3f} ms'
    return tput, busbw, duration


def sync_all():
    torch.cuda.synchronize()
    dist.barrier()


def max_numel(comm_op, dtype, mem_factor, local_rank):
    dtype_size = _element_size(dtype)
    max_memory_per_gpu = torch.cuda.get_device_properties(local_rank).total_memory * mem_factor
    if comm_op == 'all_reduce' or comm_op == 'p2p' or comm_op == 'broadcast':
        elements_per_gpu = int(max_memory_per_gpu // dtype_size)
    elif comm_op == 'all_gather':
        # all_gather performance is lower for non-powers of two, and the output buffer size scales with world size
        # Therefore, divide by world size and round down to nearest power of 2
        elements_per_gpu = int(max_memory_per_gpu // dtype_size // dist.get_world_size())
        elements_per_gpu = int(pow(2, int(math.log(elements_per_gpu, 2))))
    elif comm_op == 'all_to_all':
        # Number of elements must be divisible by world_size
        # all_to_all performance is lower for non-powers of two. Round down like all_gather.
        elements_per_gpu = int(max_memory_per_gpu // dtype_size)
        elements_per_gpu = int(dist.get_world_size() * round(elements_per_gpu / dist.get_world_size()))
        elements_per_gpu = int(pow(2, int(math.log(elements_per_gpu, 2))))
    else:
        print(f"This communication operation: {comm_op} is not supported yet")
        exit(0)
    return elements_per_gpu


# Helper function to pretty-print message sizes
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


# Copied from torch. Need to add the func here for old torch compatibility.
def _element_size(dtype):
    """
    Returns the element size for a dtype, in bytes
    """
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f'expected torch.dtype, but got {type(dtype)}')

    if dtype.is_complex:
        return torch.finfo(dtype).bits >> 2
    elif dtype.is_floating_point:
        return torch.finfo(dtype).bits >> 3
    elif dtype == torch.bool:
        # NOTE: torch.bool is not supported in torch.iinfo()
        return 1
    else:
        return torch.iinfo(dtype).bits >> 3
