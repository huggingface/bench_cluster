# https://github.com/EleutherAI/cookbook/blob/main/benchmarks/communication/run_all.py
import os
from bench_cluster.communication.utils import *
from bench_cluster.communication.all_reduce import run_all_reduce
from bench_cluster.communication.all_gather import run_all_gather
from bench_cluster.communication.all_to_all import run_all_to_all
from bench_cluster.communication.p2p import run_p2p
from bench_cluster.communication.broadcast import run_broadcast

def network_bench(
    gpus: int,
    trials: int,
    warmups: int,
    maxsize: int,
    async_op: bool,
    bw_unit: str,
    scan: bool,
    raw: bool,
    dtype: str,
    mem_factor: float,
    debug: bool = False,
):
    local_rank = int(os.environ["LOCAL_RANK"])
    init_torch_distributed(backend='nccl', local_rank=local_rank)

    ops_to_run = ['all_gather', 'all_reduce', 'all_to_all', 'broadcast', 'p2p']

    #NOTE(fmom): If you receive SIGTERM signal, lower the mem-factor
    for comm_op in ops_to_run:
        if comm_op == 'all_gather':
            run_all_gather(local_rank, trials, warmups, maxsize, async_op, bw_unit, scan, raw, dtype, mem_factor, debug)
        if comm_op == 'all_reduce':
            run_all_reduce(local_rank, trials, warmups, maxsize, async_op, bw_unit, scan, raw, dtype, mem_factor, debug)
        if comm_op == 'all_to_all':
            run_all_to_all(local_rank, trials, warmups, maxsize, async_op, bw_unit, scan, raw, dtype, mem_factor, debug)
        if comm_op == 'p2p':
            run_p2p(local_rank, trials, warmups, maxsize, async_op, bw_unit, scan, raw, dtype, mem_factor, debug)
        if comm_op == 'broadcast':
            run_broadcast(local_rank, trials, warmups, maxsize, async_op, bw_unit, scan, raw, dtype, mem_factor, debug)