# https://github.com/EleutherAI/cookbook/blob/main/benchmarks/communication/run_all.py
import os
import subprocess
from jinja2 import Template

def network_bench(
    out_dir: str,
    gpus: int,
    qos: str,
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
    os.makedirs(out_dir, exist_ok=True)
    
    root_path = os.path.join(out_dir, f"network_bench_{gpus}_GPUS")
    slurm_script = "/fsx/ferdinandmom/ferdinand-hf/bench_cluster/bench_cluster/template/base_network_bench.slurm"
    
    with open(slurm_script, "r") as f:
        base_network_bench_file = f.read()

    base_network_bench_template = Template(base_network_bench_file)

    nodes = max(1, gpus // 8)
    n_proc_per_node = min(8, gpus // nodes)
    assert nodes * n_proc_per_node == gpus
    
    context_bench = {
        'nodes': nodes,
        'n_proc_per_node': n_proc_per_node,
        'qos': qos,
        'root_path': root_path,
        'trials': trials,
        'warmups': warmups,
        'maxsize': maxsize,
        'async_op': async_op,
        'bw_unit': bw_unit,
        'scan': scan,
        'raw': raw,
        'dtype': dtype,
        'mem_factor': mem_factor,
        'debug': debug
    }

    with open(slurm_script, 'w') as file:
        file.write(base_network_bench_template.render(context_bench))

    subprocess.run(["sbatch", slurm_script])
    print(f"Submitted network benchmark job with {gpus} GPUs")