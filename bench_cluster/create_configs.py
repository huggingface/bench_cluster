from typing import List
from bench_cluster.template.base_config import base_config
import itertools
import yaml
import os
from tqdm import tqdm

def create_configs(out_dir: str, model: str, gpus: int):
    print(f"Creating configs for {model} given {gpus} GPUs")
    
    #TODO(fmom): add support for seqlen and micro batch size
    #TODO(fmom): add support for models
    
    # Generate all possible combinations of three numbers from 1 to gpus
    combinations_3D_parallelism = set()    
    for combination in itertools.product(range(1, gpus+1), repeat=3):
        if combination[0] * combination[1] * combination[2] == gpus:
            # Add all permutations of the combination
            for perm in itertools.permutations(combination):
                combinations_3D_parallelism.add(perm)
    
    # Create directories and write config files
    path = os.path.join(out_dir, model)
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Initialize tqdm progress bar for the combinations loop
    for (dp, tp, pp) in tqdm(combinations_3D_parallelism, desc="Creating configs", unit="config"):
        config = base_config.copy()

        config['parallelism']['dp'] = dp
        config['parallelism']['tp'] = tp
        config['parallelism']['pp'] = pp
        
        run_path = os.path.join(path, f"dp-{dp}_tp-{tp}_pp-{pp}")
        if not os.path.exists(run_path):
            os.makedirs(run_path)
            with open(os.path.join(run_path, "config.yaml"), "w") as file:
                yaml.dump(config, file, default_flow_style=False, sort_keys=False)

        del config