from copy import deepcopy
from typing import List
from bench_cluster.template.base_config import base_config
import itertools
import yaml
import os
from tqdm import tqdm
from transformers import AutoTokenizer
import math

def find_combinations_power_of_2(x):
    assert (x != 0) and (x & (x - 1) == 0)
    
    combinations = []
    # Iterate through powers of 2 up to x
    for exponent in range(int(math.log2(x)) + 1):
        micro_batch_size = 2 ** exponent
        if x % micro_batch_size == 0:
            batch_accumulation_per_replica = x // micro_batch_size
            combinations.append((batch_accumulation_per_replica, micro_batch_size))
    
    return combinations

def update_config_based_on_model(model: str, config: dict):
    if model == "llama-1B":
        config["model"]["model_config"]["hidden_size"] = 2048
        config["model"]["model_config"]["intermediate_size"] = 8192
        config["model"]["model_config"]["num_attention_heads"] = 32
        config["model"]["model_config"]["num_hidden_layers"] = 24
        config["model"]["model_config"]["num_key_value_heads"] = 32
    elif model == "llama-7B":
        config["model"]["model_config"]["hidden_size"] = 4096
        config["model"]["model_config"]["intermediate_size"] = 11008
        config["model"]["model_config"]["num_attention_heads"] = 32
        config["model"]["model_config"]["num_hidden_layers"] = 32
        config["model"]["model_config"]["num_key_value_heads"] = 32
    else:
        raise ValueError(f"Model {model} is not supported")  

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["tokenizer_name_or_path"])
    config["model"]["model_config"]["vocab_size"] = tokenizer.vocab_size
    config["model"]["model_config"]["max_position_embeddings"] = config["tokens"]["sequence_length"]

def create_configs(out_dir: str, model: str, gpus: int):
    print(f"Creating configs for {model} given {gpus} GPUs")
    
    # Generate all possible combinations of three numbers from 1 to gpus
    combinations_3D_parallelism = set()    
    for combination in itertools.product(range(1, gpus+1), repeat=3):
        dp, tp, pp = combination
        if dp * tp * pp <= gpus and tp <= 8:
            # Add all permutations of the combination
            for perm in itertools.permutations(combination):
                combinations_3D_parallelism.add(perm)

    # Create directories and write config files
    path = os.path.join(out_dir, model)
    if not os.path.exists(path):
        os.makedirs(path)
    
    config_content = deepcopy(base_config)
    update_config_based_on_model(model, config_content)
    
    min_global_batch_size, max_global_batch_size = 4*1e6, 8*1e6

    # Initialize tqdm progress bar for the combinations loop
    for (dp, tp, pp) in tqdm(combinations_3D_parallelism, desc="Creating configs", unit="config"):

        config_content['parallelism']['dp'] = dp
        config_content['parallelism']['tp'] = tp
        config_content['parallelism']['pp'] = pp
        
        # GBZ = batch_accumulation_per_replica * micro_batch_size * dp.size() * seqlen
        remaining_global_batch_size = int(max_global_batch_size // (dp * config_content["tokens"]["sequence_length"]))
        # Find the largest power of 2 that is less than or equal to remaining_global_batch_size
        remaining_global_batch_size = 2 ** (remaining_global_batch_size.bit_length() - 1)
        for (batch_accumulation_per_replica, micro_batch_size) in find_combinations_power_of_2(remaining_global_batch_size):
            
            if batch_accumulation_per_replica * micro_batch_size * dp * config_content["tokens"]["sequence_length"] < min_global_batch_size:
                continue
            
            config_content['parallelism']['batch_accumulation_per_replica'] = batch_accumulation_per_replica
            config_content['parallelism']['micro_batch_size'] = micro_batch_size
            
            # Create a directory for each combination of parallelism
            run_path = os.path.join(path, f"dp-{dp}_tp-{tp}_pp-{pp}_mbz-{micro_batch_size}")
            if not os.path.exists(run_path):
                os.makedirs(run_path)
                with open(os.path.join(run_path, "config.yaml"), "w") as new_config:
                    yaml.dump(config_content, new_config, default_flow_style=False, sort_keys=False)
    
    del config_content