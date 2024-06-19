from copy import deepcopy
from typing import List
from bench_cluster.template.base_config import base_config
import itertools
import yaml
import os
from tqdm import tqdm
from transformers import AutoTokenizer

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
    
    #TODO(fmom): add support for
    # 1) Seqlen
    # 2) micro batch size
    # 3) deepspeed zero stage 1

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
    
    config_content = deepcopy(base_config)
    update_config_based_on_model(model, config_content)

    #TODO(fmom): Do I need to bound tp < 8 ? (to limit to a single node)
    # Initialize tqdm progress bar for the combinations loop
    for (dp, tp, pp) in tqdm(combinations_3D_parallelism, desc="Creating configs", unit="config"):

        config_content['parallelism']['dp'] = dp
        config_content['parallelism']['tp'] = tp
        config_content['parallelism']['pp'] = pp
        
        run_path = os.path.join(path, f"dp-{dp}_tp-{tp}_pp-{pp}")
        if not os.path.exists(run_path):
            os.makedirs(run_path)
            with open(os.path.join(run_path, "config.yaml"), "w") as new_config:
                yaml.dump(config_content, new_config, default_flow_style=False, sort_keys=False)
    
    del config_content