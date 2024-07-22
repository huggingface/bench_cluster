from copy import deepcopy
from typing import List
import numpy as np
from bench_cluster.template.base_config import base_config
import itertools
import yaml
import os
from transformers import AutoTokenizer
import math
import pandas as pd

def find_combinations_within_global_batch_size_range(dp, seq_len, min_global_batch_size, max_global_batch_size, step, bapr_max: None):
    def round_to_next_multiple_of(multiple, pair_list):        
        round_up = lambda n: math.ceil(n / multiple) * multiple
        res = []
        for a, b in pair_list:
            new_a = round_up(a) if a != 1 else a
            new_b = round_up(b) if b != 1 else b
            res.append((new_a, new_b))
        return res

    combinations = []
    
    for i in range(min_global_batch_size, max_global_batch_size + 1, step):
        remaining_global_batch_size = i // (dp * seq_len)
        all_pairs = [(a, b) for a, b in itertools.product(range(1, remaining_global_batch_size + 1), repeat=2) if a * b == remaining_global_batch_size]
        all_pairs = round_to_next_multiple_of(multiple=2, pair_list=all_pairs)
        
        for bapr, mbs in all_pairs:
            if bapr_max is not None and bapr > bapr_max:
                continue
            current_global_batch_size = dp * seq_len * bapr * mbs
            # Include as well the case where the current_global_batch size similar to max_global_batch size
            if current_global_batch_size >= min_global_batch_size and current_global_batch_size <= max_global_batch_size + step:
                combinations.append((bapr, mbs))

    return combinations

def update_config_based_on_model(model: str, config: dict):
    
    # Setting num_attention_heads = num_key_value_heads for all models <=> using MHA for all layers
    
    if model == "llama-1B":
        # HuggingFaceFW/ablation-model-fineweb-v1
        config["model"]["model_config"]["hidden_size"] = 2048
        config["model"]["model_config"]["intermediate_size"] = 4096
        config["model"]["model_config"]["num_attention_heads"] = 32
        config["model"]["model_config"]["num_hidden_layers"] = 24
        config["model"]["model_config"]["num_key_value_heads"] = 32
        config["model"]["model_config"]["max_position_embeddings"] = config["tokens"]["sequence_length"]
    elif model == "llama-7B":
        # meta-llama/Llama-2-7b-hf
        config["model"]["model_config"]["hidden_size"] = 4096
        config["model"]["model_config"]["intermediate_size"] = 11008
        config["model"]["model_config"]["num_attention_heads"] = 32
        config["model"]["model_config"]["num_hidden_layers"] = 32
        config["model"]["model_config"]["num_key_value_heads"] = 32
        config["model"]["model_config"]["max_position_embeddings"] = config["tokens"]["sequence_length"]
    elif model == "llama-70B":
        # meta-llama/Llama-2-70b-hf
        config["model"]["model_config"]["hidden_size"] = 8192
        config["model"]["model_config"]["intermediate_size"] = 28672
        config["model"]["model_config"]["num_key_value_heads"] = 64
        config["model"]["model_config"]["num_hidden_layers"] = 80
        config["model"]["model_config"]["num_key_value_heads"] = 64
        config["model"]["model_config"]["max_position_embeddings"] = config["tokens"]["sequence_length"]
    elif model == "llama-340B":
        # nvidia/Nemotron-4-340B-Base
        config["model"]["model_config"]["hidden_size"] = 18432
        config["model"]["model_config"]["intermediate_size"] = 73728
        config["model"]["model_config"]["num_attention_heads"] = 96
        config["model"]["model_config"]["num_hidden_layers"] = 96
        config["model"]["model_config"]["num_key_value_heads"] = 96
        config["model"]["model_config"]["max_position_embeddings"] = config["tokens"]["sequence_length"]
    elif model == "llama-400B":
        config["model"]["model_config"]["hidden_size"] = 16384
        config["model"]["model_config"]["intermediate_size"] = 1.2 *  config["model"]["model_config"]["hidden_size"]
        config["model"]["model_config"]["num_attention_heads"] = 128
        config["model"]["model_config"]["num_hidden_layers"] = 126
        config["model"]["model_config"]["num_key_value_heads"] = 128
        config["model"]["model_config"]["max_position_embeddings"] = config["tokens"]["sequence_length"]
    else:
        raise ValueError(f"Model {model} is not supported")  

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["tokenizer_name_or_path"])
    config["model"]["model_config"]["vocab_size"] = tokenizer.vocab_size

def is_enough_layers_for_pp(pp_size, config):
    
    def _get_block_compute_costs(config):
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        model_config = config["model"]["model_config"]
        d_ff = model_config["intermediate_size"]
        d_qkv = model_config["hidden_size"] // model_config["num_attention_heads"]
        
        block_compute_costs = {
            # This is the last lm_head
            "lm_head": model_config["vocab_size"] * model_config["hidden_size"],
        }
        for i in range(model_config["num_hidden_layers"]):
            # CausalSelfAttention (qkv proj + attn out) + MLP
            block_compute_costs[f"decoder{i}"] = 4 * model_config["num_attention_heads"] * d_qkv * model_config["hidden_size"] + 3 * d_ff * model_config["hidden_size"]

        return block_compute_costs

    # compute PP block repartition
    block_compute_costs = _get_block_compute_costs(config)
    num_layers = config["model"]["model_config"]["num_hidden_layers"]
    pipeline_blocks = ["token_embedding"] + [f"decoder{i}" for i in range(num_layers)] + ["final_layer_norm", "lm_head", "cast_to_fp32", "loss"]
    block_cumulative_costs = np.cumsum(
        [
            block_compute_costs[name] if name in block_compute_costs else 0
            for name in pipeline_blocks
        ]
    )
    
    # Assign ranks to blocks
    block2rank = {block: 0 for block in pipeline_blocks}
    target_pp_ranks = list(range(pp_size))
    thresholds = [block_cumulative_costs[-1] * ((rank + 1) / pp_size) for rank in range(pp_size)]
    assert thresholds[-1] >= block_cumulative_costs[-1]
    target_pp_rank_idx = 0
    
    for block, cumulative_cost in zip(pipeline_blocks, block_cumulative_costs):
        assert target_pp_rank_idx < pp_size
        block2rank[block] = target_pp_ranks[target_pp_rank_idx]
        
        if cumulative_cost > thresholds[target_pp_rank_idx]:
            target_pp_rank_idx += 1

    block2rank["token_embedding"] = target_pp_ranks[0]
    block2rank["loss"] = target_pp_ranks[target_pp_rank_idx]
    
    # Check if all ranks have a block assigned to it
    unique_ranks = sorted(set(block2rank.values()))
    expected_ranks = list(range(pp_size))

    return unique_ranks == expected_ranks

def create_configs(
    out_dir: str,
    model: str,
    gpus: int,
    dp_max: int,
    tp_max: int,
    pp_max: int,
    bapr_max: int,
    gbs_range: tuple[int],
    no_profiler: bool = False,
    cluster: str = "hf",
    exp_name: str = None,
    seq_len: int = 4096,
    recompute_layer: bool = False,
):
    print(f"Creating configs for {model} given {gpus} GPUs")
    
    config_content = deepcopy(base_config)
    config_content["tokens"]["sequence_length"] = seq_len
    config_content["parallelism"]["recompute_layer"] = recompute_layer
    update_config_based_on_model(model, config_content)
    
    if cluster == "hf":
        tp_max_cluster = 8
    elif cluster == "swiss-ai":
        tp_max_cluster = 4 # GH200

    # Generate all possible combinations of three numbers from 1 to gpus
    combinations_3D_parallelism = set()
    dp_range = range(1, gpus + 1) if dp_max is None else range(1, min(dp_max, gpus) + 1)
    tp_range = range(1, tp_max_cluster + 1) if tp_max is None else range(1, min(tp_max, tp_max_cluster) + 1)  # tp <= 8
    pp_range = range(1, gpus + 1) if pp_max is None else range(1, min(pp_max, gpus) + 1)

    # Generate combinations
    for dp in dp_range:
        for tp in tp_range:
            for pp in pp_range:
                if dp * tp * pp == gpus and is_enough_layers_for_pp(pp, config_content):
                    combinations_3D_parallelism.add((dp, tp, pp))

    # Create directories and write config files
    if exp_name is not None:
        path = os.path.join(out_dir, model + f"/{exp_name}")
    else:
        path = os.path.join(out_dir, model + f"/{gpus}_GPUS")
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    min_global_batch_size, max_global_batch_size, step = gbs_range

    count = 0
    # Initialize tqdm progress bar for the combinations loop
    for (dp, tp, pp) in combinations_3D_parallelism:

        config_content['parallelism']['dp'] = dp
        config_content['parallelism']['tp'] = tp
        config_content['parallelism']['pp'] = pp
        
        bapr_mbs_combo = find_combinations_within_global_batch_size_range(dp, config_content["tokens"]["sequence_length"], min_global_batch_size, max_global_batch_size, step, bapr_max)
        
        for (batch_accumulation_per_replica, micro_batch_size) in bapr_mbs_combo:
            
            if batch_accumulation_per_replica < pp - 1:
                # self.n_micro_batches_per_batch = self.config.tokens.batch_accumulation_per_replica
                # self.pipeline_engine.nb_microbatches = self.n_micro_batches_per_batch
                #NOTE: assert self.nb_microbatches >= pg.size() - 1
                continue
                
            # Compute global batch_size and print
            gbs = dp * micro_batch_size * batch_accumulation_per_replica * seq_len
            # Print in human readable format
            print(f"Global batch size : {gbs:,}")
            
            config_content['tokens']['batch_accumulation_per_replica'] = batch_accumulation_per_replica
            config_content['tokens']['micro_batch_size'] = micro_batch_size
            
            # Create a directory for each combination of parallelism
            run_path = os.path.join(path, f"dp-{dp}_tp-{tp}_pp-{pp}_mbz-{micro_batch_size}_bapr-{batch_accumulation_per_replica}")
            if recompute_layer:
                run_path += "_recompute_layer"
            
            # Get absoulte path for run_path
            if no_profiler:
                config_content['profiler'] = None
            else:
                config_content['profiler']['profiler_export_path'] = os.path.abspath(run_path)
            
            if not os.path.exists(run_path):
                os.makedirs(run_path)
                with open(os.path.join(run_path, "config.yaml"), "w") as new_config:
                    yaml.dump(config_content, new_config, default_flow_style=False, sort_keys=False)
                
                count += 1
    print(f"Total number of configs created: {count}")
    # check if file exists
    del config_content