base_config = {
    'general': {
        'project': 'bench_cluster',
        'seed': 42
    },
    'model': {
        'ddp_bucket_cap_mb': 25,
        'dtype': 'bfloat16',
        'init_method': {
            'std': 0.025
        },
        'make_vocab_size_divisible_by': 1,
        'model_config' : {
            'bos_token_id': 1,
            'eos_token_id': 2,
            'hidden_act': 'silu',
            'hidden_size': 2048,
            'initializer_range': 0.02,
            'intermediate_size': 8192,
            'is_llama_config': True,
            'max_position_embeddings': 2048,
            'num_attention_heads': 32,
            'num_hidden_layers': 24,
            'num_key_value_heads': 32,
            'pad_token_id': None,
            'pretraining_tp': 1,
            'rms_norm_eps': 1.0e-05,
            'rope_scaling': None,
            'rope_theta': 10000.0,
            'tie_word_embeddings': True,
            'use_cache': True,
            'vocab_size': 50272
        },
    },
    'optimizer': {
        'accumulate_grad_in_fp32': True,
        'clip_grad': 1.0,
        'learning_rate_scheduler': {
            'learning_rate': 0.0001,
            'lr_decay_style': 'linear',
            'lr_warmup_style': 'linear',
            'lr_warmup_steps': 1,
            'min_decay_lr': 1.0e-05
        },
        'optimizer_factory': {
            'adam_beta1': 0.9,
            'adam_beta2': 0.95,
            'adam_eps': 1.0e-08,
            'name': 'adamW',
            'torch_adam_is_fused': True
        },
        'weight_decay': 0.01,
        'zero_stage': 1
    },
    'parallelism': {
        "dp": 1,
        'expert_parallel_size': 1,
        "pp": 1,
        "pp_engine": "1f1b",
        "tp": 1,
        'tp_linear_async_communication': False,
        'tp_mode': 'REDUCE_SCATTER',
        'recompute_layer': False
    },
    'profiler': {
        'profiler_export_path': None,
    },
    'tokenizer':
    {
        'tokenizer_max_length': None,
        'tokenizer_name_or_path': 'openai-community/gpt2',
        'tokenizer_revision': None
    },
    'data_stages': [
        {
            'name': 'Training Stage',
            'start_training_step': 1,
            'data': {
                'dataset': {
                    'dataset_overwrite_cache': False,
                    'dataset_processing_num_proc_per_process': 64,
                    'hf_dataset_config_name': None,
                    'hf_dataset_or_datasets': 'roneneldan/TinyStories',
                    'hf_dataset_splits': 'train',
                    'text_column_name': 'text'
                },
                'num_loading_workers': 0,
                'seed': 42
            }
        }
    ],
    'lighteval': None,
    'tokens': {
        'train_steps': 20,
        'val_check_interval': -1,
        'batch_accumulation_per_replica': 1,
        'limit_test_batches': 0,
        'limit_val_batches': 0,
        'micro_batch_size': 2,
        'sequence_length': 4096,
    },
    'logging': {
        'iteration_step_info_interval': 1,
        'log_level': 'info',
        'log_level_replica': 'info'
    },
    'checkpoints': {
        'checkpoint_interval': 100000,
        'checkpoints_path': '/dev/null',
        'resume_checkpoint_path': None
    }
}
