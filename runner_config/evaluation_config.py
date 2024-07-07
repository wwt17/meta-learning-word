import os

grids = [
    [
        {
            "main_file": ["evaluation.py"],
            "data_dir": [r"word_use_data/childes/word"],
            "pretrained_model": [
                f"ckpt/meta-word_data_dir_word_use_data:childes:word_config_gpt2_concat_False_context_length_128_no_new_token_False_n_examples_{n_examples}_max_sample_times_0_batch_size_8_lr_0.0001_weight_decay_{weight_decay}_seed_0/best",
            ],
            "n_examples": [n_examples],
            "eval_n_classes": [tuple(range(2, 11))],
            "print_decoded_prefix": [True],
        }
        for n_examples in range(4, 11)
        for weight_decay in [0.05, 0.07, 0.1, 0.15, 0.12][-1:]
    ],
    [
        {
            "main_file": ["evaluation.py"],
            "data_dir": [r"word_use_data/childes/word"],
            "pretrained_model": [
                f"ckpt/meta-word_data_dir_word_use_data:childes:word_config_model_config:pythia-160m_concat_False_no_new_token_False_n_examples_{n_examples}_max_sample_times_0_batch_size_8_lr_0.0003_weight_decay_0.07_seed_0/best",
            ],
            "n_examples": [n_examples],
            "eval_n_classes": [tuple(range(2, 11))],
            "print_decoded_prefix": [True],
        }
        for n_examples in range(4, 11)
    ],
    [
        {
            "main_file": ["evaluation.py"],
            "data_dir": [r"word_use_data/childes/word", r"word_use_data/babylm_data/babylm_10M/word"],
            "pretrained_model": [
                f"ckpt/meta-word_data_dir_word_use_data:babylm_data:babylm_10M:word_config_model_config:pythia-160m_concat_False_no_new_token_False_n_examples_{n_examples}_max_sample_times_0_batch_size_8_lr_0.0003_weight_decay_0.15_seed_0/best",
            ],
            "n_examples": [n_examples],
            "eval_n_classes": [tuple(range(2, 11))],
            "print_decoded_prefix": [True],
        }
        for n_examples in range(4, 11)
    ],
    [
        {
            "main_file": ["evaluation.py"],
            "data_dir": [r"word_use_data/childes/word", r"word_use_data/babylm_data/babylm_10M/word", r"word_use_data/babylm_data/babylm_100M/word"],
            "pretrained_model": [
                f"ckpt/meta-word_data_dir_word_use_data:babylm_data:babylm_100M:word_config_model_config:pythia-160m_concat_False_no_new_token_False_n_examples_{n_examples}_max_sample_times_0_batch_size_8_lr_0.0001_weight_decay_0.07_seed_0/best",
            ],
            "n_examples": [n_examples],
            "eval_n_classes": [tuple(range(2, 11))],
            "print_decoded_prefix": [True],
        }
        for n_examples in range(10, 11)
    ],
    [
        {
            "main_file": ["evaluation.py"],
            "data_dir": [r"word_use_data/childes/word", r"word_use_data/babylm_data/babylm_10M/word", r"word_use_data/babylm_data/babylm_100M/word"][:1],
            "pretrained_model": [f"EleutherAI/pythia-{model_size}" for model_size in ['70m', '160m', '410m']], #["gpt2", os.environ["SCRATCH"]+r"/Meta-Llama-3-8B"],
            "revision": [f"step{step}" for step in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(range(1000, 143001, 1000)) if step <= 4000],
            "n_examples": list(range(10, 11)),
            "eval_n_classes": [tuple(range(2, 11))],
            "print_decoded_prefix": [True],
            "new_word": [" dax"],
            "prompt": ["The following lines are lowercased example sentences using a new word 'dax' in random order, one per line:"],
            "sep": [" *"],
            "prepend": [" "],
        }
    ],
][-1]
# ordered flags to display in job name
flags = [
    "data_dir",
    "pretrained_model",
    "revision",
    "n_examples",
]