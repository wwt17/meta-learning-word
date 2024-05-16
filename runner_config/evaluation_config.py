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
            "tokenizer": [r"word_use_data/babylm_data/babylm_10M/word/tokenizer"],
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
            "tokenizer": [r"word_use_data/babylm_data/babylm_100M/word/tokenizer"],
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
            "data_dir": [r"word_use_data/childes/word", r"word_use_data/babylm_data/babylm_10M/word", r"word_use_data/babylm_data/babylm_100M/word"],
            "tokenizer": [pretrained_model],
            "pretrained_model": [pretrained_model],
            "n_examples": list(range(2, 11)),
            "eval_n_classes": [tuple(range(2, 11))],
            "print_decoded_prefix": [True],
        }
        for pretrained_model in ["gpt2", r"EleutherAI/pythia-160m"]
    ],
][-1]
# ordered flags to display in job name
flags = [
    "data_dir",
    "pretrained_model",
    "n_examples",
]