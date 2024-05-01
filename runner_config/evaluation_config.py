grids = [
    [
        {
            "main_file": ["evaluation.py"],
            "data_dir": [r"word_use_data/childes/word"],
            "pretrained_model": [
                f"ckpt/meta-word_data_dir_word_use_data:childes:word_config_model_config:pythia-160m_concat_False_no_new_token_False_n_examples_{n_examples}_max_sample_times_0_batch_size_8_lr_0.0003_weight_decay_0.07_seed_0/best",
            ],
            "n_examples": [n_examples],
            "eval_n_classes": [tuple(range(2, 11))],
        }
        for n_examples in range(4, 11)
    ],
    [
        {
            "main_file": ["evaluation.py"],
            "data_dir": [r"word_use_data/babylm_data/babylm_10M/word"],
            "pretrained_model": [
                f"ckpt/meta-word_data_dir_word_use_data:babylm_data:babylm_10M:word_config_model_config:pythia-160m_concat_False_no_new_token_False_n_examples_{n_examples}_max_sample_times_0_batch_size_8_lr_0.0003_weight_decay_0.15_seed_0/best",
            ],
            "n_examples": [n_examples],
            "eval_n_classes": [tuple(range(2, 11))],
        }
        for n_examples in range(4, 11)
    ],
    [
        {
            "main_file": ["evaluation.py"],
            "data_dir": [r"word_use_data/childes/word"],
            "tokenizer": ["gpt2"],
            "pretrained_model": ["gpt2"],
            "n_examples": list(range(4, 11)),
            "eval_n_classes": [tuple(range(2, 11))],
        }
    ],
    [
        {
            "main_file": ["evaluation.py"],
            "data_dir": [r"word_use_data/childes/word"],
            "tokenizer": [r"EleutherAI/pythia-160m"],
            "pretrained_model": [r"EleutherAI/pythia-160m"],
            "n_examples": list(range(4, 11)),
            "eval_n_classes": [tuple(range(2, 11))],
        }
    ],
    [
        {
            "main_file": ["evaluation.py"],
            "data_dir": [r"word_use_data/babylm_data/babylm_10M/word"],
            "tokenizer": [r"EleutherAI/pythia-160m"],
            "pretrained_model": [r"EleutherAI/pythia-160m"],
            "n_examples": list(range(4, 11)),
            "eval_n_classes": [tuple(range(2, 11))],
        }
    ],
][1]
# ordered flags to display in job name
flags = [
    "data_dir",
    "pretrained_model",
    "n_examples",
]