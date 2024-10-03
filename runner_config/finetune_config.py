import os

grids = [
    {
        "main_file": ["main.py"],
        "pretrained_model": [os.environ["SCRATCH"]+r"/Meta-Llama-3-8B"],
        "new_word": ["<|new_word|>", "<|reserved_special_token_0|>"][-1:],
        "embedding_init": ["mean"],
        "prompt": [
            "",
            "<|prompt_0|><|prompt_1|><|prompt_2|><|prompt_3|><|prompt_4|>",
            "<|reserved_special_token_2|><|reserved_special_token_3|><|reserved_special_token_4|><|reserved_special_token_5|><|reserved_special_token_6|>",
            "The following lines are lowercased example sentences using a new word '<|new_word|>' in random order, one per line:",
            "The following lines are lowercased example sentences using a new word '<|reserved_special_token_0|>' in random order, one per line:",
        ][0:1],
        "sep": [" *", "<|sep|>", "<|reserved_special_token_1|>"][-1:],
        #"add_tokens": [("<|new_word|>", "<|sep|>", "<|prompt_0|>", "<|prompt_1|>", "<|prompt_2|>", "<|prompt_3|>", "<|prompt_4|>")],
        "train_params": [("new_word", "sep", "prompt")[:2]],
        "n_epochs": [15],
        "max_sample_times": [0],
        "eval_n_classes": [(4, 6, 8)],
        "loss_reduction": ["sum"],
        "weight_decay": [0.],
        "factor": [0.1],
        "patience": [2],
        "seed": [0, 1, 2],
        "eval_seed": [0],
        "logging_step": [100],
        "eval_step": [1000],
        **c
    }
    for c in [
        {
            "data_dir": ["word_use_data/childes/word"],
            "n_examples": [5],
            "batch_size": [32],
            "lr": [3e-3],
            "train_max_length": [80],
        },
        {
            "data_dir": ["word_use_data/childes/word"],
            "n_examples": [10],
            "batch_size": [8],
            "lr": [3e-4],
            "train_max_length": [160],
        },
        {
            "data_dir": ["word_use_data/babylm_data/babylm_10M/word"],
            "n_examples": [5],
            "batch_size": [16],
            "lr": [1e-3],
            "train_max_length": [160],
        },
    ]
]
# ordered flags to display in job name
flags = [
    "pretrained_model",
    "data_dir",
    "embedding_init",
    #"prompt",
    "train_params",
    "n_examples",
    "train_max_length",
    "batch_size",
    "lr",
    "seed",
    "eval_step",
]