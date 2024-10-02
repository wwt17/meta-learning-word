grids = [
    {
        "main_file": ["main.py"],
        "config": ["model_config/pythia-160m"],
        "lm": [True],
        "concat": [False],
        "no_new_token": [False],
        "freq_cutoff": [4],
        "n_epochs": [30],
        "batch_size": [8],
        "n_examples": [5, 10],
        "max_sample_times": [0],
        "eval_n_classes": [(4, 6, 8)],
        "loss_reduction": ["sum"],
        "lr": [3e-4],
        "factor": [0.1],
        "patience": [2],
        "seed": [0, 1, 2],
        "eval_seed": [0],
        "logging_step": [100],
        **c,
    }
    for c in [
        {
            "data_dir": ["word_use_data/childes/word"],
            "weight_decay": [0.07],
        },
        {
            "data_dir": ["word_use_data/babylm_data/babylm_10M/word"],
            "weight_decay": [0.15],
        },
    ]
]
# ordered flags to display in job name
flags = [
    "data_dir",
    "config",
    "concat",
    "no_new_token",
    "n_examples",
    "max_sample_times",
    "batch_size",
    "lr",
    "weight_decay",
    "seed",
]