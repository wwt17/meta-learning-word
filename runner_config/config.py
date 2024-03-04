grids = [
    {
        "main_file": ["main"],
        "data_dir": ["word_use_data/childes/word"],
        "config": ["gpt2"],
        "n_epochs": [50],
        "batch_size": [16],
        "n_examples": [2, 3, 4, 5, 6][2:3],
        "eval_n_classes": ["2 3 4 5 6"],
        "loss_reduction": ["sum"],
        "lr": [3e-4, 1e-4, 3e-5][1:2],
        "weight_decay": [0.15],
        "factor": [0.1],
        "patience": [2],
        "seed": [0],
        "eval_seed": [0],
        "logging_step": [100],
    },
]
# ordered flags to display in job name
flags = [
    "data_dir",
    "config",
    "n_examples",
    "batch_size",
    "lr",
    "weight_decay",
    "seed",
]