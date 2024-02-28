grids = [
    {
        "main_file": ["main"],
        "data_dir": ["word_use_data/childes/word"],
        "config": ["gpt2"],
        "n_epochs": [50],
        "batch_size": [16],
        "n_examples": [4],
        "eval_n_classes": [2, 3],
        "loss_reduction": ["sum"],
        "lr": [3e-4],
        "weight_decay": [0.1],
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
    "eval_n_classes",
    "batch_size",
    "lr",
    "weight_decay",
    "seed",
]