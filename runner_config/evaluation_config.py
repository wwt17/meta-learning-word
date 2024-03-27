grids = [
    {
        "main_file": ["evaluation.py"],
        "data_dir": [r"word_use_data_clean/childes/word"],
        "pretrained_model": [
            f"ckpt/meta-word_data_dir_word_use_data_clean:childes:word_config_gpt2_concat_False_context_length_128_no_new_token_False_n_examples_{n_examples}_max_sample_times_0_batch_size_8_lr_0.0001_weight_decay_0.12_seed_0/best",
        ],
        "n_examples": [n_examples],
        "eval_n_classes": [(2, 3, 4, 5, 6)],
    }
    for n_examples in [3, 4, 5, 6, 7, 8, 9, 10]
]
# ordered flags to display in job name
flags = [
    "data_dir",
    "pretrained_model",
    "n_examples",
]