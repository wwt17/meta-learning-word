import os
import itertools

try:
    llama_path = os.environ["SCRATCH"]
except KeyError:
    llama_path = ""

grids = {
    "meta-trained_on_childes_with_gpt2_arch": [
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
    "meta_trained_with_pythia_arch": [
        {
            "main_file": ["evaluation.py"],
            "data_dir": [r"word_use_data/childes/word", r"word_use_data/babylm_data/babylm_10M/word", r"word_use_data/babylm_data/babylm_100M/word"][:2],
            "split": ["test"],
            "pretrained_model": list(itertools.chain.from_iterable(
                [
                    f"ckpt/meta-word_data_dir_word_use_data:childes:word_config_model_config:pythia-160m_concat_False_no_new_token_False_n_examples_{n_examples}_max_sample_times_0_batch_size_8_lr_0.0003_weight_decay_0.07_seed_{seed}/best",
                    f"ckpt/meta-word_data_dir_word_use_data:babylm_data:babylm_10M:word_config_model_config:pythia-160m_concat_False_no_new_token_False_n_examples_{n_examples}_max_sample_times_0_batch_size_8_lr_0.0003_weight_decay_0.15_seed_{seed}/best",
                    f"ckpt/meta-word_data_dir_word_use_data:babylm_data:babylm_100M:word_config_model_config:pythia-160m_concat_False_no_new_token_False_n_examples_{n_examples}_max_sample_times_0_batch_size_8_lr_0.0001_weight_decay_0.07_seed_{seed}/best",
                ][:2]
                for seed in [0, 1, 2]
            )),
            "n_examples": [n_examples],
            "eval_n_classes": [tuple(range(2, 11))],
            "print_decoded_prefix": [True],
        }
        for n_examples in [5, 10]
    ],
    "pretrained_LM": [
        {
            "main_file": ["evaluation.py"],
            "data_dir": [r"word_use_data/childes/word", r"word_use_data/babylm_data/babylm_10M/word", r"word_use_data/babylm_data/babylm_100M/word", r"chimeras.json"][:2],
            #"data_order": ["original"],
            "pretrained_model": [
                *(f"EleutherAI/pythia-{model_size}" for model_size in ['70m', '160m', '410m']),
                "gpt2",
                llama_path+r"/Meta-Llama-3-8B",
                llama_path+r"/Meta-Llama-3-8B-Instruct",
            ][-1:],
            #"revision": [f"step{step}" for step in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(range(1000, 143001, 1000)) if step <= 4000],
            "n_examples": list(range(2, 11)),
            "eval_n_classes": [tuple(range(2, 11))],
            "print_decoded_prefix": [True],
            "new_word": [" dax", " wug"][0:1],
            "prompt": [
                "",
                "The following lines are lowercased example sentences using a new word 'dax' in random order, one per line:",
                "The following lines are example sentences using a new word 'wug' in random order, one per line:",
            ][1:2],
            "sep": [" *"],
        }
    ],
    "pythia_finetuned": [
        {
            "main_file": ["evaluation.py"],
            "data_dir": [r"word_use_data/childes/word"],
            "pretrained_model": [
                f"ckpt/meta-word_pretrained_model_EleutherAI:pythia-160m_data_dir_word_use_data:childes:word_lm_False_embedding_init_mean_prompt__train_params_new_word_sep_n_examples_10_max_sample_times_0_batch_size_8_lr_0.0003_weight_decay_0.0_seed_0_eval_step_500/best",
                f"ckpt/meta-word_pretrained_model_EleutherAI:pythia-160m_data_dir_word_use_data:childes:word_embedding_init_mean_train_params_new_word_sep_prompt_n_examples_10_max_sample_times_0_batch_size_8_lr_3e-05_weight_decay_0.0_seed_0_eval_step_500/best",
            ][0:1],
            "tokenizer": [r"EleutherAI/pythia-160m"],
            "n_examples": list(range(10, 11)),
            "eval_n_classes": [tuple(range(2, 11))],
            "print_decoded_prefix": [True],
            "new_word": ["<|new_word|>"],
            "prompt": [
                "",
                "<|prompt_0|><|prompt_1|><|prompt_2|><|prompt_3|><|prompt_4|>",
            ][0:1],
            "sep": ["<|sep|>"],
            "add_tokens": [
                ("<|new_word|>", "<|sep|>"),
                ("<|new_word|>", "<|sep|>", "<|prompt_0|>", "<|prompt_1|>", "<|prompt_2|>", "<|prompt_3|>", "<|prompt_4|>"),
            ][0:1],
        }
    ],
    "Llama_finetuned": [
        {
            "main_file": ["evaluation.py"],
            "data_dir": [r"word_use_data/childes/word", r"word_use_data/babylm_data/babylm_10M/word", r"chimeras.json"][-1:],
            "data_order": ["original"],
            "pretrained_model": [
                f"ckpt/meta-word_pretrained_model_:scratch:ww2135:Meta-Llama-3-8B_data_dir_word_use_data:childes:word_embedding_init_mean_train_params_new_word_sep_n_examples_10_max_sample_times_0_batch_size_8_lr_0.0003_seed_0_eval_step_1000/best",
                f"ckpt/meta-word_pretrained_model_:scratch:ww2135:Meta-Llama-3-8B_data_dir_word_use_data:childes:word_embedding_init_mean_train_params_new_word_sep_n_examples_10_max_sample_times_0_batch_size_8_lr_0.0001_seed_0_eval_step_1000/best",
                f"ckpt/meta-word_pretrained_model_:scratch:ww2135:Meta-Llama-3-8B_data_dir_word_use_data:childes:word_embedding_init_mean_train_params_new_word_sep_prompt_n_examples_10_max_sample_times_0_batch_size_8_lr_0.001_seed_0/best",
                f"ckpt/meta-word_pretrained_model_:scratch:ww2135:Meta-Llama-3-8B_data_dir_word_use_data:childes:word_embedding_init_mean_train_params_new_word_sep_prompt_n_examples_10_max_sample_times_0_batch_size_8_lr_0.0003_seed_0_eval_step_1000/best",
                f"ckpt/meta-word_pretrained_model_:scratch:ww2135:Meta-Llama-3-8B_data_dir_word_use_data:childes:word_embedding_init_mean_train_params_new_word_sep_n_examples_5_train_max_length_80_batch_size_32_lr_0.003_seed_0_eval_step_1000/best",
                f"ckpt/meta-word_pretrained_model_:scratch:ww2135:Meta-Llama-3-8B_data_dir_word_use_data:babylm_data:babylm_10M:word_embedding_init_mean_train_params_new_word_sep_n_examples_5_train_max_length_160_batch_size_16_lr_0.001_seed_0_eval_step_1000/best",
            ][-2:],
            "tokenizer": [llama_path+r"/Meta-Llama-3-8B"],
            "n_examples": list(range(5, 6)),
            "eval_n_classes": [tuple(range(2, 11))],
            "print_decoded_prefix": [True],
            "new_word": ["<|reserved_special_token_0|>"],
            "prompt": [
                "",
                "<|reserved_special_token_2|><|reserved_special_token_3|><|reserved_special_token_4|><|reserved_special_token_5|><|reserved_special_token_6|>",
            ][0:1],
            "sep": ["<|reserved_special_token_1|>"],
        }
    ],
}["meta_trained_with_pythia_arch"]
# ordered flags to display in job name
flags = [
    "data_dir",
    "split",
    #"data_order",
    #"pretrained_model",
    #"revision",
    #"prompt",
    "n_examples",
]

syntactic = False
if syntactic:
    grids = [
        {
            "main_file": grid["main_file"],
            "data_dir": ["syntactic"],
            "split": [("dev", "test")],
            "pretrained_model": grid["pretrained_model"],
        }
        for grid in grids
    ]
    flags = [
        "data_dir",
    ]
