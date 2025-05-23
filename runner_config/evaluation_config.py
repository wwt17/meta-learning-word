import os
import itertools

try:
    llama_path = os.environ["SCRATCH"]
except KeyError:
    llama_path = "."

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
            "data_dir": [r"word_use_data/childes/word", r"word_use_data/babylm_data/babylm_10M/word", r"word_use_data/babylm_data/babylm_100M/word", r"chimeras.json", r"def_task.json", r"oxford.json"][1:2],
            "split": ["test"],
            #"data_order": ["original"],
            "append_to_prefix": [
                '',
                ' The word{new_word} in the above sentence(s) is defined as "',
                ' What is the definition of{new_word}?',
            ][0:1],
            "append_to_prefix_for_gen": ["An example sentence using the word '{new_word}':"],
            "pretrained_model": [
                *(f"EleutherAI/pythia-{model_size}" for model_size in ['70m', '160m', '410m']),
                "gpt2",
                llama_path+r"/Meta-Llama-3-8B",
                llama_path+r"/Meta-Llama-3-8B-Instruct",
                r'Llama-2-7b-hf',
                r'Meta-Llama-3-8B-hf',
                'ltg/flan-t5-definition-en-base',
                'ltg/flan-t5-definition-en-large',
                'ltg/flan-t5-definition-en-xl',
            ][-5:-4],
            #"revision": [f"step{step}" for step in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(range(1000, 143001, 1000)) if step <= 4000],
            "emb_gen_model_type": ["college"],
            "n_examples": [2, 3, 4, 5, 10][-2:],
            "eval_n_classes": [(4, 8), ()][0:1],
            "print_decoded_prefix": [False],
            "new_word": [new_word],
            #"no_new_token": [True],
            "prompt": [
                "",
                f"The following lines are lowercased example sentences using a new word '{new_word.strip()}', one per line:",
            ][1:2],
            "sep": [" *", ""][:1],
            "max_new_tokens": [100],
            "num_beams": [1, 5][1:2],
        }
        for new_word in ["", " dax", " wug", " blicket"][1:2]
    ],
    "college": [
        {
            "main_file": ["evaluation.py"],
            "data_dir": [r"word_use_data/childes/word", r"word_use_data/babylm_data/babylm_10M/word", r"word_use_data/babylm_data/babylm_100M/word", r"chimeras.json", r"def_task.json", r"oxford.json"][1:2],
            "split": ["test"],
            #"data_order": ["original"],
            "append_to_prefix": [
                '',
                ' The word{new_word} is defined as "',
            ][0:1],
            "append_to_prefix_for_gen": ["A single example sentence using the word '{new_word}' (in one line):\n"],
            "pretrained_model": [r'Llama-2-7b-hf'],
            "emb_gen_model_type": ["college"],
            "n_examples": [2, 3, 4, 5, 10][-2:-1],
            "eval_n_classes": [(4, 8), ()][0:1],
            "print_decoded_prefix": [True],
            "new_word": ["<nonce>"],
            "sep": [""],
            "add_tokens": [("<nonce>",)],
            "max_new_tokens": [100],
            "num_beams": [1, 5][1:2],
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
            "data_dir": [c["data_dir"], r"chimeras.json", r"def_task.json", r"oxford.json"][1:2],
            #"split": ["test"],
            "data_order": ["original"],
            "append_to_prefix": [
                '',
                ' The word{new_word} in the above sentence(s) is defined as "',
            ][:1],
            "pretrained_model": [
                f"ckpt/meta-word_pretrained_model_{pretrained_model.replace('/', ':')}_data_dir_{c['data_dir'].replace('/', ':')}_embedding_init_mean{'' if prompt is None else '_prompt_'+prompt}_train_params_new_word_sep_n_examples_{c['n_examples']}_train_max_length_{c['train_max_length']}_batch_size_{c['batch_size']}_lr_{c['lr']}_seed_{seed}_eval_step_1000/best"
                for seed in [0, 1, 2]
            ],
            "tokenizer": [pretrained_model],
            "n_examples": [2, 3, 4, c["n_examples"]][-1:],
            "eval_n_classes": [tuple(range(2, 11)), (4, 8), ()][1:2],
            "print_decoded_prefix": [False],
            "new_word": ["<|reserved_special_token_0|>"],
            "prompt": [
                "",
                "<|reserved_special_token_2|><|reserved_special_token_3|><|reserved_special_token_4|><|reserved_special_token_5|><|reserved_special_token_6|>",
            ][0:1],
            "sep": ["<|reserved_special_token_1|>"],
            "max_new_tokens": [100],
            "num_beams": [1, 5][:1],
        }
        for pretrained_model, prompt in [
            (llama_path+r"/Meta-Llama-3-8B", None),
            (llama_path+r"/Meta-Llama-3-8B-Instruct", ""),
        ][-1:]
        for c in [
            {
                "data_dir": "word_use_data/childes/word",
                "n_examples": 5,
                "batch_size": 32,
                "lr": 3e-3,
                "train_max_length": 80,
            },
            {
                "data_dir": "word_use_data/childes/word",
                "n_examples": 10,
                "batch_size": 8,
                "lr": 3e-4,
                "train_max_length": 160,
            },
            {
                "data_dir": "word_use_data/babylm_data/babylm_10M/word",
                "n_examples": 5,
                "batch_size": 16,
                "lr": 1e-3,
                "train_max_length": 160,
            },
        ][-1:]
    ],
    "Llama-2_finetuned": [
        {
            "main_file": ["evaluation.py"],
            "data_dir": [c["data_dir"], r"chimeras.json", r"def_task.json", r"oxford.json"][0:1],
            #"split": ["test"],
            #"data_order": ["original"],
            "append_to_prefix": [
                '',
                ' The word{new_word} in the above sentence(s) is defined as "',
            ][0:1],
            "pretrained_model": [
                f"ckpt/meta-word_pretrained_model_{pretrained_model.replace('/', ':')}_data_dir_{c['data_dir'].replace('/', ':')}_n_examples_{c['n_examples']}_train_max_length_{c['train_max_length']}_batch_size_{c['batch_size']}_lr_{c['lr']}_seed_{seed}/best"
                for seed in [0, 1, 2]
            ],
            "tokenizer": [pretrained_model],
            "n_examples": [2, 3, 4, c["n_examples"]][:1],
            "eval_n_classes": [tuple(range(2, 11)), (4, 8), ()][-1:],
            "print_decoded_prefix": [False],
            "new_word": ["<|new_word|>"],
            "prompt": [
                ""
            ],
            "sep": ["<|sep|>"],
            "add_tokens": [("<|new_word|>", "<|sep|>")],
            "max_new_tokens": [100],
            "num_beams": [1, 5][1:],
        }
        for pretrained_model in [
            r"Llama-2-7b-hf"
        ]
        for c in [
            {
                "data_dir": "word_use_data/childes/word",
                "n_examples": 5,
                "batch_size": 32,
                "lr": 1e-3,
                "train_max_length": 80,
            },
            {
                "data_dir": "word_use_data/childes/word",
                "n_examples": 10,
                "batch_size": 8,
                "lr": 1e-3,
                "train_max_length": 160,
            },
            {
                "data_dir": "word_use_data/babylm_data/babylm_10M/word",
                "n_examples": 5,
                "batch_size": 16,
                "lr": 3e-3,
                "train_max_length": 160,
            },
        ][-1:]
    ],
}["Llama-2_finetuned"]
# ordered flags to display in job name
flags = [
    "data_dir",
    #"split",
    #"data_order",
    #"pretrained_model",
    #"revision",
    #"emb_gen_model_type",
    #"prompt",
    #"new_word",
    #"no_new_token",
    "n_examples",
    "max_new_tokens",
]

syntactic = False
if syntactic:
    grids = [
        {
            "data_dir": ["syntactic"],
            "split": [("dev", "test")],
            **{
                key: grid[key]
                for key in [
                    "main_file",
                    "pretrained_model",
                    "tokenizer",
                    "new_word",
                    "prompt",
                    "sep",
                    "add_tokens",
                ]
                if key in grid
            }
        }
        for grid in grids
    ]
    flags = [
        "data_dir",
        #"pretrained_model",
        #"prompt",
        #"new_word",
    ]
