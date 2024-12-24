import os
import argparse
import copy
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


try:
    llama_path = os.environ["SCRATCH"]
except KeyError:
    llama_path = '.'
pretrained_models = {
    'Llama-3-8B': llama_path+r'/Meta-Llama-3-8B',
    'Llama-3-8B-Instruct': llama_path+r'/Meta-Llama-3-8B-Instruct',
}


syn_ctgs = ["n", "v", "adj", "adv"]
ctgs = ["ctg0", "ctg1"]
data_kinds = {
    'diff': '_different_{ctg}_{split}.txt'.format,
    'ident': '_identical_{ctg}_{split}.txt'.format,
}


def collect_results_from_file(file):
    results = {}

    for line in file:
        match = re.fullmatch(r"(?P<key>[\w-]+)=(|(\d+/\d+)=)(?P<value>[\d\.]*)%", line.strip())
        if match is None:
            continue
        results[match["key"]] = float(match["value"])

    if 'test_n_v_ctg0_diff_acc' in results:
        for split in ['dev', 'test']:
            for syn_ctg_pair in combinations(syn_ctgs, 2):
                for data_kind in data_kinds:
                    s = 0.
                    for ctg in ctgs:
                        name = f"{split}_{'_'.join(syn_ctg_pair)}_{ctg}_{data_kind}_acc"
                        s += results[name]
                    s /= len(ctgs)
                    results[f"{split}_{'_'.join(syn_ctg_pair)}_mean_{data_kind}_acc"] = s

    return results


def mean_std(a):
    return f"{np.mean(a):.1f}({np.std(a):.1f})"


def zipdicts(*ds):
    return {
        key: [d[key] for d in ds]
        for key in ds[0].keys()
        if all((key in d for d in ds))
    }


if __name__ == "__main__":
    filenames = [
        #f"ckpt/meta-word_data_dir_word_use_data:childes:word_config_model_config:pythia-160m_concat_False_no_new_token_False_n_examples_{n_examples}_max_sample_times_0_batch_size_8_lr_0.0003_weight_decay_0.07_seed_{seed}/best/{eval_name}/slurm.out"
        #f"ckpt/meta-word_data_dir_word_use_data:babylm_data:babylm_10M:word_config_model_config:pythia-160m_concat_False_no_new_token_False_n_examples_{c['n_examples']}_max_sample_times_0_batch_size_8_lr_0.0003_weight_decay_0.15_seed_{seed}/best/{eval_name}/slurm.out"
        f"ckpt/meta-word_pretrained_model_{pretrained_models['Llama-3-8B'].replace('/', ':')}_data_dir_{c['data_dir'].replace('/', ':')}_embedding_init_mean_train_params_new_word_sep_n_examples_{c['n_examples']}_train_max_length_{c['train_max_length']}_batch_size_{c['batch_size']}_lr_{c['lr']}_seed_{seed}_eval_step_1000/best/{eval_name}/slurm.out"
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
        ][2:3]
        for eval_name in [
            f"meta-word-eval_data_dir_{c['data_dir'].replace('/', ':')}_split_test_n_examples_{c['n_examples']}_max_new_tokens_100",
            "meta-word-eval_data_dir_syntactic",
        ][1:2]
        for seed in [0, 1, 2]
    ]
    results = []
    for filename in filenames:
        with open(filename) as file:
            res = collect_results_from_file(file)
            results.append(res)
            #print(res)
    res = zipdicts(*results)
    print(json.dumps(res, indent=2))
    res = {
        key: mean_std(values)
        for key, values in res.items()
    }
    print(json.dumps(res, indent=2))